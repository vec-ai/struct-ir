import json
from pathlib import Path
import os
import sys
from datetime import datetime

import torch


def printf(*args, **kwargs):
    print(datetime.now().isoformat(sep=" ", timespec='milliseconds'), '-', *args, **kwargs)


def parse_to_dict():
    kwargs = dict()
    args = sys.argv[1:]
    if len(args) == 0:
        return kwargs
    i = 0
    while i < len(args):
        if args[i].startswith('--'):
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                kwargs[args[i][2:]] = args[i + 1]
            elif '=' in args[i]:
                k, v = args[i].split('=')
                kwargs[k[2:]] = v
            else:
                kwargs[args[i][2:]] = True
        i += 1
    return kwargs


def read_jsonl(path) -> list:
    data = list()
    # printf('read_jsonl', path)
    with open(path) as file:
        for i, line in enumerate(file, start=1):
            obj = json.loads(line)
            data.append(obj)
    printf(f'{len(data)=}', path)
    return data


def encode_one(embed_func, corpus_path, path_ce, path_qe, qids=None, doc_type='json', batch_size=64, schema=None):
    def _batched_encode(data, is_query=True):
        batch_num = len(data) // batch_size + 1
        ids = list()
        embeddings = list()
        batch = list()
        for k, text in data.items():
            ids.append(k)
            batch.append(doc_transform(text))
            if len(batch) >= batch_size:
                if len(embeddings) % 50 == 0:
                    printf(f'Batch {len(embeddings)}/{batch_num}')
                chunk = embed_func(batch, is_query=is_query)
                embeddings.append(chunk)
                batch.clear()
        else:
            if len(batch) > 0:
                chunk = embed_func(batch, is_query=is_query)
                embeddings.append(chunk)
        embeddings = torch.cat(embeddings, dim=0)
        return ids, embeddings

    if doc_type == 'json':
        def doc_transform(text):
            return text

    elif doc_type == 'yaml':
        import yaml

        def doc_transform(text):
            return yaml.dump(json.loads(text), allow_unicode=True)

    elif doc_type == 'json_compact':
        def doc_transform(text):
            return json.dumps(json.loads(text), ensure_ascii=False)

    elif doc_type == 'json_schema':
        assert isinstance(schema, str), schema

        def doc_transform(text):
            return f"Schema:{schema}\n\n\nObject in this Schema:{text}"

    else:
        raise ValueError(f'Unknown doc_type: {doc_type}')

    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.inference_mode():
        # Encode queries
        if not os.path.exists(path_qe):
            queries = read_jsonl(corpus_path.replace('.corpus.', '.queries.test.'))
            qids_in_qrels = set(qids)
            queries = {i['_id']: i['text'] for i in queries if i['_id'] in qids_in_qrels}
            qids, q_embed = _batched_encode(queries, is_query=True)
            torch.save((qids, q_embed), path_qe)
            printf('Encode done', len(qids), path_qe)
        else:
            qids, q_embed = torch.load(path_qe, map_location=map_location)
        # Encode corpus
        if not os.path.exists(path_ce):
            corpus = read_jsonl(corpus_path)
            corpus = {i['_id']: i['text'] for i in corpus}
            doc_ids, c_embed = _batched_encode(corpus, is_query=False)
            torch.save((doc_ids, c_embed), path_ce)
            printf('Encode done', len(doc_ids), path_ce)
        else:
            doc_ids, c_embed = torch.load(path_ce, map_location=map_location)
    return (qids, q_embed), (doc_ids, c_embed)


def main(**kwargs):
    import models as embed_models
    from search import search_one

    # Arguments
    if 'model_name' not in kwargs:
        kwargs['model_name'] = 'drama'
    if 'qrels_name' not in kwargs:
        kwargs['qrels_name'] = 'qrels.test'
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 64
    kwargs['batch_size'] = int(kwargs['batch_size'])

    printf(kwargs)
    _func = getattr(embed_models, 'prepare_' + kwargs['model_name'].lower())

    if kwargs['model_name'].startswith('my'):
        assert 'model_path' in kwargs
        model_name = kwargs['model_path'].split('results/models/')[-1]
        kwargs['model_name'] = 'my/' + model_name
    elif 'model_path' in kwargs:
        model_name = kwargs['model_path'].split('results/models/')[-1]
        kwargs['model_name'] = kwargs['model_name'] + '/' + model_name

    _embed = _func(kwargs.pop('model_path')) if 'model_path' in kwargs else _func()
    node_num = int(kwargs.pop('node_num', 1))
    node_rank = int(kwargs.pop('node_rank', 0))
    subset = kwargs.pop('subset', None)

    path_list = sorted(str(p) for p in Path('./data/').glob('*/*.corpus.jsonl'))
    if subset:
        path_list = [p for p in path_list if subset in p]
    printf(f"{len(path_list)=}")

    model_name, qrels_name = kwargs.pop('model_name'), kwargs.pop('qrels_name')
    if 'doc_type' not in kwargs:
        kwargs['doc_type'] = 'json'
    doc_type, save_topk = kwargs["doc_type"], bool(kwargs.pop('save_topk', True))
    if doc_type == 'json_schema':
        with open('assets/merged_schemas.json') as f:
            all_schemas = json.load(f)
            printf(f.name)
    path_list = [corpus_path for _i, corpus_path in enumerate(path_list) if node_num < 2 or _i % node_num == node_rank]
    for _i, corpus_path in enumerate(path_list):
        printf(f"[{_i}/{len(path_list)}]")
        with open(corpus_path.replace('.corpus.', f'.{qrels_name}.')[:-1]) as f:
            qrels = json.load(f)
            printf(f'Read {len(qrels)=} {f.name}')
        path_ce = corpus_path.replace('data/', f'results/{model_name}/{doc_type}/').replace('.jsonl', '.pth')
        path_qe = path_ce.replace('.corpus.', f'.queries.{qrels_name}.')
        _dn = Path(path_ce).parent
        if not os.path.exists(_dn):
            os.makedirs(_dn, exist_ok=True)

        schema_str = None
        if doc_type == 'json_schema':
            domain, schema_name = corpus_path.split('/')[-2:]
            schema_name = schema_name.replace('.corpus.jsonl', '')
            if schema_name not in all_schemas[domain]['schemas']:
                domain = 'resume_search'
            schema = all_schemas[domain]['schemas'][schema_name]
            if 'common_fields' in all_schemas[domain]:
                common = all_schemas[domain]['common_fields'].copy()
                common.update(schema)
                schema = common
            schema_str = json.dumps(schema, ensure_ascii=False, indent=4)

        qe, ce = encode_one(_embed, corpus_path, path_ce, path_qe, list(qrels), schema=schema_str, **kwargs)
        result = search_one(qe, ce, qrels)
        result_heaps = result.pop('result_heaps')

        path_metrics = path_qe.replace('.pth', '.metrics.json')
        with open(path_metrics, 'w') as f:
            json.dump(result, f, indent=4)
        printf(f"{path_metrics=}")
        if save_topk:
            path_result = path_qe.replace('.pth', '.result.json')
            with open(path_result, 'w') as f:
                json.dump(result_heaps, f, indent=2)
            printf(f"{path_result=}")

    printf('exit')
    return


if __name__ == '__main__':
    main(**parse_to_dict())
