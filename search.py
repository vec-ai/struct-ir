import glob
import json
import os
import sys
from datetime import datetime

import pandas as pd
import pytrec_eval
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
            else:
                kwargs[args[i][2:]] = True
        i += 1
    return kwargs


def compute_metrics(qrels, results, k_values):
    from typing import Dict, List

    # https://github.com/beir-cellar/beir/blob/main/beir/retrieval/custom_metrics.py
    def mrr(qrels: Dict[str, Dict[str, int]],
            results: Dict[str, Dict[str, float]],
            k_values: List[int]
        ) -> Dict[str, float]:

        MRR = {}
        
        for k in k_values:
            MRR[f"mrr_at_{k}"] = 0.0
        
        k_max, top_hits = max(k_values), {}
        
        for query_id, doc_scores in results.items():
            # top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
            top_hits[query_id] = list(doc_scores.items())[0:k_max]

        for query_id, hits in top_hits.items():
            relevant_docs = set([doc_id for doc_id, score in qrels[query_id].items() if score > 0])
            for k in k_values:
                for rank, hit in enumerate(hits[0:k]):
                    if hit[0] in relevant_docs:
                        MRR[f"mrr_at_{k}"] += 1.0 / (rank + 1)
                        break

        for k in k_values:
            MRR[f"mrr_at_{k}"] = MRR[f"mrr_at_{k}"] / len(qrels)

        return MRR

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    # qrels = {str(qid): {str(docid): s for docid, s in v.items()} for qid, v in qrels.items()}
    # results = {str(qid): {str(docid): s for s, docid in v} for qid, v in result_heaps.items()}
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores_by_query = evaluator.evaluate(results)
    scores = pd.DataFrame.from_dict(scores_by_query.values()).mean()
    # metrics = mrr(qrels, results, k_values)
    metrics = dict()  # TODO
    for prefix in ('map_cut', 'ndcg_cut', 'recall', 'P'):
        name = 'precision' if prefix == 'P' else prefix.split('_')[0]
        for k in k_values:
            metrics[f'{name}_at_{k}'] = scores[f'{prefix}_{k}']
    return metrics


def cos_sim(a: torch.Tensor, b: torch.Tensor, do_norm=True):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    if do_norm:
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)

    return torch.mm(a, b.transpose(0, 1))


def read_jsonl(path) -> list:
    data = list()
    printf('read_jsonl', path)
    with open(path) as file:
        for i, line in enumerate(file, start=1):
            obj = json.loads(line)
            data.append(obj)
            if i % 50000 == 0:
                printf('Read', i)
    printf('len', len(data))
    return data


def _search(queries_embed_or_path, corpus_embed_or_path, top_k=100, result_heaps=None):
    import heapq

    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(queries_embed_or_path, str):
        query_ids, eq = torch.load(queries_embed_or_path, map_location=map_location)
        printf('Read', eq.size(0), queries_embed_or_path)
    else:
        query_ids, eq = queries_embed_or_path
        eq = eq.to(map_location)
    if isinstance(corpus_embed_or_path, str):
        doc_ids, ed = torch.load(corpus_embed_or_path, map_location=map_location)
        printf('Read', ed.size(0), corpus_embed_or_path)
    else:
        doc_ids, ed = corpus_embed_or_path
        ed = ed.to(map_location)

    scores = cos_sim(eq.float(), ed.float())
    scores[torch.isnan(scores)] = -1
    kth = min(top_k + 1, scores.size(1))
    top_k_values, top_k_idx = scores.topk(kth, dim=1, largest=True)
    top_k_idx, top_k_values = top_k_idx.cpu().tolist(), top_k_values.cpu().tolist()

    result_heaps = result_heaps or dict()
    for i, query_id in enumerate(query_ids):
        if query_id not in result_heaps:
            result_heaps[query_id] = list()
        for j, score in zip(top_k_idx[i], top_k_values[i]):
            doc_id = doc_ids[j]
            if len(result_heaps[query_id]) < top_k:
                # Push item on the heap
                heapq.heappush(result_heaps[query_id], (score, doc_id))
            else:
                # If item is larger than the smallest in the heap,
                # push it on the heap then pop the smallest element.
                # Tuples are compared by each item.
                heapq.heappushpop(result_heaps[query_id], (score, doc_id))

    return result_heaps


def search_one(
    queries_embed_or_path: str | tuple,
    corpus_embed_or_path: str | tuple,
    qrels: dict,
    k_values=[5, 10, 20, 50, 100],
    result_heaps=None,
    path_queries=None,
    force_search=False,
) -> dict:
    # TODO: k_values 500

    if not result_heaps or force_search:
        result_heaps = _search(
            queries_embed_or_path, corpus_embed_or_path, top_k=max(k_values),
            result_heaps=result_heaps
        )

    if len(qrels) != len(result_heaps):
        print(f'!!!!!!!!! {len(result_heaps)=}  {len(qrels)=}')
        return

    results = {q: {d: s for s, d in h} for q, h in result_heaps.items()}
    metrics = compute_metrics(qrels, results, k_values)
    printf(f"{metrics['recall_at_10']=:.4f} {metrics['recall_at_20']=:.4f} {metrics['recall_at_100']=:.4f}")

    obj = dict(metrics=metrics, result_heaps=result_heaps)
    if not path_queries:
        return obj

    queries = read_jsonl(path_queries)
    queries = {i['_id']: i for i in queries if i ['_id'] in qrels}
    result_by_filter, result_by_semantic = dict(), dict()
    qrels_by_filter, qrels_by_semantic = dict(), dict()
    for qid, r in qrels.items():
        nf, ns = json.loads(queries[qid]['cond'])[-2:]
        if nf not in result_by_filter:
            result_by_filter[nf] = dict()
            qrels_by_filter[nf] = dict()
        result_by_filter[nf][qid] = results[qid]
        qrels_by_filter[nf][qid] = r
        if ns not in result_by_semantic:
            result_by_semantic[ns] = dict()
            qrels_by_semantic[ns] = dict()
        result_by_semantic[ns][qid] = results[qid]
        qrels_by_semantic[ns][qid] = r
    metrics_by_filter = {
        k: compute_metrics(qrels_by_filter[k], result_by_filter[k], k_values) for k in qrels_by_filter
    }
    metrics_by_semantic = {
        k: compute_metrics(qrels_by_semantic[k], result_by_semantic[k], k_values) for k in qrels_by_semantic
    }

    obj.update(metrics_by_filter=metrics_by_filter, metrics_by_semantic=metrics_by_semantic)
    return obj


def main(**kwargs):
    # Arguments
    if 'result_path' not in kwargs:
        kwargs['result_path'] = 'results/drama/json'
    if 'qrels_name' not in kwargs:
        kwargs['qrels_name'] = 'qrels-v1.test'
    for k in ('save_topk', 'in_domain'):
        if k not in kwargs:
            kwargs[k] = False
        kwargs[k] = bool(kwargs[k])

    printf(kwargs)

    subset = kwargs.pop('subset', None)
    path_list = list()
    # path_list += sorted(glob.glob(f'{kwargs["result_path"]}/*.corpus.pth'))
    path_list += sorted(glob.glob(f'{kwargs["result_path"]}/*/*.corpus.pth'))
    # path_list += sorted(glob.glob(f'{kwargs["result_path"]}/*/*/*.corpus.pth'))
    # path_list += sorted(glob.glob(f'{kwargs["result_path"]}/*/*/*/*.corpus.pth'))
    # path_list += sorted(glob.glob(f'{kwargs["result_path"]}/*/*/*/*/*.corpus.pth'))
    if subset:
        path_list = [p for p in path_list if subset in p]
    printf(f"{len(path_list)=}")

    resume_names = {
        'project_manageme', 'finance_&_accoun', 'education_&_teac', 'software_enginee', 'sales_&_marketin',
        'data_science:_re', 'human_resources:', 'healthcare:_resu', 'engineering:_res', 'graphic_design:_'
    }
    qrels_name, save_topk, in_domain = kwargs['qrels_name'], kwargs['save_topk'], kwargs['in_domain']
    all_qrels = dict()
    for i, path_ce in enumerate(path_list):
        domain, schema = path_ce.split('/')[-2:]
        schema = schema.split('.')[0]
        if schema in resume_names:
            domain = 'resume_search'
        if domain not in all_qrels:
            all_qrels[domain] = dict()
        if schema not in all_qrels[domain]:
            path_qrels = f'data/{"human_resources" if schema in resume_names else domain}/{schema}.{qrels_name}.json'
            with open(path_qrels) as f:
                qrels = json.load(f)
                printf(f"Read {len(qrels)=}, {f.name}")
            all_qrels[domain][schema] = qrels
        else:
            qrels = all_qrels[domain][schema]

    i = 0
    all_embeds = dict()
    for domain in all_qrels:
        for schema in all_qrels[domain]:
            i += 1
            # data = read_jsonl(p)
            printf(f"[{i}/{len(path_list)}]")
            path_ce = f'{kwargs["result_path"]}/{"human_resources" if schema in resume_names else domain}/{schema}.corpus.pth'
            path_qe = path_ce.replace('.corpus.', f'.queries.{"qrels-fix.test"}.')
            if not os.path.exists(path_ce) or not os.path.exists(path_qe):
                printf('Not exists', path_ce, path_qe)
                continue
            qrels = all_qrels[domain][schema]
            path_result = path_ce.replace('.corpus.pth', f'.{qrels_name}.result.json')
            result_heaps = None
            _pr = path_ce.replace('.corpus.pth', '.queries.qrels-fix.test.result.json')
            if os.path.exists(_pr):
                with open(_pr) as f:
                    result_heaps = json.load(f)
                    result_heaps = {k: [tuple(i) for i in v] for k, v in result_heaps.items()}
            path_metrics = path_ce.replace('.corpus.pth', f'.{qrels_name}.metrics.json')
            if domain in all_embeds:
                eq, ed = all_embeds[domain][schema]
                result = search_one(eq, ed, qrels, result_heaps=result_heaps)
            else:
                result = search_one(path_qe, path_ce, qrels, result_heaps=result_heaps)
            result_heaps = result.pop('result_heaps')
            if not os.path.exists(path_metrics):
                with open(path_metrics, 'w') as f:
                    json.dump(result, f, indent=4)
                printf(f"Dump {path_metrics=}")
            if save_topk:
                with open(path_result, 'w') as f:
                    json.dump(result_heaps, f, indent=2)
                printf(f"Dump {path_result=}")
            if in_domain:
                path_metrics_domain = path_metrics.replace('.metrics.json', '.metrics-domain.json')
                if os.path.exists(path_metrics_domain):
                    continue
                if domain not in all_embeds:
                    other_embeds = dict()
                    map_location = 'cpu'
                    for other_schema in all_qrels[domain]:
                        other_path_qe = path_qe.replace(schema +'.', other_schema+'.')
                        query_ids, eq = torch.load(other_path_qe, map_location=map_location)
                        printf('Read', eq.size(0), other_path_qe)
                        other_path_ce = path_ce.replace(schema +'.', other_schema+'.')
                        doc_ids, ed = torch.load(other_path_ce, map_location=map_location)
                        printf('Read', ed.size(0), other_path_ce)
                        other_embeds[other_schema] = ((query_ids, eq), (doc_ids, ed))
                    all_embeds[domain] = other_embeds

                path_result_domain = path_result.replace('.result.json', '.result-domain.json')
                if os.path.exists(path_result_domain):
                    with open(path_result_domain) as f:
                        result_heaps = json.load(f)
                    result = search_one(path_qe, other_ce, qrels, result_heaps=result_heaps)
                    result_heaps = result.pop('result_heaps')
                else:
                    for other_schema in all_qrels[domain]:
                        if other_schema == schema:
                            continue
                        eq = all_embeds[domain][schema][0]
                        ed = all_embeds[domain][other_schema][1]
                        result = search_one(
                            eq, ed, qrels, result_heaps=result_heaps, force_search=True
                        )
                        result_heaps = result.pop('result_heaps')
                    if save_topk:
                        with open(path_result_domain, 'w') as f:
                            json.dump(result_heaps, f, indent=2)
                        printf(f"Dump {path_result_domain=}")
                with open(path_metrics_domain, 'w') as f:
                    json.dump(result, f, indent=4)
                printf(f"Dump {path_metrics_domain=}")
    printf('exit')
    return


if __name__ == '__main__':
    main(**parse_to_dict())
