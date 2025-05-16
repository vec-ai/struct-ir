import glob
import json


with open('data/merged_schemas.json') as f:
    definitions = json.load(f)
resume_names = set(definitions['resume_search']['schemas'])
print(resume_names)

KEYS = ['recall_at_20', 'ndcg_at_10', 'recall_at_100']
HEAD = None

def compute_one(prefix, print_detail=False, subset=None):
    path_list = glob.glob(prefix + '/*/*' + result_pattern)
    if subset:
        path_list = [p for p in path_list if subset in p]
    if print_detail:
        print('path_list len', len(path_list))
        print(path_list[0])

    metrics = list()
    gathered = dict()
    for p in path_list:
        domain, schema = p.split('/')[-2:]
        schema = schema.split('.')[0]  # replace(result_pattern, '')  # '.queries' + 
        if schema in resume_names:
            domain = 'resume_search'
        with open(p, 'r') as f:
            try:
                result = json.load(f)
            except:
                print('!!!!!!!!!!!!!!!!!!', p)
                exit()
        scores = [result['metrics'][k] * 100 for k in KEYS]
        if print_detail:
            scores = ','.join(str(i) for i in [domain, schema, *scores])
        metrics.append(scores)
        if domain not in gathered:
            gathered[domain] = {k: list() for k in KEYS}
        for k in KEYS:
            gathered[domain][k].append(result['metrics'][k] * 100)

    if print_detail:
        for row in sorted(metrics):
            print(row)
    head = ['model', '#schemas', 'AVG_R-20', 'AVG_N-10', 'M-AVG_R-20', 'M-AVG_N-10']
    row = [prefix.replace('results/', ''), str(len(path_list))]
    row.append(str(sum(sum(v['recall_at_20']) for v in gathered.values()) / len(metrics)))
    row.append(str(sum(sum(v['ndcg_at_10']) for v in gathered.values()) / len(metrics)))
    row.append(str(sum((sum(v['recall_at_20']) / len(v['recall_at_20'])) for v in gathered.values()) / len(gathered)))
    row.append(str(sum((sum(v['ndcg_at_10']) / len(v['ndcg_at_10'])) for v in gathered.values()) / len(gathered)))
    for d, v in gathered.items():
        for k in ['recall_at_20', 'ndcg_at_10']:
            head.append(f'{d}-{k}')
            row.append(str(sum(v[k]) / len(v[k])))
    for d, v in gathered.items():
        k = 'recall_at_100'
        head.append(f'{d}-{k}')
        row.append(str(sum(v[k]) / len(v[k])))
    global HEAD
    head_str = ','.join(head)
    if HEAD is None or HEAD != head_str:
        print(head_str)
        HEAD = head_str
    print(','.join(row))


result_pattern = '.qrels-v1.test.metrics-domain.json'

# compute_one('results/drama/json/', print_detail=True)
# exit()

ps = [
    # 'results/bm25s/json',
    'results/instructor/json',
    'results/bge/json',
    'results/jina3/json',
    'results/nomic2/json',
    'results/drama/json',
    'results/qwen/json',
    'results/e5mistral/json',
    'results/gritlm/json',
    'results/nvembedv2/json',
]
for p in ps:
    compute_one(p)  # subset
