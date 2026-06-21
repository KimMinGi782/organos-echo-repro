# rag-retrieval-compression-benchmark-repro

Reproducible retrieval-compression benchmark for OrganOS Echo candidate selection on BEIR datasets.

## Method

Baseline:
- BM25

Candidate Compression:
- OrganOS Echo
- Branch = 512
- Depth = 12
- Width = 1

Evaluation:
- Recall@10
- nDCG@10

## BEIR Results

| Dataset | Recall@10 | nDCG@10 | Active | Compression |
|----------|----------|----------|----------|----------|
| SciFact | 0.6862 | 0.5597 | 22 | 235× |
| NFCorpus | 0.1240 | 0.2672 | 19 | 191× |
| FiQA | 0.2037 | 0.1591 | 124 | 465× |
| TREC-COVID | 0.0058 | 0.7136 | 12 | 14,277× |

## Reproduction

```bash
pip install -r requirements.txt
python run.py


# ============================================================
# ORGANOS PUBLIC REPRODUCIBLE BENCHMARK
# BM25 + Branch512 Echo
# ============================================================

!pip -q install beir rank-bm25

import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi

from beir import util
from beir.datasets.data_loader import GenericDataLoader

# ============================================================
# CONFIG
# ============================================================

DATASET = "scifact"

BRANCH = 512
DEPTH = 12
WIDTH = 1

TOP_K = 128
EVAL_K = 10

# ============================================================
# METRICS
# ============================================================

def recall_at_k(ranked_idx, rel_doc_ids, idx_to_doc_id, k=10):

    ranked_doc_ids = [
        idx_to_doc_id[i]
        for i in ranked_idx[:k]
    ]

    return (
        len(set(ranked_doc_ids) & set(rel_doc_ids))
        / max(1, len(rel_doc_ids))
    )

def ndcg_at_k(ranked_idx, rel_doc_ids, idx_to_doc_id, k=10):

    rel = set(rel_doc_ids)

    dcg = 0.0

    for rank, idx in enumerate(ranked_idx[:k]):

        if idx_to_doc_id[idx] in rel:
            dcg += 1.0 / np.log2(rank + 2)

    ideal = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(k, len(rel)))
    )

    return dcg / ideal if ideal > 0 else 0.0

# ============================================================
# ORGANOS ECHO
# ============================================================

def organos_echo(scores):

    current = np.argsort(scores)[::-1]

    echo_pools = {
        r: []
        for r in range(2, DEPTH + 1)
    }

    stages = 0

    while len(current) > TOP_K:

        stages += 1
        winners = []

        for i in range(0, len(current), BRANCH):

            block = current[i:i+BRANCH]

            if len(block) == 0:
                continue

            order = np.argsort(
                scores[block]
            )[::-1]

            winners.append(
                block[order[0]]
            )

            for r in range(2, DEPTH + 1):

                if len(order) >= r:
                    echo_pools[r].append(
                        block[order[r-1]]
                    )

        current = np.array(
            winners,
            dtype=np.int64
        )

    selected = [current]

    for r in range(2, DEPTH + 1):

        pool = np.array(
            echo_pools[r],
            dtype=np.int64
        )

        if len(pool) > 0:

            echo_top = pool[
                np.argsort(
                    scores[pool]
                )[::-1][:WIDTH]
            ]

            selected.append(
                echo_top
            )

    merged = np.concatenate(selected)

    final = merged[
        np.argsort(
            scores[merged]
        )[::-1]
    ][:TOP_K]

    return (
        final,
        len(merged),
        len(current),
        stages
    )

# ============================================================
# LOAD DATA
# ============================================================

url = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/"
    f"{DATASET}.zip"
)

data_path = util.download_and_unzip(
    url,
    "./datasets"
)

corpus, queries, qrels = (
    GenericDataLoader(
        data_folder=data_path
    ).load(split="test")
)

doc_ids = list(corpus.keys())

idx_to_doc_id = {
    i: did
    for i, did in enumerate(doc_ids)
}

texts = [

    (
        corpus[d].get("title","")
        + " "
        + corpus[d].get("text","")
    ).lower()

    for d in doc_ids
]

tokenized_docs = [
    t.split()
    for t in texts
]

bm25 = BM25Okapi(
    tokenized_docs
)

query_items = [

    (qid, qtext)

    for qid, qtext
    in queries.items()

    if qid in qrels
]

print("corpus:", len(corpus))
print("queries:", len(query_items))

# ============================================================
# RUN
# ============================================================

bm25_recalls = []
bm25_ndcgs = []

org_recalls = []
org_ndcgs = []

actives = []
mains = []
stages_list = []

for qid, qtext in query_items:

    rel_doc_ids = list(
        qrels[qid].keys()
    )

    scores = bm25.get_scores(
        qtext.lower().split()
    )

    ranked_full = np.argsort(
        scores
    )[::-1][:TOP_K]

    bm25_recalls.append(
        recall_at_k(
            ranked_full,
            rel_doc_ids,
            idx_to_doc_id,
            EVAL_K
        )
    )

    bm25_ndcgs.append(
        ndcg_at_k(
            ranked_full,
            rel_doc_ids,
            idx_to_doc_id,
            EVAL_K
        )
    )

    ranked_org, active, main, stages = (
        organos_echo(scores)
    )

    org_recalls.append(
        recall_at_k(
            ranked_org,
            rel_doc_ids,
            idx_to_doc_id,
            EVAL_K
        )
    )

    org_ndcgs.append(
        ndcg_at_k(
            ranked_org,
            rel_doc_ids,
            idx_to_doc_id,
            EVAL_K
        )
    )

    actives.append(active)
    mains.append(main)
    stages_list.append(stages)

# ============================================================
# REPORT
# ============================================================

print("="*80)
print("ORGANOS PUBLIC BENCHMARK")
print("="*80)

print("BM25 Recall@10 :",
      round(np.mean(bm25_recalls),4))

print("ORG Recall@10  :",
      round(np.mean(org_recalls),4))

print("BM25 nDCG@10   :",
      round(np.mean(bm25_ndcgs),4))

print("ORG nDCG@10    :",
      round(np.mean(org_ndcgs),4))

print("Avg Active     :",
      round(np.mean(actives),2))

print("Avg Main       :",
      round(np.mean(mains),2))

print("Avg Stages     :",
      round(np.mean(stages_list),2))

print("Compression    :",
      round(
          len(corpus) /
          np.mean(actives),
          2
      ),
      "x"
)