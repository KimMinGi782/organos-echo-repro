# organos-echo-repro
Reproducible benchmark for OrganOS Echo candidate compression on BEIR datasets.
# OrganOS Echo

Independent retrieval-compression research.

## BEIR Benchmark Results

| Dataset | Recall@10 | nDCG@10 | Compression |
|----------|----------|----------|----------|
| SciFact | 0.6862 | 0.5597 | 235× |
| NFCorpus | 0.1240 | 0.2672 | 191× |
| FIQA | 0.2037 | 0.1591 | 465× |
| TREC-COVID | 0.0058 | 0.7136 | 14,277× |

## Large Scale Simulation

100M Candidate Simulation

- Active Candidates: 12
- Compression: 8,333,333×
- Branch: 512
- Depth: 12
- Width: 1

## Status

Public benchmark repository.

More benchmark exports, screenshots, and reproducible code will be uploaded.