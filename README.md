# RAG Playbook

A hands-on playbook on RAG (retrieval-augmented generation): chunking,
embeddings, retrieval, grounded generation, evaluation and failure modes,
worked through on real data with a free, fully local stack.

The program, progress table and per-module detail are in
[CURRICULUM.md](CURRICULUM.md).

| # | Notebook | Topic |
|---|----------|-------|
| 1 | `01_why_rag_exists.ipynb` | Why RAG exists: the model alone, the right articles added, the whole collection, and retraining |
| 2 | `02_embeddings.ipynb` | Embeddings: turning text into numbers, cosine similarity, and where similar numbers hide different facts |
| 3 | — | Retrieval |
| 4 | — | Evaluating retrieval |
| 5 | — | Chunking |
| 6 | — | Generation |
| 7 | — | Evaluating answers |
| 8 | — | Failure modes |
| 9 | — | Capstone: the whole pipeline |

Related repos: [Causal Inference Playbook](https://github.com/omri-sabag83/Causal-Inference-Playbook),
[Scikit-Learn Playbook](https://github.com/omri-sabag83/Scikit-Learn-Playbook),
[Root-Cause Analysis Playbook](https://github.com/omri-sabag83/Root-Cause-Analysis-Playbook).

## Data

[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) (Tang and Yang, 2024):
609 news articles and 2,556 questions with their correct answers and supporting
sentences. Licence: ODC-BY. The articles are reproduced from their original news
outlets, so **the files are not in this repo**: `data/get_data.py` downloads a
fixed version and checks it against SHA-256 checksums. The notebooks keep their
outputs, so every number can be read without the data.

## Setup

Both models run locally through [Ollama](https://ollama.com), a free program
that downloads and runs AI models. Tested with Ollama v0.40.15 on macOS.

```bash
# 1. Ollama (any install method from ollama.com; tested: the command-line program from the GitHub release)
ollama pull nomic-embed-text        # embedding model, ~270 MB
ollama pull llama3.2:3b             # language model, ~2 GB

# 2. Python
conda activate base                 # tested env: Python 3.13.5, versions in requirements.txt
python data/get_data.py             # downloads and checksum-verifies the dataset
jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

The notebooks start Ollama when they need it and stop it when they finish.
Model outputs are saved in `data/cache/`, so a re-run repeats the same numbers.
