"""Download the MultiHop-RAG dataset into data/raw/ and verify it.

Usage (from the project root): python data/get_data.py

MultiHop-RAG (Tang and Yang, 2024) is licensed ODC-BY (Open Data Commons
Attribution). Its corpus reproduces news articles from their original outlets,
so the files are NOT in this repo (data/raw/ is gitignored). The notebooks keep
their outputs, so every number can still be read without them.

The URLs are pinned to one fixed version of the dataset on Hugging Face, and each
file is checked against a fixed SHA-256 checksum, so a changed file fails loudly.
"""
import hashlib
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"

# One fixed version (commit) of https://huggingface.co/datasets/yixuantt/MultiHopRAG
HF = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/"

# file name -> (URL, SHA-256, what it holds)
FILES = {
    "corpus.json": (HF + "corpus.json",
                    "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
                    "609 news articles: title, source, publish date, category, body"),
    "MultiHopRAG.json": (HF + "MultiHopRAG.json",
                         "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
                         "2,556 questions: question, correct answer, question type, evidence sentences"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RAW.mkdir(exist_ok=True)
    for name, (url, expected, _) in FILES.items():
        target = RAW / name
        if not target.exists():
            urllib.request.urlretrieve(url, target)
        got = sha256(target)
        if got != expected:
            raise ValueError(f"{name}: checksum {got} does not match the expected {expected}")
        print(f"ok  {name}")


if __name__ == "__main__":
    main()
