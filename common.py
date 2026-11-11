"""Shared loaders, model calls and table printing for every module.

load_corpus()     the 609 news articles, one row each
load_questions()  the 2,556 questions, one row each, with their evidence sentences
embed()           text -> embedding vectors (nomic-embed-text, through Ollama)
generate()        prompt -> answer text (llama3.2:3b, through Ollama)
show()            print a table with text columns left-aligned, numbers right-aligned

Both model calls are saved in data/cache/ (one SQLite file), so re-running a
notebook gives the same numbers without recomputing them. Ollama is started on
first use and stopped when the notebook's kernel exits.
"""
import atexit
import hashlib
import json
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"

# Chart palette: reference categorical slots (same as the earlier playbooks).
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#888888"
# Grade colours, used in every chart of right / did not answer / wrong answers:
# grade -> (bar colour, colour of the number written on the bar).
GRADE_COLORS = {"right": ("#2E8B57", "white"), "did not answer": ("#BBBBBB", "black"),
                "wrong": ("#E02424", "white")}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110})

# Every chart image gets a thin black frame (shared with the other repos: chart_style.py).
import chart_style  # noqa: E402,F401

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://127.0.0.1:11434"
SEED = 42


# ---------- data ----------

def _need(name: str) -> Path:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python data/get_data.py` first.")
    return path


def load_corpus() -> pd.DataFrame:
    """The 609 news articles: title, author, source, published_at, category, url, body."""
    df = pd.DataFrame(json.loads(_need("corpus.json").read_text()))
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def load_questions() -> pd.DataFrame:
    """The 2,556 questions: query, answer, question_type, evidence_list (list of dicts with a 'fact')."""
    return pd.DataFrame(json.loads(_need("MultiHopRAG.json").read_text()))


# ---------- Ollama: start on demand, stop at exit ----------

_server = None


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_URL + "/api/version", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _ensure_ollama() -> None:
    global _server
    if _ollama_up():
        return
    exe = shutil.which("ollama") or str(Path.home() / "ollama" / "ollama")
    _server = subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(_server.terminate)
    for _ in range(60):
        if _ollama_up():
            return
        time.sleep(0.5)
    raise RuntimeError("Ollama did not start: see README, Setup.")


def _post(path: str, body: dict) -> dict:
    _ensure_ollama()
    req = urllib.request.Request(OLLAMA_URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


# ---------- cache ----------

def _db() -> sqlite3.Connection:
    CACHE.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE / "model_calls.sqlite")
    con.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT)")
    return con


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# ---------- model calls ----------

def embed(texts: list[str], prefix: str = "", batch: int = 32) -> np.ndarray:
    """One embedding vector per text, as rows of a matrix.

    prefix: nomic-embed-text expects "search_query: " before a question and
    "search_document: " before a passage.
    """
    texts = [prefix + t for t in texts]
    con = _db()
    keys = [_key(EMBED_MODEL, t) for t in texts]
    found = dict(con.execute(f"SELECT key, value FROM calls WHERE key IN ({','.join('?' * len(keys))})", keys)) if keys else {}
    missing = [i for i, k in enumerate(keys) if k not in found]
    for s in range(0, len(missing), batch):
        idx = missing[s:s + batch]
        out = _post("/api/embed", {"model": EMBED_MODEL, "input": [texts[i] for i in idx]})["embeddings"]
        rows = [(keys[i], json.dumps(v)) for i, v in zip(idx, out)]
        con.executemany("INSERT OR REPLACE INTO calls VALUES (?, ?)", rows)
        con.commit()
        found.update(rows)
    con.close()
    return np.array([json.loads(found[k]) for k in keys])


def generate(prompt: str, system: str = "", max_tokens: int = 300, num_ctx: int = 4096,
             full: bool = False) -> str | dict:
    """The language model's answer to one prompt. Temperature 0 and a fixed seed, so it repeats.

    num_ctx: the context window to use, in tokens (word-pieces). Ollama silently drops
    the start of a longer prompt, so a prompt that fills the window raises an error here.
    full=True returns a dict: answer, prompt_tokens, prompt_seconds, answer_tokens, answer_seconds.
    """
    options = {"temperature": 0, "seed": SEED, "num_predict": max_tokens, "num_ctx": num_ctx}
    k = _key(LLM_MODEL, system, prompt, options)
    con = _db()
    row = con.execute("SELECT value FROM calls WHERE key = ?", (k,)).fetchone()
    if row is None:
        r = _post("/api/generate", {"model": LLM_MODEL, "prompt": prompt, "system": system,
                                    "stream": False, "options": options})
        out = {"answer": r["response"].strip(),
               "prompt_tokens": r.get("prompt_eval_count", 0),
               "prompt_seconds": r.get("prompt_eval_duration", 0) / 1e9,
               "answer_tokens": r.get("eval_count", 0),
               "answer_seconds": r.get("eval_duration", 0) / 1e9}
        con.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (k, json.dumps(out)))
        con.commit()
    else:
        out = json.loads(row[0])
    con.close()
    if out["prompt_tokens"] >= num_ctx - max_tokens:
        raise ValueError(f"Prompt ({out['prompt_tokens']} tokens) fills the {num_ctx}-token window: raise num_ctx.")
    return out if full else out["answer"]


def context_length() -> int:
    """The longest context window the language model was trained for, in tokens."""
    info = _post("/api/show", {"model": LLM_MODEL})["model_info"]
    return next(v for k, v in info.items() if k.endswith(".context_length"))


# ---------- tables ----------

def show(df: pd.DataFrame, index: bool = False) -> None:
    """Print a table with text columns left-aligned and numeric columns right-aligned."""
    import re
    if index:
        df = df.reset_index()
    numeric = re.compile(r"^[-+−]?[\d,]*\.?\d+%?$")
    cols = [str(c) for c in df.columns]
    cells = [[str(v) for v in df[c]] for c in df.columns]
    blank = {"n/a", "no data", "—", ""}
    is_text = [not all(numeric.match(v) or v in blank for v in col) or all(v in blank for v in col) for col in cells]
    widths = [max([len(h)] + [len(v) for v in col]) for h, col in zip(cols, cells)]
    pad = lambda s, w, left: s.ljust(w) if left else s.rjust(w)
    print("  ".join(pad(h, w, t) for h, w, t in zip(cols, widths, is_text)).rstrip())
    for r in range(len(df)):
        print("  ".join(pad(col[r], w, t) for col, w, t in zip(cells, widths, is_text)).rstrip())
