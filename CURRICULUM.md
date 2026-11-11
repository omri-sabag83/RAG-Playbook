# RAG Playbook — Learning Program

**Goal:** a hands-on command of RAG (retrieval-augmented generation): a language
model answers a question using material retrieved from a document collection at
answer time, instead of relying only on what it learned in training. The aim is
to build a working RAG pipeline, explain every part of it, measure whether it
works, and know when it is the wrong tool. Every part is worked through on real
data.

**Starting point:** a Data Analyst background, with no prior work on language
models beyond having heard the term.

**Free and local:** both models run on the user's own computer through Ollama, a
free program that downloads and runs AI models. No paid service is used.

---

## Progress

`🟩🟩⬜⬜⬜⬜⬜⬜⬜` **22% complete (2/9 modules)**

⬜ Not started · 🟨 In progress · 🟩 Completed

| # | Module | Data | Status | Completed On |
|---|--------|------|--------|---------------|
| 1 | Why RAG Exists | MultiHop-RAG (shared) | 🟩 Complete | 2026-11-11 |
| 2 | Embeddings | MultiHop-RAG (shared) | 🟩 Complete | 2026-11-11 |
| 3 | Retrieval | MultiHop-RAG (shared) | ⬜ Not started | |
| 4 | Evaluating Retrieval | MultiHop-RAG (shared) | ⬜ Not started | |
| 5 | Chunking | MultiHop-RAG (shared) | ⬜ Not started | |
| 6 | Generation | MultiHop-RAG (shared) | ⬜ Not started | |
| 7 | Evaluating Answers | MultiHop-RAG (shared) | ⬜ Not started | |
| 8 | Failure Modes | MultiHop-RAG (shared) | ⬜ Not started | |
| 9 | Capstone: The Whole Pipeline | MultiHop-RAG (shared) | ⬜ Not started | |

---

## How this program works

**One shared dataset:** MultiHop-RAG (Tang and Yang, 2024), built to test RAG
systems. It has two parts:
- 609 news articles from late 2023, the documents to search
- 2,556 questions, each with its correct answer and the sentences in the
  articles that support the answer

Because every question comes with its correct answer and its supporting
sentences, each module can grade its own results instead of judging them by eye.

**Two models, both run locally:**
- an embedding model (`nomic-embed-text`), which turns a piece of text into a
  list of numbers, so that texts with similar meaning get similar numbers
- a language model (`llama3.2:3b`), which writes the answer

**Samples, stated:** the language model runs on the computer's processor, which
is slow. Modules that grade many answers use a fixed random sample of questions,
and each one states its size.

**The dataset can change per module:** if another dataset would teach a module
better, that module says so and explains why.

**Each module has:** concepts, why it matters, at least one worked example on
real numbers, and key takeaways. Findings are tagged Observation / Hypothesis /
Conclusion.

---

## Modules

## <u>Module 1 — Why RAG Exists</u>

**Concepts**
- What a language model knows: only the facts that were in its training text,
  up to a cutoff date
- Three ways to give it new facts: retrain it, paste the whole collection into
  the prompt, or search the collection and paste in only the matching documents
  (RAG, retrieval-augmented generation)
- The context window: the most text a model can read at once
- The five parts a RAG system is built from, each built in a later module

**Worked example:** the same questions answered with no articles, with the
right articles pasted in, and with the whole collection, which does not fit.
Retraining is explained, not run: it needs a graphics card.

## <u>Module 2 — Embeddings</u>

**Concepts**
- An embedding: a piece of text turned into a fixed-length list of numbers
- Cosine similarity: one number, from −1 to 1, for how close two embeddings are
- Why similar meaning lands close, and where it fails (negation, names, numbers)

**Worked example:** sentences from the articles, their embeddings, and which
pairs land close.

## <u>Module 3 — Retrieval</u>

**Concepts**
- The vector store: every chunk's embedding, kept in one table to search
- Top-k search: the k chunks closest to the question
- Keyword search (BM25, a word-matching score) vs. meaning search (embeddings)
- Hybrid search: combining the two rankings

**Worked example:** one question searched three ways, side by side.

## <u>Module 4 — Evaluating Retrieval</u>

**Concepts**
- What "found the right material" means: the retrieved chunks contain the
  labelled supporting sentences
- Recall@k: the share of supporting sentences found in the top k
- MRR (mean reciprocal rank): how high the first right chunk ranks
- Comparing search methods on the same questions

**Worked example:** keyword, meaning and hybrid search graded on the same
sample of questions.

## <u>Module 5 — Chunking</u>

**Concepts**
- Why documents are split before they are searched
- Chunk size, overlap between chunks, and splitting by sentences or paragraphs
- The trade-off: small chunks are precise but lose context, large chunks keep
  context but blur the match

**Worked example:** several split choices, each graded with Module 4's measures.

## <u>Module 6 — Generation</u>

**Concepts**
- Building the prompt: the question, the retrieved chunks, and the rules
- Grounding: answering only from the retrieved chunks
- Citing: each claim names the chunk it came from
- Saying "not enough information" when the chunks do not hold the answer

**Worked example:** answers built from the retrieved chunks, with their sources.

## <u>Module 7 — Evaluating Answers</u>

**Concepts**
- Correctness: does the answer match the labelled answer?
- Faithfulness: does the answer use only what was retrieved?
- Using a language model as a grader, and checking the grader by hand
- Separating a retrieval failure from a generation failure

**Worked example:** a sample of answers graded both ways, with the grader's
errors counted.

## <u>Module 8 — Failure Modes</u>

**Concepts**
- Questions that need several articles, comparisons, and dates
- Questions with no answer in the collection
- A relevant chunk lost in a long prompt
- When RAG is the wrong tool: counting and totals belong in a database query

**Worked example:** the pipeline's results split by question type, with the
failures traced to their cause.

## <u>Module 9 — Capstone: The Whole Pipeline</u>

**Concepts**
- The full pipeline end to end on new questions
- Which setting moved the results most
- A guide from the problem's shape to the right tool

**Worked example:** the full pipeline run and graded, with every setting from
Modules 3–8 compared in one table.
