# PaperMind AI

**AI-powered research paper analyzer.** Upload a PDF and get a structured summary, key findings, semantic search and question answering, with the exact supporting passage and page number beside every answer. When the paper does not contain enough evidence, PaperMind AI says so instead of guessing.

The machine-learning core is a **fine-tuned MiniLM bi-encoder** for question-to-passage retrieval, trained on Google Colab (T4 GPU) and served on a CPU-only laptop.

---

## Table of contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [The ML task and model](#the-ml-task-and-model)
4. [Results](#results)
5. [Project structure](#project-structure)
6. [Setup](#setup)
7. [Using the app](#using-the-app)
8. [API reference](#api-reference)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Security](#security)
12. [Limitations](#limitations)
13. [Future work](#future-work)
14. [Troubleshooting](#troubleshooting)
15. [References](#references)

---

## Features

| Feature | Description |
|---|---|
| PDF upload and processing | Validated upload, text extraction with page numbers, section-aware passages |
| Structured summary | Research problem, objectives, methodology, results and conclusions |
| Ask your paper | Natural-language questions answered from the paper, with evidence |
| Semantic search | Finds passages by meaning, not only exact keywords |
| Evidence and page references | Supporting excerpt, section, page number and similarity score for every result |
| Key information extraction | Objectives, methodologies, findings, limitations and future directions |
| Hallucination mitigation | Detects insufficient evidence and says so |
| Exportable analysis | Download the summary, findings and Q&A log as Markdown or HTML |

**No text is generated.** Every answer, summary sentence and key finding is a span copied from the paper, so the system cannot invent facts.

---

## How it works

```
React (Vite) ──/api──▶ Flask ──▶ PyMuPDF (text + pages) ─▶ section-aware passages
                                   │
                                   ├─▶ fine-tuned MiniLM bi-encoder ─▶ FAISS index (per paper)
                                   ├─▶ extractive reader (MiniLM, SQuAD 2.0) ─▶ answer span + evidence
                                   └─▶ SQLite (papers, passages, Q&A history)
```

**From PDF to answer**

1. The user uploads a PDF. The server checks it and extracts text with page numbers.
2. Text is cleaned and split into passages of about 140 words. Two-column papers are read column by column, headings give each passage a section, and passages never cross a page boundary.
3. The fine-tuned encoder embeds every passage and the vectors are stored in a per-paper FAISS index.
4. A structured summary and key findings are extracted from the passages.
5. For a question, the encoder embeds it and retrieves the nearest passages.
6. **Gate 1:** if the best similarity is below the threshold (0.255 for the fine-tuned model), the app answers "not enough evidence".
7. **Gate 2:** otherwise a reader model picks an answer span. If it prefers "no answer", the app shows the relevant passages without an answer.
8. The answer is shown with its passages, page numbers and similarity scores.

---

## The ML task and model

**Task:** question-to-passage semantic retrieval. Given a question, rank a paper's passages so the one containing the answer comes first. The same similarity score decides when evidence is insufficient.

**Base model:** `sentence-transformers/all-MiniLM-L6-v2`, an encoder-only transformer distilled from a larger model.

| Property | Value |
|---|---|
| Layers | 6 |
| Hidden size (embedding dimension) | 384 |
| Attention heads | 12 (32 dimensions each) |
| Feed-forward size | 1,536 |
| Vocabulary | 30,522 WordPiece tokens (uncased) |
| Max input length used | 256 tokens |
| Parameters | 22,713,216 |

Text is encoded as: tokens, then 6 transformer layers, then mean pooling, then L2 normalisation, giving a 384-dimensional unit vector. Similarity is cosine similarity, computed as a dot product.

**Training (Colab, Tesla T4)**

| Setting | Value |
|---|---|
| Loss | MultipleNegativesRankingLoss (in-batch negatives), scale 20 |
| Batch sampler | No-duplicates (avoids false negatives) |
| Epochs / batch size | 2 / 64 |
| Learning rate | 3e-5, 10% warmup, linear decay |
| Optimiser / precision | AdamW / fp16 mixed precision |
| Seed | 42 |
| Training time | 10.23 minutes |

**Data**

| Split | Size | Source |
|---|---|---|
| Train | 88,872 pairs | SQuAD v1.1 train (minus dev articles) + SciQ train |
| Dev | 9,203 pairs | SQuAD dev articles + SciQ validation |
| Test: SQuAD | 10,570 questions | SQuAD v1.1 validation |
| Test: SciQ | 826 questions | SciQ test (passages over 80 characters) |

SQuAD is split by article so no paragraph appears on both sides of a split. Test sets were never used for training or for choosing the threshold.

**Reader (not trained here):** `deepset/minilm-uncased-squad2` extracts an answer span and can abstain.

---

## Results

All numbers come from `reports/training_report.json` produced by the Colab notebook.

### Passage retrieval (held-out test sets)

**SQuAD test (10,570 questions)**

| Model | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| TF-IDF | 0.7043 | 0.8518 | 0.8908 | 0.9305 | 0.7856 | 0.8210 |
| BM25 | **0.7467** | **0.8723** | **0.9053** | 0.9375 | **0.8160** | **0.8457** |
| MiniLM (pretrained) | 0.6251 | 0.8077 | 0.8640 | 0.9172 | 0.7270 | 0.7734 |
| MiniLM (fine-tuned) | 0.6774 | 0.8435 | 0.8954 | **0.9395** | 0.7694 | 0.8109 |

**SciQ test (826 questions)**

| Model | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| TF-IDF | 0.8753 | 0.9552 | 0.9734 | **0.9806** | 0.9164 | 0.9324 |
| BM25 | **0.9140** | **0.9673** | **0.9770** | **0.9806** | **0.9420** | **0.9518** |
| MiniLM (pretrained) | 0.7954 | 0.9274 | 0.9516 | 0.9661 | 0.8628 | 0.8886 |
| MiniLM (fine-tuned) | 0.8680 | 0.9588 | 0.9709 | 0.9794 | 0.9143 | 0.9306 |

**What the results say**

- Fine-tuning improved every metric over the pretrained model. Recall@1 rose by 5.2 points (SQuAD) and 7.3 points (SciQ); MRR@10 by 4.2 and 5.1.
- BM25 still ranked passages better than the fine-tuned model on these benchmarks. Plausible reasons (not tested): SQuAD questions share many exact words with their passage, each benchmark ranks thousands of passages, and the fine-tuning was short with one seed.
- Each result is from a single training run, so differences of about one point on the small SciQ test set should not be over-read.

### Evidence-sufficiency threshold

The threshold was chosen on the dev set (a question with its true passage versus a random other passage, best F1), then evaluated on the test sets.

| Model | Test set | Threshold | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|---|
| MiniLM (pretrained) | SQuAD | 0.270 | 0.9736 | 0.9790 | 0.9679 | 0.9735 | 0.9958 |
| MiniLM (pretrained) | SciQ | 0.270 | 0.9582 | 0.9366 | 0.9831 | 0.9592 | 0.9909 |
| MiniLM (fine-tuned) | SQuAD | **0.255** | 0.9802 | 0.9810 | 0.9794 | 0.9802 | 0.9972 |
| MiniLM (fine-tuned) | SciQ | **0.255** | 0.9576 | 0.9325 | 0.9867 | 0.9588 | 0.9919 |

**Caveat:** the negatives are random passages, which are easy to reject. Real unanswerable questions about the same paper are closer in topic, so these accuracies are optimistic. The threshold can be changed (see [Configuration](#configuration)).

---

## Project structure

```
papermind-ai/
├── README.md
├── colab/
│   └── PaperMind_Training.ipynb      # fine-tuning, baselines, evaluation, threshold, export
├── backend/
│   ├── app.py                        # Flask API, upload validation, routes
│   ├── config.py                     # settings and environment variables
│   ├── pdf_processing.py             # PyMuPDF extraction, reading order, passages
│   ├── ml.py                         # fine-tuned encoder and extractive reader
│   ├── retrieval.py                  # per-paper FAISS index
│   ├── qa.py                         # retrieval, threshold, answer logic
│   ├── analysis.py                   # structured summary and key findings
│   ├── db.py                         # SQLite schema and queries
│   ├── report.py                     # Markdown / HTML report export
│   ├── requirements.txt
│   ├── models/
│   │   └── papermind-minilm/         # fine-tuned model from Colab goes here
│   └── tests/
│       └── test_pipeline.py          # end-to-end API test with stub models
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx
        ├── api.js
        ├── main.jsx
        └── styles.css
```

Created at runtime and not committed: `backend/data/` (uploads, indexes, SQLite database), `backend/.venv` or `backend/venv`, `frontend/node_modules`.

---

## Setup

**Requirements:** Python 3.10 or newer, Node.js 18 or newer, a Google account (for Colab). No GPU is needed on your laptop.

### Step 1: Train on Google Colab (about 20 minutes)

1. Open <https://colab.research.google.com>, choose **File, Upload notebook**, and select `colab/PaperMind_Training.ipynb`.
2. Choose **Runtime, Change runtime type, T4 GPU**.
3. Choose **Runtime, Run all**. For a quick smoke test, first set `MAX_TRAIN_PAIRS = 5000` in the configuration cell, then set it back to `None` for the real run.
4. Run one of the last two cells (browser download or copy to Google Drive) to get `papermind_colab_outputs.zip`.
5. Unzip it on your laptop:
   - copy the folder `papermind-minilm/` to `backend/models/papermind-minilm/`
   - keep the `reports/` folder (metrics, plots, `training_report.json`)

The backend still runs without this step by falling back to the pretrained MiniLM, but then it uses the untuned model and a default threshold.

### Step 2: Backend

```bash
cd backend
python -m venv .venv

# activate the environment
#   Windows PowerShell:  .venv\Scripts\Activate.ps1
#   Windows cmd:         .venv\Scripts\activate.bat
#   macOS / Linux:       source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

The API starts on <http://127.0.0.1:5000>. Open `/api/health`; it should report `"source": "fine-tuned"` when your model folder is in place. The reader model (about 130 MB) is downloaded from Hugging Face on first use and cached.

> The Sentence-Transformers version used for inference must be at least the version used for training (recorded in `training_report.json`). `requirements.txt` pins `sentence-transformers>=6.2`.

### Step 3: Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. During development Vite proxies `/api` to the Flask server.

---

## Using the app

1. Click **Add a paper** (or drag a PDF onto the window). Analysis takes roughly 20 to 60 seconds for a 10 to 15 page paper on a CPU.
2. **Summary** shows the abstract and five structured fields.
3. **Key information** lists objectives, methodologies, findings, limitations and future directions.
4. **Ask your paper** answers factual questions, for example "Which dataset was used?". Each answer is one of:
   - **Answered:** a span from the paper, highlighted inside its evidence.
   - **Relevant passages only:** evidence found, but no specific span to extract.
   - **Not enough evidence in this paper:** similarity was below the threshold.
5. **Semantic search** ranks passages by meaning and shows page, section and similarity.
6. **Export** downloads the report as Markdown or HTML (open the HTML and print to PDF).

Answers come from a SQuAD-style extractive reader, so specific factual questions work best. Open-ended questions ("explain the contribution") usually return passages only.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Status and model information |
| POST | `/api/papers` | Upload a PDF (multipart field `file`) |
| GET | `/api/papers` | List papers |
| GET | `/api/papers/<id>` | Paper with its analysis |
| DELETE | `/api/papers/<id>` | Delete the paper, file, index and history |
| POST | `/api/papers/<id>/ask` | Body `{"question": "..."}`; returns status, answer and evidence |
| POST | `/api/papers/<id>/search` | Body `{"query": "...", "k": 5}`; returns ranked passages |
| GET | `/api/papers/<id>/history` | Previous questions and answers |
| GET | `/api/papers/<id>/report?format=md\|html` | Download the report |

`<id>` is a 32-character hexadecimal identifier. Questions are limited to 500 characters.

---

## Configuration

Set environment variables before starting the backend.

| Variable | Default | Meaning |
|---|---|---|
| `PAPERMIND_THRESHOLD` | from `papermind_config.json` | Minimum cosine similarity to attempt an answer. Raise for stricter refusals, lower for fewer. Overrides the calibrated value. |
| `PAPERMIND_MODEL_DIR` | `backend/models/papermind-minilm` | Fine-tuned model folder |
| `PAPERMIND_BASE_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Fallback model |
| `PAPERMIND_READER_MODEL` | `deepset/minilm-uncased-squad2` | Extractive reader |
| `PAPERMIND_MAX_UPLOAD_MB` | 25 | Maximum upload size |
| `PAPERMIND_MAX_PAGES` | 80 | Maximum pages per PDF |
| `PAPERMIND_DATA` | `backend/data` | Storage for uploads, indexes and the database |
| `PAPERMIND_CORS_ORIGINS` | local Vite origins | Allowed front-end origins |

PowerShell example: `$env:PAPERMIND_THRESHOLD="0.35"; python app.py`

If summaries come back sparse with the real models, lower `MIN_SENTENCE_SIM` in `backend/analysis.py`.

---

## Testing

```bash
cd backend
python -m pytest tests -q
```

The test runs the whole API with small stand-in models (no downloads, no GPU). It covers upload validation, section detection, summary and key-info extraction, answered and insufficient-evidence questions, input validation, search, history, report export, index reload after restart and deletion.

---

## Security

- Uploads are checked for extension, the `%PDF-` signature, size and page count.
- Files are stored under random server-generated names and are never executed.
- All SQL queries are parameterised.
- CORS is restricted to the local front end.
- The Flask debugger is off.
- React escapes all displayed text.
- No API keys exist anywhere in the project.
- Deleting a paper removes its PDF, index and database rows.

The application is designed for local, single-user use. Add authentication before exposing it to a network.

---

## Limitations

- The training data is Wikipedia and science-exam text, not full research papers, and answer quality has not yet been measured on a set of real papers.
- Results come from one training run with one seed and one hyperparameter setting.
- The evidence threshold was calibrated with random negative passages, so its accuracy is optimistic.
- Answers are short extracted spans; open-ended questions return passages only.
- Scanned PDFs are not supported (no OCR). Tables and complex layouts can produce noisy passages.
- Summary and key-info items rely on similarity plus cue phrases; limitations and future work appear only if the paper contains matching phrases.
- PyMuPDF is AGPL-licensed. This is fine for coursework; check its terms before distributing the app.

---

## Future work

- Hybrid retrieval combining BM25 and dense scores.
- Hard negatives taken from the same paper, and threshold recalibration on them.
- Fine-tuning on scientific question-answering data and evaluation on real papers.
- Multiple seeds and an ablation of epochs and learning rate.
- A cross-encoder re-ranker for the top passages, and OCR for scanned documents.
- Multi-paper comparison, quizzes and revision notes, citation extraction, PDF reports.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Colab says no GPU | Runtime, Change runtime type, T4 GPU, then run all again |
| Colab out of memory | Lower `BATCH_SIZE` to 32 or `MAX_SEQ_LEN` to 192 |
| `No module named 'sentence_transformers.base.modules.normalize'` | Your Sentence-Transformers is older than the one used in Colab. Run `pip install -U sentence-transformers` |
| `/api/health` says `pretrained-baseline` | The model folder is not at `backend/models/papermind-minilm/` (it must contain `modules.json`) |
| Frontend shows "Cannot reach the backend" | Start `python app.py` in `backend/` and keep it running |
| Upload takes long | Analysis runs on CPU; 20 to 60 seconds for a 10 to 15 page paper is normal |
| Port 5000 busy (macOS AirPlay) | Change the port in `app.py` and the proxy in `frontend/vite.config.js` |
| `faiss-cpu` or `torch` fails to install | Use a fresh virtual environment, ideally Python 3.11 or 3.12 |

---

## References

1. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS, 2020.
2. S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in Information Retrieval, 2009.
3. N. Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," NeurIPS Datasets and Benchmarks, 2021.
4. V. Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering," EMNLP, 2020.
5. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP-IJCNLP, 2019.
6. J. Johnson, M. Douze and H. Jégou, "Billion-scale similarity search with GPUs," IEEE Transactions on Big Data, 2021.
7. A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.
8. J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL-HLT, 2019.
9. W. Wang et al., "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers," NeurIPS, 2020.
10. M. Henderson et al., "Efficient Natural Language Response Suggestion for Smart Reply," arXiv:1705.00652, 2017.
11. P. Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text," EMNLP, 2016.
12. P. Rajpurkar, R. Jia and P. Liang, "Know What You Don't Know: Unanswerable Questions for SQuAD," ACL, 2018.
13. J. Welbl, N. F. Liu and M. Gardner, "Crowdsourcing Multiple Choice Science Questions," W-NUT, 2017.
14. Z. Ji et al., "Survey of Hallucination in Natural Language Generation," ACM Computing Surveys, 2023.
15. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019.
16. P. Micikevicius et al., "Mixed Precision Training," ICLR, 2018.
17. Model cards on Hugging Face: `sentence-transformers/all-MiniLM-L6-v2` and `deepset/minilm-uncased-squad2`.

---

## Author and license

**Author:** [Your name, roll number, institution]

**License:** [Add a license, for example MIT, before publishing]