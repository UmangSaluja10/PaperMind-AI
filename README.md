# PaperMind AI — setup guide

AI-powered research-paper analyzer: upload a PDF → structured summary, key information, semantic search, grounded Q&A with page references, "not enough evidence" detection, and a downloadable report.

## 0. How the work is split

| Where | What runs there | Why |
|---|---|---|
| **Google Colab (T4 GPU)** | `colab/PaperMind_Training.ipynb` — fine-tunes the transformer, evaluates it against baselines, saves the model | Training needs a GPU |
| **Your laptop, VS Code (CPU only)** | Flask backend + React frontend | Inference with small models (MiniLM, ~22M params) is fine on CPU |

```
React (Vite) ──/api──▶ Flask ──▶ PyMuPDF (text + pages) ─▶ section-aware passages
                                   │
                                   ├─▶ fine-tuned MiniLM bi-encoder ─▶ FAISS index (per paper)
                                   ├─▶ extractive reader (MiniLM-SQuAD2) ─▶ answer span + evidence
                                   └─▶ SQLite (papers, passages, Q&A history)
```

**The ML task you fine-tune:** question → passage semantic retrieval (bi-encoder, contrastive loss). The retrieval similarity score also drives the "insufficient evidence" decision, using a threshold calibrated on a dev set.

Nothing in the app *generates* free text, so it cannot invent facts: every answer, summary sentence and key-info item is a span copied from the paper, shown with its page number.

## 1. Prerequisites (laptop)

- **Python 3.10–3.12** (3.11 recommended; some ML wheels lag behind the newest Python)
- **Node.js 18+**
- **VS Code** with the *Python* extension
- A Google account (for Colab)

## 2. Step A — Train on Colab (do this first, ~20 min)

1. Go to <https://colab.research.google.com> → **File → Upload notebook** → choose `colab/PaperMind_Training.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.** (Nothing to paste: every cell is already written. Settings you may change are in section 2 of the notebook: epochs, batch size, learning rate, `MAX_TRAIN_PAIRS`.)
   - First time, set `MAX_TRAIN_PAIRS = 5000` for a ~3-minute smoke test, then set it back to `None` for the real run.
4. At the end, run **one** of the last two cells (browser download, or copy to Google Drive). You get `papermind_colab_outputs.zip`.
5. Unzip it on your laptop:
   - copy the folder **`papermind-minilm/`** → `papermind-ai/backend/models/papermind-minilm/`
   - keep **`reports/`** (metrics table, comparison plots, `training_report.json`) for your CA2 write-up.

> The backend still works without this step (it falls back to the pretrained MiniLM), but then you have no fine-tuned model to show for the project.

## 3. Step B — Backend (VS Code terminal, on your laptop)

Open the `papermind-ai` folder in VS Code (**File → Open Folder**), then open a terminal (**Terminal → New Terminal**):

```bash
cd backend

# create + activate a virtual environment
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (cmd):          .venv\Scripts\activate.bat
# macOS / Linux:          source .venv/bin/activate

pip install -r requirements.txt

# optional but recommended: check the plumbing (uses tiny stub models, no downloads)
python -m pytest tests -q

python app.py          # -> http://127.0.0.1:5000
```

In VS Code press **Ctrl+Shift+P → "Python: Select Interpreter" →** pick `backend/.venv`.

First run: the reader model `deepset/minilm-uncased-squad2` (~130 MB) is downloaded from Hugging Face on the first question, then cached. Check `http://127.0.0.1:5000/api/health` — it should say `"source": "fine-tuned"`.

## 4. Step C — Frontend (second terminal)

```bash
cd frontend
npm install
npm run dev            # -> http://localhost:5173
```

Open <http://localhost:5173>, click **Upload PDF**. Keep both terminals running.

## 5. What each file does

| File | Purpose |
|---|---|
| `colab/PaperMind_Training.ipynb` | Data prep, baselines (TF-IDF, BM25, pretrained MiniLM), fine-tuning, evaluation, threshold calibration, plots, model export |
| `backend/app.py` | Flask API, upload validation, routes |
| `backend/pdf_processing.py` | PyMuPDF text extraction, header/footer removal, section detection, passage chunking with page numbers |
| `backend/ml.py` | Loads your fine-tuned bi-encoder + the extractive reader |
| `backend/retrieval.py` | FAISS index per paper |
| `backend/qa.py` | Retrieve → threshold check → read span → `answered` / `excerpts_only` / `insufficient_evidence` |
| `backend/analysis.py` | Structured summary + key-info extraction (extractive, with page refs) |
| `backend/db.py` | SQLite tables and queries |
| `backend/report.py` | Markdown / HTML report |
| `backend/tests/test_pipeline.py` | End-to-end API test with stub models |
| `frontend/src/App.jsx`, `api.js`, `styles.css` | The whole UI |

## 6. Configuration (optional, environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PAPERMIND_THRESHOLD` | from `papermind_config.json` | Minimum cosine similarity to attempt an answer. Raise → more refusals, lower → fewer. Overrides the calibrated value. |
| `PAPERMIND_MODEL_DIR` | `backend/models/papermind-minilm` | Fine-tuned model folder |
| `PAPERMIND_READER_MODEL` | `deepset/minilm-uncased-squad2` | Extractive reader |
| `PAPERMIND_MAX_UPLOAD_MB` / `PAPERMIND_MAX_PAGES` | 25 / 80 | Upload limits |

Example (PowerShell): `$env:PAPERMIND_THRESHOLD="0.35"; python app.py`

## 7. Mapping to the CA2 requirements

- **Transformer concept:** encoder-only transformer (self-attention, 6 layers, 384 hidden, ~22.7M params), bi-encoder with mean pooling, contrastive (Multiple-Negatives-Ranking) loss, fine-tuning from a pretrained checkpoint.
- **Defined task:** question–passage retrieval; *not* "the whole app is trained from scratch".
- **Baseline + held-out evaluation:** TF-IDF, BM25 and the pretrained model vs the fine-tuned model on SQuAD-validation and SciQ-test (never seen in training); Recall@k, MRR@10, nDCG@10. The threshold is calibrated on a separate dev split and evaluated on test (accuracy, F1, AUC).
- **Reproducibility:** `training_report.json` stores datasets, split sizes, hyperparameters, seed, library versions, GPU, training time and the actual results. **Use those real numbers in your report — none are pre-filled here.**
- **Documented architecture/parameter count:** printed in the notebook (section 6) and saved in the report.

## 8. Honest limitations (say these in your report too)

- Answers are **extractive spans** from a SQuAD-style reader: good for factual questions ("which optimizer…?"), weak for open-ended ones ("explain the contribution"). For those the app shows `excerpts_only` evidence.
- The training data is Wikipedia + science-exam text, not full research papers; expect some domain gap.
- The evidence threshold is calibrated with *random-passage* negatives. Real off-topic questions about the same paper are harder, so tune `PAPERMIND_THRESHOLD` on a few papers you know.
- Scanned PDFs (images) are rejected — no OCR. Two-column layouts and tables can produce noisy passages.
- Summary/key-info items are chosen by similarity + cue words; "Limitations" and "Future work" only appear if the paper contains matching cue phrases. If summaries come back empty with real models, lower `MIN_SENTENCE_SIM` in `backend/analysis.py`.
- PyMuPDF is AGPL-licensed; fine for coursework, check the terms if you ever distribute the app.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Colab says `No GPU!` | Runtime → Change runtime type → T4 GPU, then Run all again |
| Colab out of memory | Lower `BATCH_SIZE` to 32 or `MAX_SEQ_LEN` to 192 |
| `load_dataset` error | Re-run the cell; if it persists, set `USE_SCIQ = False` |
| Frontend: "Cannot reach the backend" | Start `python app.py` in `backend/`; keep it running |
| `/api/health` says `pretrained-baseline` | The model folder isn't at `backend/models/papermind-minilm/` (it must contain `modules.json`) |
| `faiss-cpu` / `torch` install fails | Use Python 3.11 or 3.12 in a fresh venv |
| Upload takes long | Analysis runs on CPU, roughly 20–60 s for a 10–15 page paper |
| Port 5000 busy (macOS AirPlay) | Change the port in `app.py` and the proxy in `frontend/vite.config.js` |

## 10. Security notes

Uploads are checked (extension, `%PDF-` signature, size and page limits), saved under random server-generated names and never executed; CORS is restricted to the local dev origin; the Flask debugger is off; all SQL is parameterised; React escapes all text; there are **no API keys anywhere** (`.env` is git-ignored anyway). Papers can be deleted from the UI, which removes the PDF, index and database rows. The app is designed for local single-user use — add authentication before exposing it to a network.
