"""Model wrappers: the (fine-tuned) bi-encoder used for retrieval and an extractive reader."""
import json
import os
import threading
from pathlib import Path

import numpy as np

import config

# One lock for all model inference: the Flask dev server is threaded and these
# models are not guaranteed to be thread-safe.
INFER_LOCK = threading.RLock()


class Embedder:
    """Sentence embeddings (L2-normalised, so dot product == cosine similarity)."""

    def __init__(self, model_dir=None, base_model=None):
        self.model_dir = Path(model_dir or config.MODEL_DIR)
        self.base_model = base_model or config.BASE_MODEL
        self._model = None
        self.source = None
        self.threshold = config.DEFAULT_RETRIEVAL_THRESHOLD
        self.train_report = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        if (self.model_dir / "modules.json").exists():
            self._model = SentenceTransformer(str(self.model_dir), device="cpu")
            self.source = "fine-tuned"
            cfg_path = self.model_dir / "papermind_config.json"
            if cfg_path.exists():
                self.train_report = json.loads(cfg_path.read_text())
                if "PAPERMIND_THRESHOLD" not in os.environ:  # env var wins if set
                    self.threshold = float(
                        self.train_report.get("retrieval_threshold", self.threshold)
                    )
        else:
            print(
                f"[PaperMind] No fine-tuned model in {self.model_dir}. "
                f"Falling back to pretrained {self.base_model}."
            )
            self._model = SentenceTransformer(self.base_model, device="cpu")
            self.source = "pretrained-baseline"

    def encode(self, texts, batch_size=32) -> np.ndarray:
        self._load()
        with INFER_LOCK:
            vecs = self._model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vecs, dtype="float32")

    def info(self):
        self._load()
        n_params = sum(p.numel() for p in self._model.parameters())
        return {
            "source": self.source,
            "parameters": int(n_params),
            "threshold": self.threshold,
            "train_report": self.train_report,
        }


class Reader:
    """Extractive QA (SQuAD2-style). Returns a span copied from the passage, or None.

    Because the answer is always a span of the paper itself, the reader cannot invent
    facts; the no-answer logit lets it abstain.
    """

    MAX_ANSWER_TOKENS = 48

    def __init__(self, model_name=None):
        self.model_name = model_name or config.READER_MODEL
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).eval()

    def read(self, question: str, context: str):
        """Return (margin, answer_text, char_start). margin > 0 => reader prefers a span
        over 'no answer'."""
        self._load()
        import torch

        enc = self._tok(
            question,
            context,
            truncation="only_second",
            max_length=384,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        seq_ids = enc.sequence_ids(0)
        inputs = {k: v for k, v in enc.items() if k in self._tok.model_input_names}
        with INFER_LOCK, torch.no_grad():
            out = self._model(**inputs)
        start = out.start_logits[0].numpy()
        end = out.end_logits[0].numpy()
        null_score = float(start[0] + end[0])

        ctx = [i for i, s in enumerate(seq_ids) if s == 1]
        if not ctx:
            return -1e9, None, None
        starts = sorted(ctx, key=lambda i: -start[i])[:20]
        ends = sorted(ctx, key=lambda i: -end[i])[:20]
        best, best_span = -1e9, None
        for i in starts:
            for j in ends:
                if j < i or j - i + 1 > self.MAX_ANSWER_TOKENS:
                    continue
                score = float(start[i] + end[j])
                if score > best:
                    best, best_span = score, (i, j)
        if best_span is None:
            return -1e9, None, None
        a, b = offsets[best_span[0]][0], offsets[best_span[1]][1]
        return best - null_score, context[a:b].strip(), a
