"""End-to-end API test using lightweight stub models (no downloads, no GPU).
Run:  python -m pytest tests -q     (or: python tests/test_pipeline.py)
"""
import hashlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PAPERMIND_DATA"] = tempfile.mkdtemp()

import pymupdf  # noqa: E402

import config  # noqa: E402
from app import create_app  # noqa: E402
from pdf_processing import split_sentences  # noqa: E402
import analysis  # noqa: E402

analysis.MIN_SENTENCE_SIM = 0.0  # bag-of-words stub gives lower cosines than a real model


class StubEmbedder:
    threshold = 0.20

    def encode(self, texts, batch_size=32):
        out = np.zeros((len(texts), 8192), dtype="float32")
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z]{4,}", t.lower()):
                out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % 8192] += 1
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out

    def info(self):
        return {"source": "stub", "parameters": 0, "threshold": self.threshold}


class StubReader:
    def read(self, question, context):
        q = set(re.findall(r"[a-z]{4,}", question.lower()))
        best = max(split_sentences(context), key=lambda s: len(q & set(re.findall(r"[a-z]{4,}", s.lower()))))
        overlap = len(q & set(re.findall(r"[a-z]{4,}", best.lower())))
        return overlap - 1.0, best, context.find(best)


SECTIONS = [
    ("Abstract", "We study the problem of detecting plant diseases from leaf images. Existing approaches fail on small datasets, which remains a challenge. In this paper we propose a lightweight convolutional network that achieves 94 percent accuracy and outperforms the baseline."),
    ("1. Introduction", "Plant diseases cause major crop losses worldwide and early detection is a challenge for farmers. The main contribution of this paper is a compact model for field deployment. However, current methods require large labelled datasets and expensive hardware."),
    ("2. Methods", "We train a convolutional network on the PlantVillage dataset using the Adam optimizer with a learning rate of 0.001. We use data augmentation including rotation and flipping. The model architecture contains five convolutional blocks followed by a classifier head."),
    ("3. Results", "Our model achieves 94.2 percent accuracy on the held-out test set, which outperforms the ResNet baseline by 3 points. The improvement is significant across all disease classes. Inference takes 12 milliseconds per image on a phone."),
    ("4. Limitations", "A limitation of our work is that all images were captured under controlled lighting, so the model may not generalise to real field conditions. The dataset also lacks rare diseases."),
    ("5. Conclusion", "In conclusion, we have shown that a small network can detect plant diseases accurately. In future work we plan to collect field images and extend the approach to video."),
    ("References", "[1] Smith J. Deep learning for agriculture. Journal of Farming 2019. [2] Lee K. Crop disease datasets. Nature 2018."),
]


def make_pdf():
    doc = pymupdf.open()
    for i in range(0, len(SECTIONS), 2):
        page = doc.new_page()
        if i == 0:
            page.insert_text((72, 50), "Plant Disease Detection with Small Networks", fontsize=18)
        page.insert_text((72, 30), "Journal of Tests 2026", fontsize=8)
        page.insert_text((300, 800), f"{i // 2 + 1}", fontsize=9)
        y = 80
        for head, body in SECTIONS[i:i + 2]:
            page.insert_textbox(pymupdf.Rect(72, y, 520, y + 24), head, fontsize=13)
            y += 28
            page.insert_textbox(pymupdf.Rect(72, y, 520, y + 150), body, fontsize=10)
            y += 160
    data = doc.tobytes()
    doc.close()
    return data


def test_end_to_end():
    app = create_app(embedder=StubEmbedder(), reader=StubReader())
    c = app.test_client()

    # --- security / validation
    assert c.post("/api/papers").status_code == 400
    r = c.post("/api/papers", data={"file": (io.BytesIO(b"not a pdf at all"), "x.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 400, r.json
    r = c.post("/api/papers", data={"file": (io.BytesIO(make_pdf()), "x.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400
    r = c.post("/api/papers", data={"file": (io.BytesIO(b"%PDF-1.4 garbage"), "bad.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 422, r.json
    assert list(Path(config.UPLOAD_DIR).glob("*")) == [], "failed uploads must be cleaned up"
    assert c.get("/api/papers/..%2f..%2fetc/ask").status_code in (404, 405)
    assert c.get("/api/papers/" + "0" * 32).status_code == 404

    # --- upload
    r = c.post("/api/papers", data={"file": (io.BytesIO(make_pdf()), "../../evil name.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 201, r.json
    paper = r.json
    pid = paper["id"]
    assert ".." not in paper["filename"] and "/" not in paper["filename"]
    assert paper["n_pages"] == 4
    assert paper["title"] == "Plant Disease Detection with Small Networks", paper["title"]
    a = paper["analysis"]
    print("title:", repr(paper["title"]))

    # references excluded, headers/footers removed, sections detected
    with __import__("db").connect() as conn:
        rows = conn.execute("SELECT section, page, text FROM passages WHERE paper_id=?", (pid,)).fetchall()
    texts = " ".join(r["text"] for r in rows)
    assert "Journal of Farming" not in texts, "references must not be indexed"
    assert {r["section"] for r in rows} >= {"abstract", "introduction", "methods", "results", "limitations", "conclusion"}, {r["section"] for r in rows}

    # --- analysis
    summ = {s["key"]: s["items"] for s in a["summary"]}
    info = {s["key"]: s["items"] for s in a["key_info"]}
    assert summ["methodology"] and any(i["section"] == "methods" for i in summ["methodology"])
    assert info["limitations"] and any("controlled lighting" in i["text"] for i in info["limitations"])
    assert info["future_directions"] and any("future work" in i["text"] for i in info["future_directions"])
    assert info["findings"]

    # --- QA: answered with evidence + page
    r = c.post(f"/api/papers/{paper['id']}/ask", json={"question": "Which optimizer and learning rate were used to train the network?"})
    assert r.status_code == 200
    j = r.json
    assert j["status"] == "answered", j
    assert "Adam" in j["answer"]
    assert j["evidence"][0]["page"] >= 1 and j["evidence"][0]["text"]

    # --- QA: unrelated question -> insufficient evidence
    r = c.post(f"/api/papers/{pid}/ask", json={"question": "Who won the football world cup in Brazil stadium?"})
    assert r.json["status"] == "insufficient_evidence", r.json
    assert r.json["answer"] is None

    # --- input validation
    assert c.post(f"/api/papers/{pid}/ask", json={"question": ""}).status_code == 400
    assert c.post(f"/api/papers/{pid}/ask", json={"question": "x" * 600}).status_code == 400

    # --- semantic search
    r = c.post(f"/api/papers/{pid}/search", json={"query": "dataset augmentation rotation flipping", "k": 3})
    assert r.status_code == 200 and r.json["results"][0]["section"] == "methods"

    # --- history + report
    assert len(c.get(f"/api/papers/{pid}/history").json) == 2
    md = c.get(f"/api/papers/{pid}/report?format=md")
    assert md.status_code == 200 and "Question & answer log" in md.data.decode() and "attachment" in md.headers["Content-Disposition"]
    h = c.get(f"/api/papers/{pid}/report?format=html").data.decode()
    assert "<script" not in h and "<h1>" in h
    assert c.get(f"/api/papers/{pid}/report?format=exe").status_code == 400

    # --- restart: index reloaded from disk
    app2 = create_app(embedder=StubEmbedder(), reader=StubReader())
    r = app2.test_client().post(f"/api/papers/{pid}/ask", json={"question": "Which optimizer and learning rate were used to train the network?"})
    assert r.status_code == 200 and r.json["status"] in ("answered", "excerpts_only")

    # --- delete
    assert c.delete(f"/api/papers/{pid}").status_code == 204
    assert c.get(f"/api/papers/{pid}").status_code == 404
    assert list(Path(config.UPLOAD_DIR).glob("*")) == []
    assert list(Path(config.INDEX_DIR).glob("*")) == []
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test_end_to_end()
