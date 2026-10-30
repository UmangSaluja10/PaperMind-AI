"""PaperMind AI - Flask API.

Run:  python app.py        (serves http://127.0.0.1:5000)
"""
import re
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import analysis
import config
import db
import qa
import report
from ml import Embedder, Reader
from pdf_processing import PDFError, build_passages, load_document
from retrieval import PaperIndex

PAPER_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def create_app(embedder=None, reader=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    CORS(app, origins=config.CORS_ORIGINS)
    db.init_db()

    embedder = embedder or Embedder()
    reader = reader or Reader()
    index_cache, cache_lock = {}, threading.Lock()

    # ------------------------------------------------------------- helpers
    def err(message, status):
        return jsonify({"error": message}), status

    def valid_paper(paper_id):
        return bool(PAPER_ID_RE.match(paper_id)) and db.get_paper(paper_id) is not None

    def get_index(paper_id):
        with cache_lock:
            if paper_id not in index_cache:
                index_cache[paper_id] = PaperIndex.load(paper_id, db.get_passages(paper_id))
            return index_cache[paper_id]

    @app.after_request
    def security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(HTTPException)
    def http_error(e):
        if e.code == 413:
            return err(f"File too large (limit {config.MAX_UPLOAD_MB} MB).", 413)
        return err(e.description, e.code)

    @app.errorhandler(Exception)
    def unexpected(e):
        app.logger.exception("Unhandled error")
        return err("Internal server error.", 500)

    # ------------------------------------------------------------- routes
    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "model": embedder.info()})

    @app.post("/api/papers")
    def upload():
        f = request.files.get("file")
        if f is None or not f.filename:
            return err("No file uploaded (expected form field 'file').", 400)
        if not f.filename.lower().endswith(".pdf"):
            return err("Only .pdf files are accepted.", 400)
        head = f.stream.read(1024)
        f.stream.seek(0)
        if not head.startswith(b"%PDF-"):
            return err("This file is not a valid PDF.", 400)

        paper_id = uuid.uuid4().hex
        path = Path(config.UPLOAD_DIR) / f"{paper_id}.pdf"  # server-generated name only
        f.save(path)
        index = None
        try:
            doc = load_document(path, config.MAX_PAGES)
            passages = build_passages(doc["pages"])
            if len(passages) < 3:
                raise PDFError(
                    "Almost no text could be extracted. Scanned PDFs need OCR, "
                    "which is not supported."
                )
            vectors = embedder.encode([p["text"] for p in passages])
            index = PaperIndex.build(paper_id, passages, vectors)
            result = analysis.analyze_paper(passages, vectors, embedder)
            filename = secure_filename(f.filename)[:120] or "paper.pdf"
            db.insert_paper(paper_id, filename, doc["title"], doc["n_pages"], passages, result)
        except PDFError as e:
            path.unlink(missing_ok=True)
            if index:
                index.delete_file()
            return err(str(e), 422)
        except Exception:
            path.unlink(missing_ok=True)
            if index:
                index.delete_file()
            raise
        with cache_lock:
            index_cache[paper_id] = index
        return jsonify(db.get_paper(paper_id)), 201

    @app.get("/api/papers")
    def papers():
        return jsonify(db.list_papers())

    @app.get("/api/papers/<paper_id>")
    def paper_detail(paper_id):
        if not valid_paper(paper_id):
            return err("Paper not found.", 404)
        return jsonify(db.get_paper(paper_id))

    @app.delete("/api/papers/<paper_id>")
    def paper_delete(paper_id):
        if not valid_paper(paper_id):
            return err("Paper not found.", 404)
        db.delete_paper(paper_id)
        (Path(config.UPLOAD_DIR) / f"{paper_id}.pdf").unlink(missing_ok=True)
        (Path(config.INDEX_DIR) / f"{paper_id}.faiss").unlink(missing_ok=True)
        with cache_lock:
            index_cache.pop(paper_id, None)
        return "", 204

    def read_text_field(name):
        data = request.get_json(silent=True) or {}
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            return None, err(f"'{name}' is required.", 400)
        value = value.strip()
        if len(value) > config.MAX_QUESTION_CHARS:
            return None, err(f"'{name}' is too long (max {config.MAX_QUESTION_CHARS} chars).", 400)
        return value, None

    @app.post("/api/papers/<paper_id>/ask")
    def ask(paper_id):
        if not valid_paper(paper_id):
            return err("Paper not found.", 404)
        question, bad = read_text_field("question")
        if bad:
            return bad
        result = qa.answer_question(get_index(paper_id), embedder, reader, question)
        db.add_history(paper_id, question, result)
        return jsonify({"question": question, **result})

    @app.post("/api/papers/<paper_id>/search")
    def search(paper_id):
        if not valid_paper(paper_id):
            return err("Paper not found.", 404)
        query, bad = read_text_field("query")
        if bad:
            return bad
        k = request.get_json(silent=True).get("k", 5)
        k = k if isinstance(k, int) and 1 <= k <= 20 else 5
        hits = get_index(paper_id).search(embedder.encode([query])[0], k)
        return jsonify({"query": query, "results": hits})

    @app.get("/api/papers/<paper_id>/history")
    def history(paper_id):
        if not valid_paper(paper_id):
            return err("Paper not found.", 404)
        return jsonify(db.get_history(paper_id))

    @app.get("/api/papers/<paper_id>/report")
    def export(paper_id):
        if not valid_paper(paper_id):
            return err("Paper not found.", 404)
        fmt = request.args.get("format", "md")
        paper, hist = db.get_paper(paper_id), db.get_history(paper_id)
        stem = f"papermind-report-{paper_id[:8]}"
        if fmt == "html":
            body, mime, ext = report.build_html(paper, hist), "text/html", "html"
        elif fmt == "md":
            body, mime, ext = report.build_markdown(paper, hist), "text/markdown", "md"
        else:
            return err("format must be 'md' or 'html'.", 400)
        return Response(body, mimetype=f"{mime}; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.{ext}"'})

    return app


if __name__ == "__main__":
    # debug=False on purpose: the Werkzeug debugger allows remote code execution.
    create_app().run(host="127.0.0.1", port=5000, debug=False, threaded=True)
