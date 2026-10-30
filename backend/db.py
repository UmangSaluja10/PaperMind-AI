"""SQLite storage. All queries are parameterised."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    title TEXT,
    n_pages INTEGER,
    n_passages INTEGER,
    analysis_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS passages (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    page INTEGER NOT NULL,
    section TEXT,
    text TEXT NOT NULL,
    PRIMARY KEY (paper_id, idx)
);
CREATE TABLE IF NOT EXISTS qa_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    answer TEXT,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.executescript(SCHEMA)


def insert_paper(paper_id, filename, title, n_pages, passages, analysis):
    with connect() as c:
        c.execute(
            "INSERT INTO papers(id, filename, title, n_pages, n_passages, analysis_json, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (paper_id, filename, title, n_pages, len(passages), json.dumps(analysis), now()),
        )
        c.executemany(
            "INSERT INTO passages(paper_id, idx, page, section, text) VALUES (?,?,?,?,?)",
            [(paper_id, p["idx"], p["page"], p["section"], p["text"]) for p in passages],
        )


def list_papers():
    with connect() as c:
        rows = c.execute(
            "SELECT id, filename, title, n_pages, n_passages, created_at FROM papers"
            " ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_paper(paper_id):
    with connect() as c:
        row = c.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["analysis"] = json.loads(d.pop("analysis_json") or "{}")
    return d


def get_passages(paper_id):
    with connect() as c:
        rows = c.execute(
            "SELECT idx, page, section, text FROM passages WHERE paper_id = ? ORDER BY idx",
            (paper_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_paper(paper_id):
    with connect() as c:
        cur = c.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        return cur.rowcount > 0


def add_history(paper_id, question, result):
    with connect() as c:
        c.execute(
            "INSERT INTO qa_history(paper_id, question, status, answer, result_json, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (paper_id, question, result["status"], result.get("answer"), json.dumps(result), now()),
        )


def get_history(paper_id):
    with connect() as c:
        rows = c.execute(
            "SELECT id, question, result_json, created_at FROM qa_history"
            " WHERE paper_id = ? ORDER BY id",
            (paper_id,),
        ).fetchall()
    return [
        {"id": r["id"], "question": r["question"], "created_at": r["created_at"],
         **json.loads(r["result_json"])}
        for r in rows
    ]
