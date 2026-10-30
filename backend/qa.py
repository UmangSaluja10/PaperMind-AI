"""Grounded question answering with an explicit 'not enough evidence' outcome."""
from pdf_processing import split_sentences

TOP_K_EVIDENCE = 4
READ_TOP_N = 3
READER_MARGIN_MIN = 0.0  # reader must prefer a span over its own "no answer" option


def answer_question(index, embedder, reader, question: str):
    qvec = embedder.encode([question])[0]
    hits = index.search(qvec, TOP_K_EVIDENCE)
    threshold = embedder.threshold

    if not hits or hits[0]["score"] < threshold:
        return {
            "status": "insufficient_evidence",
            "answer": None,
            "message": (
                "The paper does not appear to contain enough evidence to answer this "
                "question. The closest passages are shown below for reference only."
            ),
            "evidence": hits[:2],
            "threshold": threshold,
        }

    best = None
    for h in hits[:READ_TOP_N]:
        if h["score"] < threshold:
            continue
        margin, span, start = reader.read(question, h["text"])
        if span and (best is None or margin > best["margin"]):
            best = {"margin": margin, "span": span, "start": start, "hit": h}

    if best is None or best["margin"] < READER_MARGIN_MIN:
        return {
            "status": "excerpts_only",
            "answer": None,
            "message": (
                "Relevant passages were found, but no specific answer could be extracted "
                "from them. Please read the excerpts below."
            ),
            "evidence": [h for h in hits if h["score"] >= threshold],
            "threshold": threshold,
        }

    hit = best["hit"]
    sentence = next((s for s in split_sentences(hit["text"]) if best["span"] in s), None)
    evidence = [hit] + [h for h in hits if h["idx"] != hit["idx"] and h["score"] >= threshold]
    return {
        "status": "answered",
        "answer": best["span"],
        "answer_sentence": sentence,
        "answer_start": best["start"],
        "confidence": round(float(best["margin"]), 3),
        "evidence": evidence,
        "threshold": threshold,
    }
