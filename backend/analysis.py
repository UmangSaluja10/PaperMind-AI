"""Structured summary + key-information extraction.

Everything returned is an *extractive* sentence from the paper (with page + section),
chosen by semantic similarity to a short description of what we are looking for.
No text is generated, so nothing can be hallucinated; if nothing in the paper matches
well enough, the field is reported as 'not found'.
"""
import re

import numpy as np

from pdf_processing import split_sentences

SUMMARY_SPEC = [
    dict(key="problem", label="Research problem", n=2, sections={"abstract", "introduction"},
         queries=["What problem or research gap does this paper address?",
                  "The problem is challenging because existing approaches fail"],
         cue=r"\b(problem|challenge|gap|limitation of existing|remains (?:an )?open|however)\b"),
    dict(key="objective", label="Objectives", n=2, sections={"abstract", "introduction"},
         queries=["What is the aim or objective of this study?",
                  "In this paper we propose and aim to"],
         cue=r"\b(we (?:propose|present|introduce|develop|investigate|study|aim)|this (?:paper|work|study)|objective|aim)\b"),
    dict(key="methodology", label="Methodology", n=3, sections={"methods", "abstract"},
         queries=["What method, model, data and experimental setup were used?",
                  "Our approach consists of the following steps"],
         cue=r"\b(we (?:use|used|train|trained|collect|collected|implement|implemented|evaluate)|dataset|architecture|algorithm|model|method)\b"),
    dict(key="results", label="Results", n=3, sections={"results", "abstract", "discussion"},
         queries=["What were the main results and performance numbers?",
                  "Our method outperforms the baseline and achieves accuracy"],
         cue=r"\b(outperform|achiev|improv|accuracy|f1|result|significant|baseline|compared)\w*"),
    dict(key="conclusion", label="Conclusions", n=2, sections={"conclusion", "discussion", "abstract"},
         queries=["What are the main conclusions of this paper?",
                  "In conclusion, we have shown that"],
         cue=r"\b(conclu|we (?:have )?(?:shown|demonstrate)|in summary|overall)\w*"),
]

KEYINFO_SPEC = [
    dict(key="objectives", label="Research objectives", n=3, sections={"abstract", "introduction"},
         queries=["The main contributions of this paper are",
                  "The goal of this research is to"],
         cue=r"\b(contribution|goal|aim|objective|we (?:propose|introduce|present))\w*"),
    dict(key="methodologies", label="Methodologies", n=3, sections={"methods"},
         queries=["The proposed method works as follows",
                  "Experimental setup, datasets and hyperparameters"],
         cue=r"\b(we (?:use|train|propose)|dataset|hyperparameter|architecture|pipeline|algorithm)\w*"),
    dict(key="findings", label="Key findings", n=4, sections={"results", "abstract", "conclusion", "discussion"},
         queries=["Our experiments show that the proposed approach achieves",
                  "The results demonstrate a significant improvement"],
         cue=r"\b(outperform|achiev|improv|show|demonstrat|reduc|increase)\w*"),
    dict(key="limitations", label="Limitations", n=3, sections={"limitations", "discussion", "conclusion"},
         queries=["A limitation of our work is", "The weaknesses and threats to validity of this study"],
         cue=r"\b(limitation|limited|drawback|shortcoming|weakness|caveat|does not|cannot|fail|future (?:work|research)|threat)\w*",
         require_cue=True),
    dict(key="future_directions", label="Future research directions", n=3,
         sections={"future_work", "conclusion", "discussion", "limitations"},
         queries=["In future work we plan to", "Future research directions include"],
         cue=r"\b(future (?:work|research|direction|study)|in the future|we (?:plan|intend|will)|remains? to be|promising direction)\w*",
         require_cue=True),
]

MIN_SENTENCE_SIM = 0.28
SECTION_BOOST = 0.06
CUE_BOOST = 0.04


def _extract_field(spec, passages, vectors, embedder):
    qv = embedder.encode(spec["queries"])
    p_scores = (vectors @ qv.T).max(axis=1)
    for i, p in enumerate(passages):
        if p["section"] in spec["sections"]:
            p_scores[i] += SECTION_BOOST
    top = np.argsort(-p_scores)[:4]

    cand = []
    for pi in top:
        for sent in split_sentences(passages[pi]["text"]):
            n_words = len(sent.split())
            if 6 <= n_words <= 70:
                cand.append((int(pi), sent))
    if not cand:
        return []

    sv = embedder.encode([s for _, s in cand])
    sims = (sv @ qv.T).max(axis=1)
    cue = re.compile(spec["cue"], re.I)
    scored = []
    for (pi, sent), sim in zip(cand, sims):
        has_cue = bool(cue.search(sent))
        if spec.get("require_cue") and not has_cue:
            continue
        if sim < MIN_SENTENCE_SIM:
            continue
        scored.append((float(sim) + (CUE_BOOST if has_cue else 0.0), pi, sent, float(sim)))
    scored.sort(key=lambda x: -x[0])

    chosen, seen = [], set()
    for _, pi, sent, sim in scored:
        if sent in seen:
            continue
        seen.add(sent)
        chosen.append({"text": sent, "page": passages[pi]["page"],
                       "section": passages[pi]["section"], "score": round(sim, 3)})
        if len(chosen) == spec["n"]:
            break
    # present in reading order (page, then original passage order)
    chosen.sort(key=lambda c: c["page"])
    return chosen


def analyze_paper(passages, vectors, embedder):
    def run(spec_list):
        return [
            {"key": s["key"], "label": s["label"],
             "items": _extract_field(s, passages, vectors, embedder)}
            for s in spec_list
        ]

    abstract = " ".join(p["text"] for p in passages if p["section"] == "abstract")[:2500] or None
    return {
        "abstract": abstract,
        "summary": run(SUMMARY_SPEC),
        "key_info": run(KEYINFO_SPEC),
    }
