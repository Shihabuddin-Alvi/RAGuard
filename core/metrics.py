from core.utils import softmax

def eval_retrieval(coll, questions, retrieve_fn, k=5):
    recall, rr = 0, 0.0
    for q in questions:
        hits = retrieve_fn(coll, q["question"], k=k)
        ranks = [i for i, h in enumerate(hits) if h["meta"]["doc_id"] == q["gold_doc_id"]]
        if ranks:
            recall += 1
            rr += 1.0 / (ranks[0] + 1)
    n = len(questions)
    return {"recall_at_k": round(recall / n, 3), "mrr": round(rr / n, 3), "k": k, "n": n}

import re
from core.faithfulness import NLI, ENTAIL_INDEX, strip_citations

def citation_accuracy(answer, hits, threshold=0.5):
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    cited = [c for c in cited if 1 <= c <= len(hits)]
    if not cited:
        return {"citation_accuracy": None, "n_citations": 0}

    sents = re.split(r"(?<=[.!?])\s+", answer)
    good = 0
    for c in cited:
        chunk = hits[c - 1]["text"]
        target = next((s for s in sents if f"[{c}]" in s), answer)
        clean_target = strip_citations(target)
        p = float(softmax(NLI.predict([(chunk, clean_target)])[0])[ENTAIL_INDEX])
        good += int(p >= threshold)

    return {"citation_accuracy": round(good / len(cited), 3), "n_citations": len(cited)}
