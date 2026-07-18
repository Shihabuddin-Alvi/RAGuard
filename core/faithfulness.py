import re
import nltk
from sentence_transformers import CrossEncoder
from core.utils import softmax

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
NLI = CrossEncoder("cross-encoder/nli-deberta-v3-base")
ENTAIL_INDEX = 1   # cross-encoder/nli-deberta-v3-base order: contradiction, entailment, neutral. Verified against model card and hand-written test cases.

def strip_citations(text):
    return re.sub(r"\s*\[[\d,\s]+\]", "", text).strip()

def faithfulness(answer, hits, threshold=0.5):
    sents = [s for s in nltk.sent_tokenize(answer) if s.strip()]
    if not sents:
        return {"faithfulness": 0.0, "sentences": []}

    rows = []
    supported = 0
    for s in sents:
        clean_s = strip_citations(s)
        best_p = 0.0
        for h in hits:
            logits = NLI.predict([(h["text"], clean_s)])
            p = float(softmax(logits[0])[ENTAIL_INDEX])
            if p > best_p:
                best_p = p
        ok = best_p >= threshold
        supported += int(ok)
        rows.append({"sentence": s, "entailment": round(best_p, 3), "supported": ok})

    return {"faithfulness": round(supported / len(sents), 3), "sentences": rows}
