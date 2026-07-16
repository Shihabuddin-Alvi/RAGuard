import nltk
from sentence_transformers import CrossEncoder
from core.utils import softmax

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
NLI = CrossEncoder("cross-encoder/nli-deberta-v3-base")
ENTAIL_INDEX = 1   # cross-encoder/nli-deberta-v3-base order: contradiction, entailment, neutral. Verify on the card.

def faithfulness(answer, hits, threshold=0.5):
    context = " ".join(h["text"] for h in hits)
    sents = [s for s in nltk.sent_tokenize(answer) if s.strip()]
    if not sents:
        return {"faithfulness": 0.0, "sentences": []}

    rows = []
    supported = 0
    for s in sents:
        logits = NLI.predict([(context, s)])
        p = float(softmax(logits[0])[ENTAIL_INDEX])
        ok = p >= threshold
        supported += int(ok)
        rows.append({"sentence": s, "entailment": round(p, 3), "supported": ok})

    return {"faithfulness": round(supported / len(sents), 3), "sentences": rows}
