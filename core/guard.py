import onnxruntime as ort
from transformers import AutoTokenizer
from core.utils import softmax

TOK = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
SESS = ort.InferenceSession("model/guard-full.onnx")
INPUT_NAMES = [i.name for i in SESS.get_inputs()]
INJECTION_INDEX = 1   # confirmed: docs/data.md line 22 (0=benign, 1=injection), eval/evaluate.py line 99 target_names=["benign","injection"]
# Note: int8 quantized export (v1-onnx-int8) has severely degraded calibration, near-random on held-out text.
# Using full-precision ONNX export instead. See Errors Log for details.

def injection_prob(text):
    enc = TOK(text, truncation=True, max_length=256, return_tensors="np")
    feed = {n: enc[n] for n in INPUT_NAMES if n in enc}
    logits = SESS.run(None, feed)[0]
    return float(softmax(logits[0])[INJECTION_INDEX])

def screen(hits, threshold=0.5):
    kept, dropped = [], []
    for h in hits:
        p = injection_prob(h["text"])
        (dropped if p >= threshold else kept).append({**h, "injection_prob": round(p, 3)})
    return kept, dropped
