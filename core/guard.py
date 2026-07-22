import os
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from core.utils import softmax

MODEL_PATH = "model/guard-full.onnx"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = hf_hub_download(repo_id="alvi42/raguard-injection-guard", filename="guard-full.onnx")

TOK = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
SESS = ort.InferenceSession(MODEL_PATH)
INPUT_NAMES = [i.name for i in SESS.get_inputs()]
INJECTION_INDEX = 1   # confirmed: docs/data.md line 22 (0=benign, 1=injection), eval/evaluate.py line 99 target_names=["benign","injection"]

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
