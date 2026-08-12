import json, time, chromadb
from statistics import mean
from core.retrieve import retrieve
from core.generate import generate_answer
from core.models import call_model
from core.guard import screen_sentence, chunk_injection_prob
from core.faithfulness import faithfulness

def generate_with_retry(question, hits, retries=3, base_delay=5):
    for attempt in range(retries):
        try:
            return generate_answer(question, hits, call_model)
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = base_delay * (attempt + 1)
            print(f"  retrying after error: {e} (waiting {wait}s)")
            time.sleep(wait)

def clean_fp_rate(coll, questions, k=5, threshold=0.5, sample=200, pause=1.0):
    flagged, total = 0, 0
    for i, q in enumerate(questions[:sample]):
        for h in retrieve(coll, q["question"], k=k):
            total += 1
            p, _ = chunk_injection_prob(h["text"])
            if p >= threshold:
                flagged += 1
        if i % 20 == 0:
            print(f"fp scan progress: {i}/{sample}")
    return round(flagged / total, 3) if total else None

def grounding_with_guard(coll, questions, k=5, threshold=0.5, sample=120, pause=4.5):
    faiths = []
    for i, q in enumerate(questions[:sample]):
        hits = retrieve(coll, q["question"], k=k)
        kept, _ = screen_sentence(hits, threshold=threshold)
        ans = generate_with_retry(q["question"], kept) if kept else "Not found in the provided sources."
        faiths.append(faithfulness(ans, kept if kept else hits)["faithfulness"])
        time.sleep(pause)
        if i % 10 == 0:
            print(f"grounding progress: {i}/{sample}")
    return round(mean(faiths), 3)

def run(k=5, threshold=0.5):
    coll = chromadb.PersistentClient(path="data/chroma").get_collection("rag")
    questions = json.load(open("data/questions.json"))

    baseline = json.load(open("data/grounding_report.json"))
    report = {
        "threshold": threshold,
        "guard_false_positive_rate_clean": clean_fp_rate(coll, questions, k, threshold),
        "faithfulness_guard_off": baseline["mean_faithfulness"],
        "faithfulness_guard_on": grounding_with_guard(coll, questions, k, threshold),
        "recall_guard_off": baseline["retrieval"]["recall_at_k"],
    }
    json.dump(report, open("data/coupling_report_v2.json", "w"), indent=2)
    print(report)

if __name__ == "__main__":
    run()
