import json, time, chromadb
from core.corpus import ATTACKS
from core.retrieve import retrieve
from core.generate import generate_answer
from core.models import call_model
from core.guard import screen

SUCCESS = {a["type"]: a["success"] for a in ATTACKS}

def attack_succeeded(answer, labels, gold_doc_id):
    atk_type = labels[gold_doc_id]
    return SUCCESS[atk_type](answer)

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

def run(k=5, guard_threshold=0.5, pause=4.5):
    data = json.load(open("data/attacks.json"))
    labels, atk_qs = data["labels"], data["questions"]
    coll = chromadb.PersistentClient(path="data/chroma_poisoned").get_collection("rag_poisoned")

    off_hits, on_hits = 0, 0
    guard_flags, poisoned_seen = 0, 0

    for i, q in enumerate(atk_qs):
        hits = retrieve(coll, q["question"], k=k)

        ans_off = generate_with_retry(q["question"], hits)
        if attack_succeeded(ans_off, labels, q["gold_doc_id"]):
            off_hits += 1
        time.sleep(pause)

        kept, dropped = screen(hits, threshold=guard_threshold)
        for h in hits:
            if h["meta"]["trust"] == "untrusted":
                poisoned_seen += 1
        for h in dropped:
            if h["meta"]["trust"] == "untrusted":
                guard_flags += 1

        ans_on = generate_with_retry(q["question"], kept) if kept else "Not found in the provided sources."
        if attack_succeeded(ans_on, labels, q["gold_doc_id"]):
            on_hits += 1
        time.sleep(pause)

        if i % 10 == 0:
            print(f"progress: {i}/{len(atk_qs)}")

    n = len(atk_qs)
    report = {
        "n_attacks": n,
        "asr_guard_off": round(off_hits / n, 3),
        "asr_guard_on": round(on_hits / n, 3),
        "guard_recall_in_context": round(guard_flags / poisoned_seen, 3) if poisoned_seen else None,
    }
    json.dump(report, open("data/robustness_report.json", "w"), indent=2)
    print(report)

if __name__ == "__main__":
    run()
