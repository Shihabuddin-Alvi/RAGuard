import json, time, chromadb
from statistics import mean
from core.retrieve import retrieve
from core.generate import generate_answer
from core.models import call_model
from core.faithfulness import faithfulness
from core.metrics import eval_retrieval, citation_accuracy

def run(sample=120, k=5, pause=4.5):
    coll = chromadb.PersistentClient(path="data/chroma").get_collection("rag")
    questions = json.load(open("data/questions.json"))
    ret = eval_retrieval(coll, questions[:300], retrieve, k=k)

    faiths, cites = [], []
    for i, q in enumerate(questions[:sample]):
        hits = retrieve(coll, q["question"], k=k)
        ans = generate_answer(q["question"], hits, call_model)
        faiths.append(faithfulness(ans, hits)["faithfulness"])
        ca = citation_accuracy(ans, hits)["citation_accuracy"]
        if ca is not None:
            cites.append(ca)
        if i % 20 == 0:
            print(f"progress: {i}/{sample}")
        time.sleep(pause)

    report = {
        "retrieval": ret,
        "mean_faithfulness": round(mean(faiths), 3),
        "mean_citation_accuracy": round(mean(cites), 3) if cites else None,
        "sample": sample,
    }
    json.dump(report, open("data/grounding_report.json", "w"), indent=2)
    print(report)

if __name__ == "__main__":
    run()
