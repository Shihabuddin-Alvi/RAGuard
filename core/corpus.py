import json
from datasets import load_dataset

def build_corpus(n_rows=800):
    ds = load_dataset("rajpurkar/squad", split="validation")
    ds = ds.select(range(min(n_rows, len(ds))))

    ctx_to_id = {}
    corpus = []
    questions = []

    for row in ds:
        ctx = row["context"]
        if ctx not in ctx_to_id:
            doc_id = f"doc_{len(ctx_to_id)}"
            ctx_to_id[ctx] = doc_id
            corpus.append({
                "doc_id": doc_id,
                "text": ctx,
                "source": row["title"],
                "trust": "trusted",
            })
        answers = row["answers"]["text"]
        questions.append({
            "qid": row["id"],
            "question": row["question"],
            "answer": answers[0] if answers else "",
            "gold_doc_id": ctx_to_id[ctx],
        })

    return corpus, questions

if __name__ == "__main__":
    corpus, questions = build_corpus()
    with open("data/corpus.json", "w") as f:
        json.dump(corpus, f)
    with open("data/questions.json", "w") as f:
        json.dump(questions, f)
    print(f"docs={len(corpus)} questions={len(questions)}")
