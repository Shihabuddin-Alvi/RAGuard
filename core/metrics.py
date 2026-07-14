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
