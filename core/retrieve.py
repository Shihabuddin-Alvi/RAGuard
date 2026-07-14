from core.index import EMB

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def retrieve(coll, question, k=5):
    q_emb = EMB.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    res = coll.query(query_embeddings=q_emb, n_results=k)
    hits = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"chunk_id": cid, "text": doc, "meta": meta, "distance": dist})
    return hits
