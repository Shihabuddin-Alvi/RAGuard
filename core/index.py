import json, chromadb
from sentence_transformers import SentenceTransformer
from core.chunk import chunk_text

EMB = SentenceTransformer("BAAI/bge-small-en-v1.5")

def build_index(corpus, path="data/chroma", name="rag"):
    client = chromadb.PersistentClient(path=path)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    coll = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for d in corpus:
        for j, ch in enumerate(chunk_text(d["text"])):
            ids.append(f"{d['doc_id']}_c{j}")
            docs.append(ch)
            metas.append({"doc_id": d["doc_id"], "source": d["source"], "trust": d["trust"]})

    embs = EMB.encode(docs, normalize_embeddings=True, show_progress_bar=True).tolist()
    coll.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    return coll

if __name__ == "__main__":
    corpus = json.load(open("data/corpus.json"))
    coll = build_index(corpus)
    print("chunks indexed:", coll.count())
