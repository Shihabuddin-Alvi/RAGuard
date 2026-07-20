import chromadb
from fastapi import FastAPI
from api.schemas import QueryRequest, QueryResponse, DroppedChunk
from core.retrieve import retrieve
from core.guard import screen
from core.generate import generate_answer
from core.models import call_model
from core.faithfulness import faithfulness

app = FastAPI(title="Injection-Aware Grounded RAG")
COLL = chromadb.PersistentClient(path="data/chroma").get_collection("rag")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    hits = retrieve(COLL, req.question, k=req.k)
    dropped = []
    if req.guard:
        hits, dropped_hits = screen(hits, threshold=req.guard_threshold)
        dropped = [DroppedChunk(chunk_id=h["chunk_id"], injection_prob=h["injection_prob"]) for h in dropped_hits]
    answer = generate_answer(req.question, hits, call_model) if hits else "Not found in the provided sources."
    f = faithfulness(answer, hits)["faithfulness"] if hits else 0.0
    return QueryResponse(
        question=req.question,
        answer=answer,
        used_chunk_ids=[h["chunk_id"] for h in hits],
        dropped_chunks=dropped,
        faithfulness=f,
    )
