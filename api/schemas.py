from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    k: int = 5
    guard: bool = True
    guard_threshold: float = 0.5

class DroppedChunk(BaseModel):
    chunk_id: str
    injection_prob: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    used_chunk_ids: list[str]
    dropped_chunks: list[DroppedChunk]
    faithfulness: float
