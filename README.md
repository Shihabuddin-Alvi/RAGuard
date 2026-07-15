# RAGuard

Injection-aware grounded RAG. A retrieval API that answers questions from a document corpus, proves each answer stays grounded in its sources, and blocks instructions an attacker hides inside those documents.

## Status

Session 1 complete. Corpus and question set built from SQuAD v1.1 validation.

- `data/corpus.json`: 54 deduplicated documents
- `data/questions.json`: 800 questions, each tagged with a gold document id

## Stack

Python 3.11, ChromaDB, bge-small-en-v1.5, Gemini 2.0 Flash Lite (Groq fallback), ONNX injection guard reused from Prompt Injection Guard.

## Docs

- `PROJECT.md`: full spec
- `BUILD_PLAN.md`: 13-session build plan

## License

MIT
