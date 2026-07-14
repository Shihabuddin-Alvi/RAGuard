GEN_PROMPT = """Answer the question using only the numbered context passages.
After each claim, cite the passage number in brackets, like [2].
If the passages do not contain the answer, reply exactly: Not found in the provided sources.

Context:
{context}

Question: {question}

Answer:"""

SYSTEM = "You answer strictly from the provided passages and cite passage numbers. You never use outside knowledge."

def format_context(hits):
    return "\n".join(f"[{i+1}] (chunk={h['chunk_id']}) {h['text']}" for i, h in enumerate(hits))

def generate_answer(question, hits, call_model):
    prompt = GEN_PROMPT.format(context=format_context(hits), question=question)
    return call_model(prompt, SYSTEM)
