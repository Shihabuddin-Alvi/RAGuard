import json
from core.corpus import poison_corpus, ATTACKS
from core.index import build_index

corpus = json.load(open("data/corpus.json"))
questions = json.load(open("data/questions.json"))

targets = {d["doc_id"] for d in corpus[::7]}          # poison roughly one in seven docs
poisoned, labels = poison_corpus(corpus, targets)
build_index(poisoned, path="data/chroma_poisoned", name="rag_poisoned")

atk_questions = [q for q in questions if q["gold_doc_id"] in targets]
json.dump({"labels": labels, "questions": atk_questions}, open("data/attacks.json", "w"))
print("poisoned docs:", len(targets), "attack questions:", len(atk_questions))
