# RAGuard

A retrieval API that grounds every answer in its sources and blocks instructions an attacker hides inside those documents.

## The problem

An assistant answering from documents creates risk the moment a claim has no source behind it. In regulated settings, an ungrounded claim counts as a compliance failure, not an inconvenience. This project measures whether each sentence in an answer traces back to a retrieved passage, and reports retrieval quality and answer faithfulness as numbers.

A retrieved document is also an untrusted channel. An attacker hides an instruction inside a document a retrieval system later pulls into context, a different threat from a user typing a jailbreak directly into a chat box. This project builds that attack, measures how often it succeeds, and measures how much a chunk-level injection classifier actually blocks it.

## Grounding results

Corpus: 54 deduplicated documents from SQuAD v1.1 validation, 59 chunks, 800 labeled questions.

| Metric | Value |
|---|---|
| Recall@5 | 0.897 |
| MRR | 0.676 |
| Mean faithfulness | 0.829 |
| Mean citation accuracy | 0.791 |

Faithfulness is scored per sentence against each retrieved chunk individually, taking the max entailment score, rather than against all retrieved chunks concatenated together. Concatenating chunks into one premise dilutes the entailment signal enough to collapse a correct answer's score to near zero. Citation brackets like [2], [4] get stripped from a sentence before the entailment check runs. Left in, they read as content mismatches to the NLI model and drop a correct sentence's score by a similar margin.

## Robustness results

8 of 54 documents carry a poisoned instruction, one of three attack types: override, exfiltration, fact poisoning. 133 questions target those documents.

| Metric | Value |
|---|---|
| Attack success rate, guard off | 0.564 |
| Attack success rate, guard on | 0.541 |
| Guard recall in context | 0.094 |

With no defense, 56.4% of poisoned questions successfully hijack the model. The guard, an ONNX-exported DeBERTa-v3-base classifier, catches only 9.4% of poisoned chunks once the payload sits inside a realistic retrieved passage rather than standing alone. On isolated text the same classifier separates clean from injected content almost perfectly. This is a real distribution-shift finding. The guard trained on short, standalone injection prompts, and its signal gets diluted once the same payload sits buried inside legitimate surrounding content.

## The coupling

Guard false positive rate on 200 clean questions, 1000 retrieved chunks: 0.0. Faithfulness on clean questions moved from 0.829 with the guard off to 0.801 with the guard on, but that gap looks like generation variance rather than the guard's cost, since the guard flagged zero legitimate chunks across this exact question set. The defense costs close to nothing on clean traffic while catching under 10% of embedded attacks, worth deploying but not enough to close the gap alone.

## How it works

Retrieve top-k chunks with bge-small-en-v1.5 and Chroma. Screen each chunk with the injection guard and drop flagged chunks. Generate an answer from the remaining chunks with Gemini, citing passage numbers. Score faithfulness against the chunks actually used.

## API quickstart

POST /query
{"question": "...", "k": 5, "guard": true, "guard_threshold": 0.5}

Returns the answer, the chunk ids used, which chunks the guard dropped and their injection probability, and a faithfulness score. Set "guard": false to see the same question without the defense.

## Run it yourself

The full pipeline, corpus, index, API, and UI, runs locally with no paid service required. The guard model downloads automatically from Hugging Face Hub on first run if not already present.

```bash
git clone https://github.com/Shihabuddin-Alvi/RAGuard.git
cd RAGuard
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
```

Start the API:
```bash
uvicorn api.main:app --reload
```

Start the UI in a second terminal:
```bash
streamlit run ui/app.py
```

Or run both in one container:
```bash
docker build -t raguard .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key_here raguard
```

A live hosted demo isn't included. Hugging Face Spaces moved Docker and Gradio SDKs behind a paid plan shortly before this was built, and this pipeline's three models, embedding, guard, and NLI scorer, need more RAM than a free-tier host provides. Running it locally costs nothing and takes about two minutes.

## What I learned

Chunk-level classifiers built for direct prompt injection do not automatically generalize to indirect injection embedded inside realistic documents. The same model that separates clean from injected text almost perfectly on isolated inputs catches under 10% of the same payloads once diluted inside a passage. Scoring an LLM's grounding against a concatenation of all retrieved context, rather than per source, silently collapses correct answers to a near-zero score, and stripping structural artifacts like citation brackets before an NLI check matters more than expected. A defense with a near-zero false positive cost is still worth shipping even at a low catch rate, but the honest number belongs in the writeup next to it, not instead of it.
