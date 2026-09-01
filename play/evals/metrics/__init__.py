"""metrics/ — metric module for reuse across tasks.

According to README guideline #3: direct task adjustment when there is a mature library (family 1 sklearn / family 2 sacrebleu, etc.),
The file is created here only if "no library is available + cross-task reuse".

Module layout (from phase 4 onwards, it is divided into two layers: "methodology family + task/paradigm", see DECISIONS §4 for details):

| Documentation | Content | Key Consumers |
|---|---|---|
| `judge_core.py` | 4 judge LM paradigms: pointwise / pairwise / g_eval / self_consistency + shared parser | qa_open / future summarization / writing |
| `judge_rag.py` | 5 RAG grounding dimensions: faithfulness / answer_correctness / context_precision / context_recall / answer_relevancy + RAG-specific parser | rag_qa |
| `retrieval.py` | 5 IR indicators: recall@k / precision@k / mrr / ndcg@k / map@k (ranx direct adjustment) | rag_retrieval |"""
