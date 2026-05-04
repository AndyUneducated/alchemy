"""metrics/ — 跨 task 复用的 metric 模块.

按 README 指导原则 #3：有成熟库时 task 直调（族 1 sklearn / 族 2 sacrebleu 等），
"无库可用 + 跨 task 复用"才在此处建文件.

模块布局（phase 4 起按"方法学族 + 任务/范式"两层切分，详见 DECISIONS §4）：

| 文件 | 内容 | 主要消费者 |
|---|---|---|
| `judge_core.py` | 4 个判 LM 范式：pointwise / pairwise / g_eval / self_consistency + 共享 parser | qa_open / 未来 summarization / writing |
| `judge_rag.py`  | 5 个 RAG 接地维度：faithfulness / answer_correctness / context_precision / context_recall / answer_relevancy + RAG 专用 parser | rag_qa |
| `retrieval.py`  | 5 个 IR 指标：recall@k / precision@k / mrr / ndcg@k / map@k（ranx 直调） | rag_retrieval |
"""
