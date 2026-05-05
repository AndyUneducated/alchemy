"""Task ABC：lm-evaluation-harness 原版语义.

一个 Task = 一个可复现的评测单元，把 dataset + prompt template + 答案解析 + 聚合方式 绑在一起。

六个抽象方法的职责分界线：
  - docs                 数据源（lazy iterator）
  - doc_to_text          构造 prompt（只在 run 模式被调用）
  - doc_to_target        gold 答案（和 doc_to_text 对称，few-shot 场景留口子）
  - doc_to_choice        MCQ 专用，默认 None
  - process_results      per-sample 评分（统一吃 Response，score/run 共用）
  - aggregation          per-sample → 全局聚合的函数字典，延迟求值
  - higher_is_better     指标方向（show UI / 多 run 排序用）

两个 few-shot 默认方法（Phase 2 加入）：
  - fewshot_docs              example 池，默认 = self.docs()，子类可指 held-out split
  - format_fewshot_example    一条 example 的字符串形式，默认 doc_to_text + doc_to_target

三个 Phase 4 引入的"对齐 lm-eval"hook（全 default 实现，不破老 task）：
  - load_prediction(doc, row)  score 路径自定 JSONL row → (Doc, Response) 翻译
  - process_docs(docs)         run 路径 LM 调用前的 docs 前置加工（RAG retrieve / column rename）
  - output_type = "none"       新增 literal，告诉 Runner 跳过 LM 调用（rag_retrieval 用）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Callable, ClassVar, Literal

from ..api import Doc, Response, SampleResult

OutputType = Literal[
    "generate_until",
    "multiple_choice",
    "loglikelihood",
    "none",  # phase 4：声明该 task 不需要 LM 调用（runner 跳 lm.generate_until）
]


class Task(ABC):
    """所有 task 的基类。子类用 @register_task 装饰自己."""

    name: ClassVar[str]
    output_type: ClassVar[OutputType]

    @abstractmethod
    def docs(self) -> Iterable[Doc]:
        """数据集，允许流式."""
        ...

    @abstractmethod
    def doc_to_text(self, doc: Doc) -> str:
        """构造 prompt（run 模式用）。字面字符串，不要被 provider 的 system prompt 改写."""
        ...

    @abstractmethod
    def doc_to_target(self, doc: Doc) -> str:
        """gold 答案。和 doc_to_text 对称，Runner 自己不碰 target，只有 process_results 碰."""
        ...

    def doc_to_choice(self, doc: Doc) -> tuple[str, ...] | None:
        """MCQ 专用，默认 None."""
        return None

    def fewshot_docs(self) -> Iterable[Doc]:
        """few-shot example 池。默认就是 self.docs()——抽样时由 Runner 排除当前 query.

        子类如果有独立 held-out split（HF dataset 的 train/dev/test 风格），
        override 此方法返回另一份 Iterable[Doc] 即可。
        """
        return self.docs()

    def format_fewshot_example(self, doc: Doc) -> str:
        """单条 example 拼成 prompt 前缀的字符串。默认 = doc_to_text + ' ' + doc_to_target.

        与 lm-eval 的默认 `target_delimiter=' '` 一致；任务可 override 改分隔符
        / 多段结构 / 删指令保留 input→output 短形式。
        """
        return f"{self.doc_to_text(doc)} {self.doc_to_target(doc)}"

    @abstractmethod
    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        """per-sample 评分：
        ① normalize 模型输出（大小写、trim、截断）
        ② 比对 target
        ③ 产 per-sample metrics

        关键约束：需要全集统计的（F1、kappa）**不要**在这里 approximate，
        把原始 pred/target 塞 `metrics` 的私有键（`_pred` / `_target`），交给 aggregation。
        """
        ...

    @abstractmethod
    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        """{metric_name: fn(list[SampleResult]) -> float} 延迟求值.

        为什么返回字典而非数值：
        - 同一批 per-sample 可以喂多个聚合函数
        - 测试时可单独替换某个聚合
        - key 就是最终指标名，Storage 直接用
        """
        ...

    @abstractmethod
    def higher_is_better(self) -> dict[str, bool]:
        """{metric_name: True 表示越大越好}. show UI 和多 run 对比排序用."""
        ...

    # ---- Phase 4 新增 hooks ----------------------------------------------

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score 路径：把 predictions JSONL 一行翻译成 `(enriched_doc, response)`.

        默认实现：仅取 `row['prediction']` 作 Response.text，doc 不动——与 Phase 1
        旧 `_load_predictions` 行为字节相同。

        子类 override 时把 row 里的 pipeline 数据（如 retrieved_ids / contexts）注入
        `doc.metadata`，把 LM-side 数据装 `Response`——遵循 path B+C：Response 只
        装 LM-side，pipeline 产物住 doc 一侧.
        """
        from dataclasses import replace as _replace

        # 默认 doc 不动；子类如需注入 metadata 应在 override 内自行 _replace.
        _ = _replace  # silence vulture
        return doc, Response(doc_id=doc.id, text=row.get("prediction"))

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """run 路径：LM 调用前对 docs 做前置加工（对齐 lm-eval 同名 hook）.

        典型用法：
        - RAG task 在此调 retrieve_fn，把 retrieved_ids/contexts 注入 doc.metadata
        - 任意 task 做 batch tokenize / 字段映射 / column rename / normalize

        默认实现：identity 透传——老 task 不受影响.

        ⚠️ 纯加工纪律（防垃圾桶）：
        - 签名约束：必须 `list[Doc] -> list[Doc]`，**不许带"任务执行"语义**
        - 副作用（日志 / metric 上报 / 状态写入）应放在 metric 闭包或 process_results 内
        - 与 doc 加工无关的初始化（资源准备 / 缓存预热）应放在 task __init__
        """
        return docs

    def collect_judge_responses(self) -> tuple[list[Response], str | None]:
        """run / score 双路径都调，返回 (judge_responses, judge_model_label).

        默认 ([], None)——无 judge 的 task / 未注入 judge_lm 的 task 都返空.
        持有 judge closure 的 task 在此 override，从 closure._recorder 拉响应列表.

        DECISIONS §7.3 evaluation tool call class：runner 双路径都收集 judge 调用记录，
        挂到 `aggregated["efficiency"]["judge"]` 子组（与被测物 task LM 的
        `aggregated["efficiency"].{latency_ms, tokens_in, tokens_out, cost_usd}` 同形 4 子组）.

        实现指引：closure 工厂（judge_pointwise / g_eval / self_consistency / judge_rag.* 5
        factory）都暴露 `closure._recorder.responses + .model_label`；task 把所有 judge closure
        的 responses 合并、取统一 model_label 即可.
        """
        return [], None
