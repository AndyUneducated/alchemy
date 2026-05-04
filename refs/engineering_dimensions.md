# 工程维度评估词典

跨 workshop 项目通用的 ADR / 决策评估框架。任一项目的 `DECISIONS.md` 里出现 `### 工程维度评估` 表格时，默认以下列 7 个维度为轴。底层标准沿用 ISO/IEC 25010，表格中并列给出本项目（LLM 应用为主）的本地化用词。

## 7 个维度


|#|ISO/IEC 25010 维度|本项目用词|关注什么|判定信号|
|---|---|---|---|---|
|1|Maintainability / Modularity (cohesion 视角)|**内聚度**|模块内相关职责是否聚拢|一个模块是不是只干一件事；改一个需求会不会散落到多处|
|2|Maintainability / Modularity (coupling 视角)|**耦合度**|模块间依赖强度|替换/删除/mock 一个模块会牵连多少其他模块|
|3|Maintainability / Analyzability|**可观测性 / 可审计性**|运行过程是否可见 + 事后是否可回放|有没有结构化 log / event / transcript|
|4|Reliability / Fault tolerance (LLM 场景特化)|**LLM 不确定性容忍**|对 LLM 失控的包容度|遇到失控是 abort、静默错，还是 self-correct + 把违规记账|
|5|Maintainability / Modifiability|**向后兼容 / 演化友好**|新增能力是否破坏旧场景|default 是否保老行为；schema 扩展是加法还是改法|
|6|Usability / Learnability|**学习曲线**|使用者用起来要学多少|配置字段数、心智模型层数、对使用者知识的假设门槛|
|7|Maintainability / Testability|**可测试性**|能否写回归测试 / 可复现实验|有无 DI 注入点、fixture 场景、确定性输入输出|


## 主流框架有、本项目忽略

ISO 25010 / 通用 ADR 模板里常见，但在本项目（个人 vibe sandbox + LLM workshop）语境下统一忽略。如某项目从 `play/` 升 `grow/` 触及相关瓶颈，需在该 ADR 内单独评估。


|ISO/IEC 25010 维度|中文|忽略理由|
|---|---|---|
|Functional Suitability|功能正确性|决策默认假定功能需求已满足，不作为权衡轴|
|Performance Efficiency|性能 / 时延 / 资源占用|`play/` 阶段非瓶颈|
|Security|安全性|个人 sandbox，无生产暴露面|
|Reliability (Availability / Recoverability / Maturity)|可用性 / 可恢复性 / 成熟度|非生产服务，重跑成本低|
|Portability|可移植性|单机本地运行，无跨平台/跨云需求|
|Compatibility / Interoperability|互操作性|多数 standalone；跨工具协议需求由维度 5 部分覆盖|
|(LLM 通用补充，非 ISO)|调用成本 / token 经济性|实验阶段不计；若成本是决策驱动因子需在该 ADR 内单独说明|


