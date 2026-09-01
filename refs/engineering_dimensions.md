# Engineering dimensions evaluation glossary

Cross-workshop ADR / decision evaluation framework. When a `### Engineering dimensions evaluation` table appears in any project's `DECISIONS.md`, the following 7 dimensions are the default axes. Underlying standards follow ISO/IEC 25010; the table also gives localized terminology for this project (primarily LLM applications).

## 7 dimensions


|#|ISO/IEC 25010 dimension|Project term|What to focus on|Decision signals|
|---|---|---|---|---|
|1|Maintainability / Modularity (cohesion)|**Cohesion**|Whether related responsibilities cluster within a module|Does one module do one thing; does a single requirement change scatter across many places|
|2|Maintainability / Modularity (coupling)|**Coupling**|Strength of dependencies between modules|How many other modules are affected when replacing, removing, or mocking one module|
|3|Maintainability / Analyzability|**Observability / auditability**|Whether runtime is visible and replayable after the fact|Structured logs / events / transcripts present or not|
|4|Reliability / Fault tolerance (LLM-specific)|**LLM uncertainty tolerance**|How much LLM misbehavior is acceptable|On failure: abort, silent wrong answer, or self-correct + record the violation|
|5|Maintainability / Modifiability|**Backward compatibility / evolution-friendly**|Whether new capabilities break old scenarios|Defaults preserve old behavior; schema extension is additive vs. breaking|
|6|Usability / Learnability|**Learning curve**|How much a user must learn to operate it|Config field count, mental model layers, assumed prior knowledge|
|7|Maintainability / Testability|**Testability**|Whether regression tests / reproducible experiments are feasible|DI injection points, fixture scenarios, deterministic I/O|


## Common in mainstream frameworks, ignored in this project

Dimensions commonly seen in ISO 25010 / generic ADR templates but uniformly ignored in this project (personal vibe sandbox + LLM workshop). If a project moves from `play/` to `grow/` and hits related bottlenecks, evaluate separately within that ADR.


|ISO/IEC 25010 dimension|Term|Reason ignored|
|---|---|---|
|Functional Suitability|Functional correctness|Decisions assume functional requirements are met; not used as a trade-off axis|
|Performance Efficiency|Performance / latency / resource use|Not a bottleneck in the `play/` phase|
|Security|Security|Personal sandbox, no production exposure|
|Reliability (Availability / Recoverability / Maturity)|Availability / recoverability / maturity|Not a production service; re-run cost is low|
|Portability|Portability|Single-machine local runs; no cross-platform / cross-cloud requirement|
|Compatibility / Interoperability|Interoperability|Mostly standalone; cross-tool protocol needs partially covered by dimension 5|
|(LLM supplement, non-ISO)|Call cost / token economics|Not tracked in experiments; if cost drives a decision, note separately in that ADR|

