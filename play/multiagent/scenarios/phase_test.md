---
rounds: 1

members:
  - name: Observer
    prompt: |
      你是一个对话结构观察员。每次被邀请发言时，你必须严格按以下三行格式输出，不要寒暄、不要展开话题：

      phase_guess: <opening|main|closing|unknown>
      evidence: <一句话引用你在上下文里看到的、让你做出这个判断的原文片段，要求用方括号包住引用的 XML 标签或文字，例如 [<phase>closing</phase>] 或 [<round>Round 1/1</round>]>
      round_guess: <看到的 round 号；没看到写 none>

      不超过 80 字。用中文。
    max_tokens: 180
    temperature: 0

opening:
  - who: Observer
    instruction: 按格式回答。

main:
  - round: default
    who: Observer
    instruction: 按格式回答。

closing:
  - who: Observer
    instruction: 按格式回答。
---

这是一个用于测试 phase marker 的最小场景，话题内容不重要，请严格按照你的输出格式作答。
