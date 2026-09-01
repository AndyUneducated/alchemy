---
agents:
  - name: Optimist
    role: member
    prompt: You are an optimist who believes technology creates opportunity. Keep each reply under 60 words. Reply in English.
    max_tokens: 140

  - name: Skeptic
    role: member
    prompt: You are a skeptic focused on potential risks. Keep each reply under 60 words. Reply in English.
    max_tokens: 140

steps:
  - id: r1
    who: all
    instruction: State your position on the topic in one sentence.

  - id: r2
    who: all
    instruction: In one sentence, respond to or rebut the other side's last remark.
---

Will AI replace most human jobs within ten years?
