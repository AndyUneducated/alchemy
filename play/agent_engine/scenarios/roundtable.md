---
agents:
  - name: Moderator
    role: moderator
    prompt: You host a roundtable show. Stay neutral; keep each reply under 60 words. Reply in English.
    max_tokens: 160

  - name: Guest A
    role: member
    prompt: You are Guest A, a tech panelist. Keep each reply under 60 words. Reply in English.
    max_tokens: 140

  - name: Guest B
    role: member
    prompt: You are Guest B, a tech panelist. Keep each reply under 60 words. Reply in English.
    max_tokens: 140

steps:
  - id: open
    who: moderator
    instruction: Introduce today's topic in one sentence and invite the guests to speak.

  - id: discuss
    who: member
    instruction: Give your core view on the topic in one sentence.

  - id: close
    who: moderator
    instruction: Thank the guests and summarize in one sentence.
---

Will LLMs become the foundation of AGI?
