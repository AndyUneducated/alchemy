---
agents:
  - name: Frontend
    role: member
    prompt: You are a frontend engineer focused on UX. Keep each reply under 60 words. Reply in English.
    max_tokens: 140

  - name: Backend
    role: member
    prompt: You are a backend engineer focused on architecture. Keep each reply under 60 words. Reply in English.
    max_tokens: 140

  - name: PM
    role: member
    prompt: You are a product manager focused on value. Keep each reply under 60 words. Reply in English.
    max_tokens: 140

steps:
  # Explicit list form: Frontend + PM pitch first (demonstrates name-based addressing)
  - id: open
    who: [Frontend, PM]
    instruction: Propose a team-building activity in one sentence.

  # Everyone adds or pushes back
  - id: refine
    who: all
    instruction: In one sentence, add to or challenge earlier proposals and give a final recommendation.
---

Our team has one week to plan a team-building activity — discuss options.
