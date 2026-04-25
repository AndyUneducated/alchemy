---
memory:
  type: window
  max_recent: 2

agents:
  - name: Windowed
    role: member
    prompt: |
      你是对话可见度观察员。每次发言严格按以下三行格式输出，不要展开话题、不要寒暄：

      visible_speakers: <用逗号分隔，列出你在历史里看到的所有发言者的名字；只看到 topic/turn 没看到任何人就写 none>
      memory_type: <你当前使用的 memory；window / full / summary 三选一，根据你的 system prompt 判断>
      summary_seen: <yes|no，是否看到 <summary> 标签>

      不超过 60 字。用中文。你当前使用的 memory 是 window。
    max_tokens: 140
    temperature: 0

  - name: FullRecall
    role: member
    prompt: |
      你是对话可见度观察员。每次发言严格按以下三行格式输出，不要展开话题、不要寒暄：

      visible_speakers: <用逗号分隔，列出你在历史里看到的所有发言者的名字；只看到 topic/turn 没看到任何人就写 none>
      memory_type: <window / full / summary，根据你的 system prompt 判断>
      summary_seen: <yes|no，是否看到 <summary> 标签>

      不超过 60 字。用中文。你当前使用的 memory 是 full。
    memory:
      type: full
    max_tokens: 140
    temperature: 0

  - name: Summarizer
    role: member
    prompt: |
      你是对话可见度观察员。每次发言严格按以下三行格式输出，不要展开话题、不要寒暄：

      visible_speakers: <用逗号分隔，列出你在历史里看到的所有发言者的名字；只看到 topic/turn 没看到任何人就写 none>
      memory_type: <window / full / summary，根据你的 system prompt 判断>
      summary_seen: <yes|no，是否看到 <summary> 标签>

      不超过 60 字。用中文。你当前使用的 memory 是 summary。
    memory:
      type: summary
      max_recent: 2
      summarizer_prompt: 把对话压成一句话，保留每个发言者最核心的立场，不超过 80 字。
    max_tokens: 160
    temperature: 0

steps:
  - id: r1
    who: all
    instruction: 严格按照三行格式回答；memory_type 根据你的 system prompt 判断。

  - id: r2
    who: all
    instruction: 严格按照三行格式回答；memory_type 根据你的 system prompt 判断。

  - id: r3
    who: all
    instruction: 严格按照三行格式回答；memory_type 根据你的 system prompt 判断。
---

这是一个专门用来验证 memory 分层的最小场景，话题内容不重要，请严格按照输出格式作答。
