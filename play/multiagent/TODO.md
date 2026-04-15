# TODO — multiagent engine

## P0

### instruction 泄露

`discussion.py` `_exec_phase` 将 instruction 追加到共享 history，导致所有后续 agent 都能看到本不属于自己的指令。例如给 moderator 的"点名追问"指令会被 members 读到，污染其行为。

应改为：instruction 只对当前 phase 的目标 agent 可见，不进入全局 history。

## P1

### 无轮次感知

`discussion.py` 的 `Round N` 仅打印到 stdout，未注入 history。agent 不知道自己在第几轮，无法自适应（如"最后一轮该收敛了"）。

应在每轮 main 开始时向 history 注入轮次标记。

### main instruction 不支持按轮次变化

phases 是静态定义的，main 阶段 instruction 每轮完全相同。场景作者无法编排"第 1 轮自由讨论 → 第 2 轮聚焦分歧 → 第 3 轮逼迫表态"的递进节奏。

可考虑支持 `instructions: [...]` 列表按轮次索引，或支持模板变量如 `{round}/{rounds}`。

## P2

### 固定发言顺序

members 每轮发言顺序与列表定义顺序相同。先发言者信息最少，后发言者拥有上下文优势，造成结构性不公平。

可考虑每轮随机打乱或轮转发言顺序。
