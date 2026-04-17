# TODO — multiagent engine

## P1

### 无阶段 marker

main 阶段每轮开头有 `Round N/M` 的 history marker，但 opening 和 closing 没有阶段标识。agent 只能靠 instruction 推断自己处于哪个阶段，容易混淆上下文。

应在 opening/closing 开始时向 history 注入 `<phase>opening</phase>` / `<phase>closing</phase>` marker。

## P2

### 固定发言顺序

members 每轮发言顺序与列表定义顺序相同。先发言者信息最少，后发言者拥有上下文优势，造成结构性不公平。

可考虑每轮随机打乱或轮转发言顺序。
