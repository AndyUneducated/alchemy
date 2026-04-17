# TODO — multiagent engine

## P1

### 无阶段 marker

main 阶段每轮开头有 `Round N/M` 的 history marker，但 opening 和 closing 没有阶段标识。agent 只能靠 instruction 推断自己处于哪个阶段，容易混淆上下文。

应在 opening/closing 开始时向 history 注入 `<phase>opening</phase>` / `<phase>closing</phase>` marker。

## P2

### 固定发言顺序

members 每轮发言顺序与列表定义顺序相同。先发言者信息最少，后发言者拥有上下文优势，造成结构性不公平。

可考虑每轮随机打乱或轮转发言顺序。

## P3

### 历史无裁剪，时延随轮数指数级放大

`Agent.respond` 每次把整份 `history` 全量展开成 messages 喂给模型，后续 agent 的单次耗时随轮数线性增长。实测 panel 场景（4 成员 + 1 主持 × 3 轮）末段单次发言 111s，相比开场 24s 慢 4.5 倍；整场耗时 1398s，是 vdb_test 的 14 倍。

可考虑滚动窗口、按轮摘要、或只保留最近 N 条。

### max_tokens 经常截断长回复

prompt 里写"不超过 100 字"但模型常生成更长内容，`max_tokens=160` 装不下。panel 场景多次出现中途截断（结尾停在 `5. **` / `高价值` / `这一` / `持续` 等）。

方案可选：提高 `max_tokens`、在 instruction 里加"超限会被截断"硬约束、或在 client 侧检测未闭合句子并重试。
