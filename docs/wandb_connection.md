# 本机 A2 / Forge W&B 连接与监控

适用项目：`/home/xxs/dog/unitree_rl_mjlab_a2`，服务为 `https://api.forge.coreweave.com`，账号/项目保持 `xingshenxu-/a2-ideal-stairs-v02-formal`。本次只修监控，不改变训练配方、环境数、奖励、课程或已耗预算。

## 已定位的问题

1. 同一凭据直连认证查询连续返回 HTTP 403，通过现有 Clash Verge 本地 HTTP 代理 `127.0.0.1:7897` 则返回 200 且认证成功。证明当前路由需要代理，不能据此断言账号失效，也不能确定具体哪个网络设备/服务规则拒绝了直连。
2. 新恢复入口原来每 30 秒交付一次，叠加请求及本地保存耗时，实测约 35 秒才推进一个事件。每轮还产生心跳，训练和评估也持续产生事件。
3. SDK入队后立即读回，暂未可见就阻止后续发送。真实在线复现证明，即使改为 2 秒轮询，只要保留单条等待确认的发送方式，遇到数十秒可见性延迟仍会积压。
4. 原来只有全部队列清空才复位120秒计时，使“持续有ACK进展”也可能被判断为超时。
5. 安装的W&B SDK 0.31.0把Public API请求交给wandb-core；历史日志有无request_id的`api_response`消息。其与本次失败的因果关系未证实。持续确认改走独立HTTPS，避免依赖这一IPC/扫描缓存路径；SDK仍负责持久化上传。

## 修复后的行为

新增 `scripts/tools/wandb_connection.py`，仅由授权恢复入口监控层调用，原132个冻结源保持原字节。

- 启动时在**当前进程**固定已有代理和Forge主机，先做真实认证查询；不修改全局代理、账号、密钥或TLS验证。主机不一致拒绝启动。端口变化可用`A2_WANDB_PROXY`显式指定本地HTTP代理。
- 每2秒重试交付，心跳每30秒产生；已有未确认心跳时不继续堆积心跳。
- 最多32条在途事件，按递增SDK step写入。成功入队可继续流水发送，远端ACK仍必须实际读回。每个intent发送前落盘，已经存在的intent绝不重新log；发送异常或跨会话不确定intent仍阻挡新发送，需原spool/远端核验。
- 有ACK推进就更新“没有交付进展”的计时；无ACK进展达到120秒或最老待确认事件等待120秒，仍保存边界并暂停。没有无限放宽超时、丢弃未确认事件或伪造ACK。
- 确认使用既有凭据向同一个Forge `/graphql` 发起HTTPS只读查询，连接超时5秒、读超时10秒，不跟随重定向；只查询待确认尾部，保留已确认ID缓存。SDK本地成功不等同于远端成功。
- 启动最多等待120秒交付确认，仍验证原初始化、配置和8组监控面板；结束先等待stop真实可见再关闭SDK。确认通道与关闭后的核验使用持久化发送步数，不再调用`run.step` RPC；真实验证曾确认该RPC在SDK关闭后会超时。
- 明确区分“等待远端可见”“HTTP状态错误”“传输异常”；不将所有未确认写成网络断开。

## 可重复连接检查

只查连接和认证，不创建run：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
PYTHONDONTWRITEBYTECODE=1 /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/tools/check_wandb_connection.py \
  --output-dir "logs/wandb_connection_check_$(date +%Y%m%d_%H%M%S)" \
  --no-live-probe --duration 180 --event-interval 5
```

实际读写验证（创建单独标为`connection_validation`的诊断run，不含训练）：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/tools/check_wandb_connection.py \
  --output-dir "logs/wandb_connection_live_$(date +%Y%m%d_%H%M%S)" \
  --live-probe --duration 180 --event-interval 5
```

检查覆盖直连/代理各3次认证、3分钟实际SDK上传/HTTPS读回、45–60秒模拟读回异常、末次全部事件及指标逐字段核对。模拟异常不关闭系统代理，不影响其他应用；不是外部网络长期可靠性的证明。

## 2026-11-13 原训练尾部补交

`logs/wandb_connection_fix_20261113/reconciled_original/result.json` 记录原run 566/566事件真实确认与逐字段验证，无重复事件。原paused run的state/events文件未改，原事件字节未改；新的远端核实状态在 `reconciled_original/telemetry/state.json`。原来“未确认”的558已在远端，未重发；559–566此前无intent，才首次交付。仅监控会话，无PPO/GPU训练。

原暂停仍是历史事实，不能覆盖旧final.json。恢复训练必须从**模型2721／预算2817**完整暂停checkpoint继续，剩183次；不能再次从2700启动或复制旧553事件前缀冒充最新状态。旧operator_launch的哈希审核只适用于旧入口，不能绕过它复用初始训练命令；执行对话应审查新的监控模块和正确的暂停点续接入口。

## 边界

本机代理必须保持运行、服务可达且凭据有效。无法保证外部网络永不中断；本修复让短暂延迟可恢复、长期异常有明确证据并安全暂停。没有屏蔽NaN、调整训练预算或改变任务验收条件。
