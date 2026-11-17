# v0.7 外部control协议与调用边界澄清 Qv07-002

主决策线程01a1d49c-5f5d-7890-ac95-9234b307139a针对继承C1的未桥接控制路径明确裁定。原SDS SHA303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca不变；本补充进入最终source/contract/验收绑定。只允许如下最小接线，不改变奖励、课程或预算。

## 协议裁决（主决策原文）

允许最小接线，外部合法action仅continue/pause/stop。

- continue继续。
- stop→user_requested_stop/failure_class=user/reason=user_requested_stop。
- pause本版同样保存后退出，仍user_requested_stop但reason=operator_pause，并标pause_requested=true、resume_authorized=false。
- 本轮不新增可自动恢复暂停态，不以continue自动解除终态；未来恢复需明确裁决。
- reason等文本不能替代action执行，resume_authorized=true也不能自行放行。
- 缺control文件沿原默认continue（尚无指令时）。
- 首次有效pause/stop一旦观察即锁存，不被后续改写撤销。
- 坏JSON/非object/缺或未知action/读取异常抛具名ControlProtocolError，按engineering异常保存退出，不计训练失败/监控故障；保留原control字节及解析错误证据。
- 写控制使用原子替换，禁止异常路径把用户control覆盖为failed。

## 调用边界裁决（主决策原文）

仅绑定stop_flag仍可能不轮询。要求run入口处理pending前检查control，每次新PPO前检查，以及完整边界结束后检查；不在collect/Adam中途强杀。

请求在更新中到达时当前更新正常计费/落盘，下一PPO不得开始；待评估事务保留，不为停机伪造完成。

保留真实独立停止证据，不重写已生效终态；stop/pause命中后不再调用resource_guard或开始最终测试等GPU工作。

真实runtime→runner以临时control文件注入continue/stop/pause/坏JSON/缺action及更新中请求测试，断言checkpoint、error_type、next PPO未执行和原件保留；覆盖首次4818前与pending恢复入口。

记裁决补充+日志、独立复核同SHA。其余奖励/课程/预算不变。
