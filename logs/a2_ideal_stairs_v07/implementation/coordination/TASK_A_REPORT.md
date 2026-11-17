# 执行A：native版本、预算与D22地形交付

日期：2026-11-17。绑定SDS SHA256：`303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca`。A以主协调分派的gpt-6.1-sol/high执行owner身份工作；不派子agent、不启动正式学习、不执行GPU工程、不push。项目根无AGENTS.md；遵守上级用户修改范围/日志要求及coordination两份brief。共享`logs/codex_changes.md`由根追加；每批已发根实际差异与证据。

## owner与修改范围

A在B明确namespace handoff后独占并修改：

- `src/tasks/velocity/v07/spec.py`、`terrain.py`、`env.py`、`env_cfg.py`、`map.py`、`contacts.py`、`events.py`、`speed.py`。
- 新测试`tests/test_v07_spec.py`、`test_v07_terrain.py`、`test_v07_environment.py`、`test_v07_geometry_routing.py`。
- 本报告、A原始CPU/RED日志、`A_OWNER_SHA256.json`。

未改旧v06/C1、旧tests、checkpoint或旧SDS；未改奖励公式、PPO路径或网络。env_cfg/map/contacts/events/speed相对机械namespace副本仅将文档版本/23tile措辞纠正，没有新计算行为。A之外的`curriculum.py`、`evaluation.py`由B拥有，A发现的相对排序/路由问题均由B实际修复。

## 实现逻辑与设计理由

1. spec原生一次声明`0.7.0`、recipe=`0.7-d22-eval-regression-r1`、schema9；架构保持`v06-route-state-v1`。没有先激活C1再覆盖常量，也不读取旧C1环境变量。速度直接声明为C1现行`.3–.6m/s`，三档门禁`.3/.5/.6`。奖励配置仅版本身份不同，删除version后和原v06公式配置完全相等；PPO/资源上限亦逐值相等。
2. D宏预算由C1的9000增加到21000；D25/D22/D20分别6000/6000/9000，总预算由36000变48000。F0/H/L/V/R仍1000/12000/6000/4000/4000，L12/L20/L40仍1500/1500/3000。全量控制样本、物理子步和Adam上限由同一48000派生为1,179,648,000 / 4,718,592,000 / 960,000。每更新512×48控制样本、4物理子步、20 Adam不变；常规checkpoint100、门禁100、诊断50、保持集200不变。
3. 物理表末尾追加D22，tile22原点`(0,176,0)`；0..21的全部TerrainSpec字段与旧表逐项对拍不变。D22为6级、高.20m、踏深.22m、宽.80m、板厚.01m、平台2m；平台从1.32m起、顶高1.20m。显式`PHYSICAL_TILE_COUNT=23`与`FINAL_TILE_ID=21`拆开，避免把新增最大ID当最终任务。
4. `CURRICULUM_TILE_ORDER=(1,…,17,22,18,19,20,21)`不含flat。与B确认按相对难度位置做near/easy：D22 near16/17、D20 near17/22、L12 near22/18；V/R near19/20，easy含22且截止D20。B实现role采样；A只声明唯一排序数据和合同。
5. TerrainSpec继续驱动碰撞几何、解析高度、map、接触索引、奖励高度、timeout与success。terrain原22表断言/query上界改从23表长度取值，踏深合法集合补`.22`。env数组原本动态派生，不另设新数组；合法tile检查仍按`len(specs)`，同时其geometry guard从`>len`改为`>=len`，拒绝等于数组长度的非法索引。物理成功/失败/超时公式没有改。
6. spec集中`REGRESSION_POLICY_CONFIG`（B消费）：保护1000个已计费PPO更新、低成功≤12/60、队列3、改善至少6不退、间隔100、三速各20例；D/L EMA在合同注明只诊断。主决策Qv07-001恢复初始化裁决及SHA已纳入FROZEN_DOCUMENTS；历史Q001～Q004文件路径/原SHA沿用真实v06文档，不伪造v07副本。
7. source closure显式声明B承诺的6个实际script及continuation_v07/runtime/regression_policy/evaluation_records模块，静态递归绑定本地依赖；没有虚构play/smoke/check_migration脚本。当前B实现未交齐时`source_closure_complete=false`且列pending，保持fail closed。最后统一freeze需由根检查实际完整闭包。

## 实际验证与原始记录

所有命令均在项目根执行，解释器`/home/xxs/anaconda3/envs/a2/bin/python`。各A日志保存命令、完整stdout及exit；没有用skip吞依赖错误。局部CPU不会取代根freeze后完整CPU/GPU验收。

| 原始记录 | 实际命令核心 | 结果 |
| --- | --- | --- |
| `A_RED_01.txt` | `python -m pytest -q tests/test_v07_spec.py tests/test_v07_terrain.py tests/test_v07_environment.py` | 实现前10 failed、2 setup errors（缺D22），exit1；无缺依赖 |
| `A_RED_02.txt` | `python -m pytest -q tests/test_v07_spec.py::test_frozen_document_contract_points_to_real_unchanged_sources` | 1 failed，exit1；机械复制Q001_v07路径不存在 |
| `A_GREEN_01.txt` | `python -m pytest -q tests/test_v07_spec.py tests/test_v07_terrain.py` | 10 passed、1 failed，exit1；下文详述浮点测试夹具 |
| `A_GREEN_02.txt` | 同上 | 11 passed、0 skipped、0 errors，exit0，4.01s |
| `A_ENV_GREEN_01.txt` | `python -m pytest -q tests/test_v07_environment.py` | 1 passed、1 failed，exit1；CPU scene误碰Warp全局CUDA默认device |
| `A_ENV_GREEN_02.txt` | `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_v07_environment.py` | 2 passed、0 skipped、0 errors，exit0，6.25s |
| `A_ROUTING_01.txt` | `python -m pytest -q tests/test_v07_geometry_routing.py` | 3 failed、3 passed，exit1；D22真实上界bug，另2个夹具误读不存在case.substage |
| `A_ROUTING_RED_02.txt` | `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_v07_geometry_routing.py` | 修正夹具后1 failed、5 passed，exit1；唯一失败仍为D22被FINAL21上界拒绝 |
| `../cpu/b_geometry_routing_green.log` | B执行同geometry_routing六项 | B修evaluation后6 passed，0.78s，B回报exit0；A已读取原stdout，没有重复跑同SHA |
| `../cpu/b_role_tiles_green2.log` | B执行test_v07_curriculum八项 | 8 passed，0.05s；A读取原stdout，采样实现归B |

当前A拥有四个测试文件共19项的对应局部GREEN：spec/terrain11 + env2 + routing6；不是完整相关CPU套件结论。spec/terrain/env产生一个上游`torch.jit.script`弃用FutureWarning，未压制。工程学习次数0、正式学习次数0。

D22测到了实际MuJoCo模型的ray与几何面、逐阶`.2/.4/.6/.8/1./1.2`顶高、platform、安全内缩、23tile map/索引、首阶接触仅归属tile22、三档timeout1828/1297/1164。真实CPU环境构造23物理tiles，并验证D22 frontier22、23项原点/N/h/d数组、五观测形状、有限值、`.3–.6`命令、vy/wz=0、原出生范围、schedule接收22且拒绝23，以及真实一步reward有限。routing验证当前D25/D22/D20完整60例、保持固定0/1各15例、最终L40/tile21的300例；这里仅验证案例路由，未运行这些300例物理评估。

## 两次非产品失败的透明说明

`A_GREEN_01`的失败对象是`events.PlatformSupportState.update`的身体平台偏移判断。单元夹具分别构造float32字面`x_top=1.32`和`body_x=2.12`，前者再加.8得到`2.1200001239776611`，后者为`2.119999885559082`，所以同为数学2.12的两条独立舍入路径不满足`>=`。这不是当前实际环境的TerrainSpec计算路径。测试改用float64验证理想数学边界（1.32+.8=2.12），保持另一侧2.119失败、debounce/五帧成功不变；没有改生产比较、容差或终止规则。

`A_FLOAT32_BOUNDARY.txt`保存真实events.tile_parameters读D22后的数值：N=6、h=`0.20000000298023224`、d=`0.2199999988079071`，实际`N*d=1.3199999332427979`、实际`N*d+.8=2.119999885559082`，恰等于float32字面2.12，因此实际float32身体线判定通过；相邻更低float32为`2.1199996471405029`。生产继续严格`>=`且没有新增epsilon。几何/标量2.12断言用标准approx表达数学值；没有把float64单元结果声称为物理环境精确实数。

`A_ENV_GREEN_01`真实scene本身用device=cpu，但继承contacts对mujoco_warp.contact_force的调用没有显式launch device；上游warp.launch使用全局默认cuda:0，拿CPU contact_frame去CUDA kernel而报设备不匹配。项目原CPU入口也明确隐藏CUDA（旧test script说明及setdefault）；根同意只在局部CPU命令显式`CUDA_VISIBLE_DEVICES=''`。第二次环境测试通过，contacts未改。第一次自动碰到CUDA kernel不属于GPU工程验证，也没有训练或正式PPO。

## 交付状态与未验证事项

A实现与限定局部CPU完成。A owner当前SHA在`A_OWNER_SHA256.json`。独立reviewer可以只读审核8文件及4tests；根统一codex日志、freeze与最终验收。

未执行：全项目完整CPU套件、统一冻结源码上的任何GPU工程stage、512+8并行实际训练、stop4817真实loaded/ready迁移、正式首4818、45例基线、30例保持协议、最终物理300例、资源峰值、行为成功率比较与push。B尚在完成policy/恢复/runner/脚本，source closure仍有预期pending；A不声称总任务已验收、GPU工程通过或爬楼效果改善。

共享修改记录：`logs/codex_changes.md`（由根逐批追加，本报告不覆盖历史）。
