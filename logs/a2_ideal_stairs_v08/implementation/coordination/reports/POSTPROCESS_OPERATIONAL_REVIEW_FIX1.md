# v08 终态后处理操作审查 — FIX1 定向复审

日期：2026-11-18。独立 reviewer `/root/v08_final_reviewer`。

**结论：O1/O2/O3全部关闭，Critical=0、Important=0、Minor=0，本操作helper blocking=0，可排队等待真实合同终态。** 此结论不表示后处理已经运行/完成，不修改W5 ENGINEERING_ACCEPTANCE，不重开冻结训练源码，也不改变正式训练或SDS预算/门槛。

## 绑定与检查范围

- helper `implementation/coordination/post_formal_pipeline_20261118.py` SHA `6b28826aa14b54a494b6c0b26c9acfce2be1fead1d19daa2754e3a837f190c41`。
- 新CPU回归 `implementation/coordination/test_post_formal_pipeline.py` SHA `d70238be759e65dafc5e93ce1873bf3f4f9536b22019e151284456ab1c9f8ec4`。
- 按原报告仅核验三项修复及直接引入的测试/行为；读取 `POSTPROCESS_FIX_WAVE_V2.json`，独立重算其中helper、测试和RED/GREEN/self-test三份实际日志SHA，全部一致。
- 仅CPU临时目录/合成黑色视频验证；没有GPU、机器人物理、学习、正式控制、源码/共享日志/Git改动，没有委派。唯一持久写入为本报告。

## O1 — ADDRESSED

helper `:282–293` 现在先获得driver锁，再创建唯一新PIPELINE目录；只有mkdir成功进入owned body后的异常才写event。`:402–408` 的main不再依据“目录存在”向旧队列写错误状态。已有目录和锁竞争都在ownership之前拒绝，因此重复启动不能覆盖另一实例的running/complete证据。

`OwnershipTests` 实际检查waiting_formal与postprocess_complete两种已有目录，其status/events逐字节不变；另持有driver锁时调用main，确认拒绝且未创建新attempt。原O1反例已被明确逆转，保留原“不自动重评”的约束。

## O2 — ADDRESSED

helper `:120–133` 将末两次连续E历史门禁绑定到 `state.evaluation['gate:<update>']`：kind=gate、phase=E、准确update、冻结gate:E病例SHA、SOURCE_SHA和FREEZE_SHA；summary要求相同kind/phase/caseSHA、完整420 valid、invalid/error0、engineering_error=false、gate_pass=true。该reference由native_inputs之前的现有全raw/sidecar/journal验证重算过，helper没有重新评估或消费训练RNG。

因此cur布尔history不再能覆盖实际false/missing/foreign/partial门禁。该限定实现满足原报告要求的最小末两E绑定，无需修改冻结CurriculumManager、重放物理门禁或新增学习预算。`GateProofTests` 的合法E正例通过，11种缺件/false/身份与来源/冻结错配/不完整分母/工程错误反例全部被拒绝。

## O3 — ADDRESSED

helper `:227–278` 现在校验固定8病例列表、对应case_id.mp4文件名、原native右侧录制协议及每例控制样本。实际媒体元数据要求1280×720、25fps；时长与 `ceil(control_steps/2)+1` 帧及25fps对应。这个计数包含原初始帧、每两控制步一帧，以及奇数终态的强制末帧，没有要求新录制配方。

完整ffmpeg解码启用 `-xerror`，非零退出或非空error log均拒绝；progress须正常end、实际解码帧数与预期精确相同，并核对decoded timeline。元数据时长容差0.011秒仅覆盖reader的0.01秒显示精度，小于一帧0.04秒，不可吞掉缺帧。输出保留实际帧数、fps/尺寸/时长、错误日志和progress引用。

`MediaProofTests` 使用真实CPU编码的合成黑色视频，分别证明偶/奇终态5/6控制步均对应4帧并完整通过；短但干净可解码视频、fps错误、尺寸错误、外来病例文件名、非零解码退出，以及返回0但写error log的mock都被拒绝。这些是媒体完整性证据，**不是机器人视频或策略成功证据**。

## 实际验证

本reviewer独立执行：

```bash
CUDA_VISIBLE_DEVICES= A2_V08_WANDB=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/xxs/anaconda3/envs/a2/bin/python -B \
logs/a2_ideal_stairs_v08/implementation/coordination/test_post_formal_pipeline.py
```

真实输出：**9组通过，1.021秒，exit0**。全部临时媒体位于/tmp并由TemporaryDirectory退出清理；未执行真实队列。原5项helper self-test已读取实际通过输出与日志SHA，本轮没有重复执行。

修复报告的RED日志SHA `efd2c61b3c312ce5ab69fd2e7c2261d01715e1f03999b34ea6d385884650ad0e`、GREEN日志SHA `cc3895f7846408345ce16d53c8ac7af3165c258f8aaaaf0fe05523f2691c735d`、self-test日志SHA `af8c18656a71385bd617f3d8656e666db03b565ae236900037745e5437595187` 均实际重算一致，不用报告自述替代文件核对。

检查时 `implementation/postprocess_v1` 仍不存在，队列尚未运行。本轮未发现修复引入的新阻断。取消/单worker/已存在输出不自动重评等未改路径沿用原范围结论，没有因此重审模型或训练源码。

## 剩余范围

允许root把此helper排队，但它必须继续等待唯一formal真实finished/exit0及已验证SDS合同终态；operator/工程异常不能触发。真实final2100与机器人8视频、最终媒体完整性和策略结果尚未发生，仍需实际回执和之后独立结果复核。工程接受、正式run及freeze不因此更改；共享修改日志由root统一维护。
