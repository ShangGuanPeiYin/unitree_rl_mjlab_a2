# A1资源失败独立只读诊断（2026-11-27）

本次仅使用Python标准库读取已有JSON/JSONL、源码及当前/proc/meminfo，并用不含args的ps读取pid/ppid/rss/comm；未导入Torch/NumPy、未新增物理/修改源码/杀进程。仅新增本报告，变更日志及Git由根代理统一处理。

## 结论与执行边界

新A1 1024 DAgger针对性资源测试实际失败，selection.passed=false、num_envs=null，错误为`resource ceiling exceeded during full update`。硬门槛是全程MemAvailable≥4294967296 bytes（4 GiB），实际最低3353313280 bytes（约3.123 GiB），不足941654016 bytes。GPU总2568 MiB、worker1120 MiB均在限额内，不能抵消host失败。

失败不是只发生在after-update阶段：最低值位于首轮update内部，且初始采样/填满replay区间和首轮sampling区间已经分别出现host越限。本次只完成1 fill＋1 update的物理采样共98304，updated49152；计划的后续warm/measurement、保存恢复均未到达。PPO资源栈、112例诊断、D1原预算50轮试训及正式AB均未启动；学习样本0。资源门禁失败后必须停止，不得以旧资源通过或新CPU通过替代。

全局probe charged=executed=3391488，updated3170304，uncertain0；硬上限3833856，余442368。已发生费用不退款，剩余数字不构成自行重测授权。

## 精确时刻与阶段

新resources.jsonl共219条周期采样。全程最低位于第212条（零基211）：epoch1795748984.8566513，即**2026-11-27 11:09:44.856651+08:00**，MemAvailable3353313280、worker RSS7112318976、总GPU2568 MiB/worker1120 MiB。

前一个阶段标记sampling_0_complete在1795748983.3634994，后一个update_0_complete在1795748985.7421005。故该最低点处于learner.update调用内（该调用同时包含逻辑批采样、前后向/Adam及末尾reservoir.add），没有更细内部时间戳，不能精确指定为其中某个算子。

| 阶段区间 | 周期样本数 | 最小MemAvailable/bytes | 低于4GiB条数 | 最大周期worker RSS/bytes |
|---|---:|---:|---:|---:|
| sampling_initial_begin → replay_full_initial_released | 117 | 3697303552 | 19 | 6830051328 |
| sampling_0_begin → sampling_0_complete | 49 | 4110860288 | 12 | 6570680320 |
| sampling_0_complete → update_0_complete | 18 | 3353313280 | 18 | 7217213440 |
| update_0_complete → rollout_0_released | 1 | 4565553152 | 0 | 6051835904 |

49条周期样本低于4 GiB。首次低于门槛在epoch1795748974.2134385，MemAvailable3998273536，位于sampling_initial_begin到replay_full_initial_released的区间。该区间未设置initial采样完成/每次fill分界，无法用日志进一步区分首次越限是在collect末尾还是reservoir填充内部。首轮sampling阶段第一次低于门槛在1795748981.822304，4294451200 bytes；sampling_0_complete阶段标记4106727424也低于4 GiB。

## 与旧完整测试的同阶段比较

| 指标/bytes | 新A1 | 旧原完整1024 DAgger |
|---|---:|---:|
| sampling_initial_begin MemAvailable | 8231247872 | 10129899520 |
| sampling_initial_begin worker RSS | 2293178368 | 2280017920 |
| 初始采样→填满后释放区间最低MemAvailable | 3697303552 | 5807124480 |
| sampling_0区间最低MemAvailable | 4110860288 | 5974470656 |
| update_0区间最低MemAvailable | 3353313280 | 5538213888 |
| update_0_complete记录的进程RSS高水位 | 7224512512 | 7352033280 |
| rollout_0_released MemAvailable | 4597637120 | 6427103232 |

新初始host可用量已比旧低1898651648 bytes（约1.77 GiB），而同阶段worker RSS只高13160448 bytes。新首update进程RSS高水位反而比旧低127520768 bytes（约121.6 MiB）。这些事实支持“机器可用余量起点较低是重要条件”，不支持直接声称“A1逻辑或trace导致worker新增约2 GiB”。不能据此排除进程间/共享页/cache等变化或未采样瞬时峰值，也不能定量把差额分配给某个其他进程。

旧完整测试1880周期样本，全程最低5326827520发生epoch1795739480.9159224（08:31:20.915922+08），位于sampling_9_complete→update_9_complete，旧无周期样本低于4 GiB。旧全程最低不是旧首update，故上表采用同阶段对照，没有拿不同轮的after-update点作等价峰值。

## 实际生命周期及尺寸（只读源码）

- `resources.py:63`–110实际创建V10Env/学生/双教师/TaskReservoir后调用runtime.collect_rollout；初始rollout反复送reservoir直到六任务各32768槽满，再del initial并release_host_memory。随后收集完整48步rollout、执行learner.update、记录update、删除rollout/metrics并release，再用累计monitor.summary判门槛。释放后余量恢复不能抹掉此前低点。
- `runtime.py:56`–72预分配CPU FP32 rollout；`runtime.py:93`–135每步从observe获得学生快照，生成同帧标签，存原obs和label。1024×48=49152样本；观测维数45＋225＋3600＋3＋35=3908，动作12，共3920个FP32/样本；纯数值rollout约770703360 bytes（735 MiB），不含Python元数据/metrics、模型/物理环境等。
- `replay.py:24`–47独立六任务32768槽，共196608样本，纯FP32观测/动作约3082813440 bytes（2940 MiB），不含元数据。按任务index_batch产生incoming切片后复制进已分配槽，不预先完整clone整轮；当前最大角色307×48=14736条，其数值incoming约231060480 bytes。fill反复使用初始rollout仅作资源压力，不进入正式训练。
- `dagger.py:14`–30首轮update执行20个逻辑更新；`replay.py:55`–66每逻辑批六任务各256当前＋256历史，共3072样本，数值约48168960 bytes；拼接/旧批/切片生命周期可短暂重叠，microbatch=256约4014080 bytes，加上模型前后向/Adam。末尾reservoir.add(current)仍在update阶段内。没有细粒度CPU内存栈/分配器快照，以上是源码尺寸账而非最低时刻归因。
- `teachers.py:67`–71 blind forward只clone本调用的[N_blind,45]设备proprio，替换6:9并前向；命令输出[N_blind,3]，helper若干[N_blind]临时张量。即使N_blind=1024，单proprio clone也仅184320 bytes。DualTeacher按mask划分的原有obs索引切片仍存在；不能把这个局部clone定性为host新增GB的原因。
- **本probe没有实例化TraceBatchV2**：全v10唯实例化点在`eval_worker.py:102`。resources→runtime.collect_rollout不会进入eval_worker；env追加frame仅暴露现成Nx4x3足坐标、Nx4力/接触、yaw/masks，不持有跨步trace。新预分配trace缓冲及NPZ保存峰值都不可能是本次probe首update的直接内存持有路径。

## 缺失证据及当前快照限制

新旧resources.jsonl均只记录机器MemAvailable及当前worker RSS/GPU；memory_stages补worker RSS高水位。均未记录父进程/各子进程PSS、USS、共享页归属、全机逐进程同期峰值或内存分配调用栈。worker已退出，不能用当前/proc伪造它当时的峰值归因；RSS也不能直接与共享内存逐项相加解释MemAvailable。

11:13:47+08左右只读当前/proc/meminfo显示MemAvailable10565748 kB、Shmem4281008 kB、SwapTotal/Free均0；ps仅确认当前存在多个桌面/浏览器/服务/node进程。这是事后快照，不证明这些进程在11:09:44各占多少，也不能据此指定清理对象。本代理未杀进程、未提出把硬门槛调低、未执行重测。

## 原始证据

- `logs/a2_v10/formal_A1_20261127_approved/resource_probe/n1024_dagger/result.json`：SHA256 `25dcaf003c752e2de5d5fffee74087ea74413a586d33d991328f06094c3583e3`。
- `logs/a2_v10/formal_A1_20261127_approved/resource_probe/n1024_dagger/resources.jsonl`：SHA256 `8c077dc2676769e2d871cac0e7098f9e99dcfc375f5bdaff038fcddcfa454ad1`。
- `logs/a2_v10/formal_A_20261127_approved/resource_probe/n1024_dagger/result.json`：SHA256 `37d422907bd3305ab10ac3da78223b49404b1b96e9446f6415055d8d27d0383e`。
- `logs/a2_v10/formal_A_20261127_approved/resource_probe/n1024_dagger/resources.jsonl`：SHA256 `609d795d7fb79e29dfa8bba0b7b3f2ca009f56eaa44a5b50fd063e494a00542f`。
- `logs/a2_v10/formal_A1_20261127_approved/resource_probe/selection.json`：SHA256 `be01e2618fba9765259a68c2f1f870d91c843db6f101ac02f973ac2226910cd9`。

本报告结论：资源失败证据充分，停止有效；更细的跨进程归因未验证。任何后续处置或新物理必须由主代理/规划在现有授权范围内裁定。
