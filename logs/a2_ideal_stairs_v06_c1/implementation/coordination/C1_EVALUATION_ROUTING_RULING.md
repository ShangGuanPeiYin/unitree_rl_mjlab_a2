# C1评估路由必要接线裁决

日期：2026-11-17。来源：主决策Codex线程01a1d49c-5f5d-7890-ac95-9234b307139a。状态：明确批准，正式训练仍须最终同SHA完整验收。

问题：case生成器不写substage，旧worker将case缺失substage回退为创建closure时的值。C1从D25创建长期closure，双保持和后续D20/L阶段因此可能被错误传入D25子级。

批准边界：生产评估以每次case身份为准。flat保持物理F0/None/tile0；stair保持物理H/None/H00。普通D/L从已验证tile的spec.lesson_id推导对应phase子级；其余phase子级None。显式case.substage与物理身份冲突拒绝，不从closure补救。保持case.phase仍可为训练来源阶段。

逐case核对kind、phase、measurement_stage、tile、lesson、速度与批协议一致；缺tile/phase或lesson冲突拒绝，不能只看首case。案例生成顺序、出生、种子、case表hash及目标地形不改。

验证要求：长期D25 closure连续调用D20/L12/L20/L40/V/R及双保持；真实env核验恢复后到期双保持可构造。所有结果只是工程证据，不冒称已执行正式4400保持或已取得其成绩。此修复纳入最终新冻结、完整CPU、同SHA工程GPU和独立验收；不改变奖励、PPO、课程、成功/安全阈值。
