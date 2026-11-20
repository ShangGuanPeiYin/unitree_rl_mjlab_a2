# 原6400：S1.0 gate_A实测报告

完成日期：2026-11-20 13:24:44 CST。本报告仅是原始教师对照，不代表v0.9恢复或改善。

固定75组×20例，共1500例；成功1051，物理/质量失败448，超时1。实际环境控制样本817,400（含所有已完成slot等待步），物理子步3,269,600。该义务已结账，学习0。

源6400 SHA：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。生产源码SHA：`44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f`。病例manifest SHA：`94f85c887889a2444682085688d605017d5117e4b0becc03dd885b6fadd6fbe1`。

结果证据：`logs/a2_v09/s1_seed42_r3/evidence/teacher_gate_A/episodes.jsonl`、`summary.json`及各批PT/geometry。完整连续指标、有效覆盖与带符号距离分位数见本机`logs/a2_v09/s1_seed42_r3/teacher_gate_A_report.json`。

失败分类：quality_failure=13, side=102, nonfoot=256, posture=39, timeout=1, body=38。全部病例保留。

## 75组完整分母

|任务|旧课程|速度m/s|停车方式|成功/20|Wilson95|终止原因（含成功）|
|---|---|---:|---|---:|---|---|
|flat_walk|FLAT|0.3|none|20/20|0.839–1.000|success:20|
|flat_walk|FLAT|0.5|none|20/20|0.839–1.000|success:20|
|flat_walk|FLAT|0.6|none|20/20|0.839–1.000|success:20|
|flat_stop|FLAT|0.3|instant|20/20|0.839–1.000|success:20|
|flat_stop|FLAT|0.5|instant|18/20|0.699–0.972|success:18, quality_failure:2|
|flat_stop|FLAT|0.6|instant|15/20|0.531–0.888|success:15, quality_failure:5|
|flat_stop|FLAT|0.3|ramp_2s|20/20|0.839–1.000|success:20|
|flat_stop|FLAT|0.5|ramp_2s|20/20|0.839–1.000|success:20|
|flat_stop|FLAT|0.6|ramp_2s|14/20|0.481–0.855|success:14, quality_failure:6|
|stairs|H00|0.3|none|0/20|0.000–0.161|side:16, nonfoot:2, posture:2|
|stairs|H00|0.5|none|0/20|0.000–0.161|nonfoot:1, side:18, posture:1|
|stairs|H00|0.6|none|1/20|0.009–0.236|side:13, nonfoot:4, posture:2, success:1|
|stairs|H01|0.3|none|1/20|0.009–0.236|side:7, nonfoot:7, posture:5, success:1|
|stairs|H01|0.5|none|3/20|0.052–0.360|success:3, nonfoot:5, side:8, posture:4|
|stairs|H01|0.6|none|4/20|0.081–0.416|side:7, success:4, posture:3, nonfoot:6|
|stairs|H02|0.3|none|7/20|0.181–0.567|success:7, side:7, nonfoot:1, posture:5|
|stairs|H02|0.5|none|14/20|0.481–0.855|success:14, side:4, nonfoot:1, posture:1|
|stairs|H02|0.6|none|20/20|0.839–1.000|success:20|
|stairs|H03|0.3|none|11/20|0.342–0.742|success:11, side:7, posture:2|
|stairs|H03|0.5|none|17/20|0.640–0.948|success:17, posture:2, nonfoot:1|
|stairs|H03|0.6|none|20/20|0.839–1.000|success:20|
|stairs|H04|0.3|none|7/20|0.181–0.567|success:7, nonfoot:7, timeout:1, posture:2, side:3|
|stairs|H04|0.5|none|16/20|0.584–0.919|success:16, posture:2, side:1, nonfoot:1|
|stairs|H04|0.6|none|17/20|0.640–0.948|success:17, nonfoot:2, side:1|
|stairs|H05|0.3|none|5/20|0.112–0.469|nonfoot:11, posture:2, success:5, body:1, side:1|
|stairs|H05|0.5|none|10/20|0.299–0.701|side:3, nonfoot:4, success:10, posture:3|
|stairs|H05|0.6|none|18/20|0.699–0.972|success:18, side:1, nonfoot:1|
|stairs|H06|0.3|none|1/20|0.009–0.236|nonfoot:16, success:1, posture:1, side:2|
|stairs|H06|0.5|none|8/20|0.219–0.613|nonfoot:11, success:8, posture:1|
|stairs|H06|0.6|none|11/20|0.342–0.742|success:11, nonfoot:9|
|stairs|H07|0.3|none|3/20|0.052–0.360|nonfoot:15, success:3, side:2|
|stairs|H07|0.5|none|7/20|0.181–0.567|nonfoot:13, success:7|
|stairs|H07|0.6|none|7/20|0.181–0.567|nonfoot:13, success:7|
|stairs|H08|0.3|none|6/20|0.145–0.519|success:6, nonfoot:11, side:1, body:2|
|stairs|H08|0.5|none|9/20|0.258–0.658|nonfoot:11, success:9|
|stairs|H08|0.6|none|7/20|0.181–0.567|nonfoot:12, success:7, posture:1|
|stairs|H09|0.3|none|3/20|0.052–0.360|nonfoot:16, success:3, body:1|
|stairs|H09|0.5|none|9/20|0.258–0.658|success:9, nonfoot:10, body:1|
|stairs|H09|0.6|none|7/20|0.181–0.567|nonfoot:13, success:7|
|stairs|H10|0.3|none|14/20|0.481–0.855|success:14, body:5, nonfoot:1|
|stairs|H10|0.5|none|3/20|0.052–0.360|nonfoot:16, success:3, body:1|
|stairs|H10|0.6|none|1/20|0.009–0.236|nonfoot:19, success:1|
|stairs|H11|0.3|none|14/20|0.481–0.855|success:14, body:6|
|stairs|H11|0.5|none|19/20|0.764–0.991|success:19, nonfoot:1|
|stairs|H11|0.6|none|12/20|0.387–0.781|success:12, body:1, nonfoot:7|
|stairs|H12|0.3|none|14/20|0.481–0.855|nonfoot:3, success:14, body:3|
|stairs|H12|0.5|none|20/20|0.839–1.000|success:20|
|stairs|H12|0.6|none|20/20|0.839–1.000|success:20|
|stairs|H13|0.3|none|15/20|0.531–0.888|success:15, body:5|
|stairs|H13|0.5|none|20/20|0.839–1.000|success:20|
|stairs|H13|0.6|none|20/20|0.839–1.000|success:20|
|stairs|H14|0.3|none|14/20|0.481–0.855|body:6, success:14|
|stairs|H14|0.5|none|20/20|0.839–1.000|success:20|
|stairs|H14|0.6|none|20/20|0.839–1.000|success:20|
|stairs|H15|0.3|none|14/20|0.481–0.855|success:14, body:1, nonfoot:5|
|stairs|H15|0.5|none|20/20|0.839–1.000|success:20|
|stairs|H15|0.6|none|20/20|0.839–1.000|success:20|
|stairs|D25|0.3|none|19/20|0.764–0.991|success:19, body:1|
|stairs|D25|0.5|none|20/20|0.839–1.000|success:20|
|stairs|D25|0.6|none|20/20|0.839–1.000|success:20|
|stairs|D20|0.3|none|20/20|0.839–1.000|success:20|
|stairs|D20|0.5|none|19/20|0.764–0.991|success:19, body:1|
|stairs|D20|0.6|none|19/20|0.764–0.991|success:19, body:1|
|stairs|L12|0.3|none|20/20|0.839–1.000|success:20|
|stairs|L12|0.5|none|20/20|0.839–1.000|success:20|
|stairs|L12|0.6|none|20/20|0.839–1.000|success:20|
|stairs|L20|0.3|none|20/20|0.839–1.000|success:20|
|stairs|L20|0.5|none|20/20|0.839–1.000|success:20|
|stairs|L20|0.6|none|20/20|0.839–1.000|success:20|
|stairs|L40|0.3|none|20/20|0.839–1.000|success:20|
|stairs|L40|0.5|none|20/20|0.839–1.000|success:20|
|stairs|L40|0.6|none|20/20|0.839–1.000|success:20|
|stairs|D22|0.3|none|20/20|0.839–1.000|success:20|
|stairs|D22|0.5|none|20/20|0.839–1.000|success:20|
|stairs|D22|0.6|none|18/20|0.699–0.972|success:18, body:2|

## 平地完整窗口连续指标

组内速度、pitch、滑移、动作变化、漂移按每例完整有效窗口指标等权平均；步幅、步频按每例指标的组中位数。有效数单独报告，缺测不当零。Wilson区间仅报告不改变门槛。

|任务/方式/速度|速度RMSE m/s（有效数）|pitch RMS °（有效数）|步幅m（有效数）|步频Hz（有效数）|漂移m（有效数）|
|---|---|---|---|---|---|
|flat_walk/none/0.3|0.0290177 (20/20)|1.44018 (20/20)|0.15683 (20/20)|2.08333 (20/20)|—|
|flat_walk/none/0.5|0.0268671 (20/20)|1.73628 (20/20)|0.226458 (20/20)|2.27273 (20/20)|—|
|flat_walk/none/0.6|0.0286767 (20/20)|1.94431 (20/20)|0.258212 (20/20)|2.38095 (20/20)|—|
|flat_stop/instant/0.3|0.0298235 (20/20)|1.44391 (20/20)|0.156923 (20/20)|2.08333 (20/20)|0.0161279 (20/20)|
|flat_stop/instant/0.5|0.0274084 (20/20)|1.72947 (20/20)|0.226835 (20/20)|2.27273 (20/20)|0.0265241 (20/20)|
|flat_stop/instant/0.6|0.0283929 (20/20)|1.94348 (20/20)|0.258685 (20/20)|2.38095 (20/20)|0.0339041 (20/20)|
|flat_stop/ramp_2s/0.3|0.0290409 (20/20)|1.43882 (20/20)|0.156796 (20/20)|2.08333 (20/20)|0.0254182 (20/20)|
|flat_stop/ramp_2s/0.5|0.0256616 (20/20)|1.73265 (20/20)|0.227176 (20/20)|2.27273 (20/20)|0.0250249 (20/20)|
|flat_stop/ramp_2s/0.6|0.028039 (20/20)|1.943 (20/20)|0.259108 (20/20)|2.38095 (20/20)|0.0280622 (20/20)|
