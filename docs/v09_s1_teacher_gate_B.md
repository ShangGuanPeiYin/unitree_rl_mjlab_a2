# 原6400：S1.0 gate_B实测报告

完成日期：2026-11-20 13:49:59 CST。本报告仅是原始教师对照，不代表v0.9恢复或改善。

固定75组×20例，共1500例；成功1037，物理/质量失败460，超时3。实际环境控制样本869,800（含所有已完成slot等待步），物理子步3,479,200。该义务已结账，学习0。

源6400 SHA：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。生产源码SHA：`44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f`。病例manifest SHA：`2024a49022bc169f33ccc121654f50c8c2ea116b791c74dfd5f3bcb9f86ac510`。

证据：`logs/a2_v09/s1_seed42_r3/evidence/teacher_gate_B/episodes.jsonl`、summary及各批PT/geometry。完整连续指标、有效覆盖与带符号距离分位数见本机`logs/a2_v09/s1_seed42_r3/teacher_gate_B_report.json`。

失败分类：quality_failure=16, side=90, posture=57, nonfoot=253, timeout=3, body=44。全部病例保留。

## 75组完整分母

|任务|旧课程|速度m/s|停车方式|成功/20|Wilson95|终止原因（含成功）|
|---|---|---:|---|---:|---|---|
|flat_walk|FLAT|0.3|none|20/20|0.839–1.000|success:20|
|flat_walk|FLAT|0.5|none|20/20|0.839–1.000|success:20|
|flat_walk|FLAT|0.6|none|20/20|0.839–1.000|success:20|
|flat_stop|FLAT|0.3|instant|20/20|0.839–1.000|success:20|
|flat_stop|FLAT|0.5|instant|18/20|0.699–0.972|success:18, quality_failure:2|
|flat_stop|FLAT|0.6|instant|13/20|0.433–0.819|quality_failure:7, success:13|
|flat_stop|FLAT|0.3|ramp_2s|20/20|0.839–1.000|success:20|
|flat_stop|FLAT|0.5|ramp_2s|20/20|0.839–1.000|success:20|
|flat_stop|FLAT|0.6|ramp_2s|13/20|0.433–0.819|quality_failure:7, success:13|
|stairs|H00|0.3|none|1/20|0.009–0.236|side:17, success:1, posture:1, nonfoot:1|
|stairs|H00|0.5|none|1/20|0.009–0.236|side:15, success:1, posture:3, nonfoot:1|
|stairs|H00|0.6|none|0/20|0.000–0.161|posture:3, side:14, nonfoot:2, timeout:1|
|stairs|H01|0.3|none|1/20|0.009–0.236|posture:9, side:7, nonfoot:3, success:1|
|stairs|H01|0.5|none|5/20|0.112–0.469|posture:12, success:5, side:3|
|stairs|H01|0.6|none|4/20|0.081–0.416|nonfoot:4, side:7, posture:5, success:4|
|stairs|H02|0.3|none|5/20|0.112–0.469|success:5, side:6, posture:5, nonfoot:4|
|stairs|H02|0.5|none|16/20|0.584–0.919|success:16, side:2, posture:2|
|stairs|H02|0.6|none|20/20|0.839–1.000|success:20|
|stairs|H03|0.3|none|12/20|0.387–0.781|success:12, side:3, posture:2, nonfoot:2, body:1|
|stairs|H03|0.5|none|15/20|0.531–0.888|success:15, nonfoot:1, posture:3, side:1|
|stairs|H03|0.6|none|20/20|0.839–1.000|success:20|
|stairs|H04|0.3|none|6/20|0.145–0.519|nonfoot:9, success:6, side:2, posture:2, timeout:1|
|stairs|H04|0.5|none|17/20|0.640–0.948|success:17, nonfoot:2, posture:1|
|stairs|H04|0.6|none|20/20|0.839–1.000|success:20|
|stairs|H05|0.3|none|5/20|0.112–0.469|nonfoot:13, success:5, timeout:1, side:1|
|stairs|H05|0.5|none|10/20|0.299–0.701|side:3, success:10, nonfoot:4, posture:3|
|stairs|H05|0.6|none|18/20|0.699–0.972|nonfoot:1, success:18, side:1|
|stairs|H06|0.3|none|0/20|0.000–0.161|nonfoot:17, side:2, posture:1|
|stairs|H06|0.5|none|8/20|0.219–0.613|nonfoot:9, side:3, success:8|
|stairs|H06|0.6|none|10/20|0.299–0.701|success:10, nonfoot:10|
|stairs|H07|0.3|none|1/20|0.009–0.236|side:2, nonfoot:14, posture:2, body:1, success:1|
|stairs|H07|0.5|none|5/20|0.112–0.469|success:5, nonfoot:15|
|stairs|H07|0.6|none|7/20|0.181–0.567|nonfoot:13, success:7|
|stairs|H08|0.3|none|7/20|0.181–0.567|nonfoot:12, success:7, posture:1|
|stairs|H08|0.5|none|14/20|0.481–0.855|success:14, nonfoot:6|
|stairs|H08|0.6|none|8/20|0.219–0.613|nonfoot:12, success:8|
|stairs|H09|0.3|none|3/20|0.052–0.360|nonfoot:13, posture:1, body:2, success:3, side:1|
|stairs|H09|0.5|none|6/20|0.145–0.519|nonfoot:13, success:6, body:1|
|stairs|H09|0.6|none|3/20|0.052–0.360|nonfoot:15, body:2, success:3|
|stairs|H10|0.3|none|9/20|0.258–0.658|body:5, nonfoot:5, success:9, posture:1|
|stairs|H10|0.5|none|1/20|0.009–0.236|nonfoot:19, success:1|
|stairs|H10|0.6|none|2/20|0.028–0.301|nonfoot:18, success:2|
|stairs|H11|0.3|none|15/20|0.531–0.888|body:4, success:15, nonfoot:1|
|stairs|H11|0.5|none|19/20|0.764–0.991|success:19, nonfoot:1|
|stairs|H11|0.6|none|16/20|0.584–0.919|success:16, nonfoot:4|
|stairs|H12|0.3|none|9/20|0.258–0.658|body:8, success:9, nonfoot:3|
|stairs|H12|0.5|none|20/20|0.839–1.000|success:20|
|stairs|H12|0.6|none|20/20|0.839–1.000|success:20|
|stairs|H13|0.3|none|15/20|0.531–0.888|success:15, body:4, nonfoot:1|
|stairs|H13|0.5|none|20/20|0.839–1.000|success:20|
|stairs|H13|0.6|none|19/20|0.764–0.991|success:19, body:1|
|stairs|H14|0.3|none|16/20|0.584–0.919|success:16, body:4|
|stairs|H14|0.5|none|20/20|0.839–1.000|success:20|
|stairs|H14|0.6|none|19/20|0.764–0.991|success:19, body:1|
|stairs|H15|0.3|none|14/20|0.481–0.855|success:14, body:3, nonfoot:3|
|stairs|H15|0.5|none|19/20|0.764–0.991|success:19, nonfoot:1|
|stairs|H15|0.6|none|19/20|0.764–0.991|nonfoot:1, success:19|
|stairs|D25|0.3|none|20/20|0.839–1.000|success:20|
|stairs|D25|0.5|none|20/20|0.839–1.000|success:20|
|stairs|D25|0.6|none|20/20|0.839–1.000|success:20|
|stairs|D20|0.3|none|20/20|0.839–1.000|success:20|
|stairs|D20|0.5|none|20/20|0.839–1.000|success:20|
|stairs|D20|0.6|none|20/20|0.839–1.000|success:20|
|stairs|L12|0.3|none|20/20|0.839–1.000|success:20|
|stairs|L12|0.5|none|20/20|0.839–1.000|success:20|
|stairs|L12|0.6|none|19/20|0.764–0.991|success:19, body:1|
|stairs|L20|0.3|none|20/20|0.839–1.000|success:20|
|stairs|L20|0.5|none|18/20|0.699–0.972|success:18, body:2|
|stairs|L20|0.6|none|18/20|0.699–0.972|success:18, body:2|
|stairs|L40|0.3|none|20/20|0.839–1.000|success:20|
|stairs|L40|0.5|none|20/20|0.839–1.000|success:20|
|stairs|L40|0.6|none|19/20|0.764–0.991|success:19, body:1|
|stairs|D22|0.3|none|20/20|0.839–1.000|success:20|
|stairs|D22|0.5|none|20/20|0.839–1.000|success:20|
|stairs|D22|0.6|none|19/20|0.764–0.991|success:19, body:1|

## 平地完整窗口连续指标

组内速度、pitch、滑移、动作变化、漂移按每例完整有效窗口指标等权平均；步幅、步频按每例指标的组中位数。有效数单独报告，缺测不当零。Wilson区间仅报告不改变门槛。

|任务/方式/速度|速度RMSE m/s（有效数）|pitch RMS °（有效数）|步幅m（有效数）|步频Hz（有效数）|漂移m（有效数）|
|---|---|---|---|---|---|
|flat_walk/none/0.3|0.0291031 (20/20)|1.44113 (20/20)|0.156863 (20/20)|2.08333 (20/20)|—|
|flat_walk/none/0.5|0.0265959 (20/20)|1.73996 (20/20)|0.226338 (20/20)|2.27273 (20/20)|—|
|flat_walk/none/0.6|0.0286386 (20/20)|1.94438 (20/20)|0.258066 (20/20)|2.38095 (20/20)|—|
|flat_stop/instant/0.3|0.0296535 (20/20)|1.44901 (20/20)|0.157229 (20/20)|2.08333 (20/20)|0.0150628 (20/20)|
|flat_stop/instant/0.5|0.0257372 (20/20)|1.73356 (20/20)|0.227303 (20/20)|2.27273 (20/20)|0.0295981 (20/20)|
|flat_stop/instant/0.6|0.0290323 (20/20)|1.94119 (20/20)|0.25807 (20/20)|2.38095 (20/20)|0.0365231 (20/20)|
|flat_stop/ramp_2s/0.3|0.0298458 (20/20)|1.45053 (20/20)|0.157155 (20/20)|2.08333 (20/20)|0.0257396 (20/20)|
|flat_stop/ramp_2s/0.5|0.0261803 (20/20)|1.7329 (20/20)|0.227233 (20/20)|2.27273 (20/20)|0.0250504 (20/20)|
|flat_stop/ramp_2s/0.6|0.0286141 (20/20)|1.94243 (20/20)|0.259022 (20/20)|2.38095 (20/20)|0.0336852 (20/20)|
