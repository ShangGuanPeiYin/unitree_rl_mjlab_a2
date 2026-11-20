# A2/6400 上楼梯20次评估

在项目根目录执行：

```bash
/home/xxs/anaconda3/envs/a2/bin/python logs/a2_6400_stair20_20261120/run.py
```

脚本显式固定：seeds=1..20、num_envs=1、device=cuda:0、phase=L、substage=L40、tile_id=21、40级、h=d=0.2m、梯宽0.8m、平台长20m宽4m、vx=0.6、vy=0、wz=0、deterministic=True；权重SHA256在执行前强制校验。每个seed只取构造环境后第一个episode，不额外reset；由环境自身按L标准范围抽样初态，初始关节为默认姿态、速度为零。按原info.episodes首次终止冻结记录，自动重置后的状态不参与该次结果。原成功/失败/超时逻辑没有修改，不执行登顶后的行走或停车评估。运行前请选用新的输出目录或备份本目录，episodes.jsonl为追加写。
