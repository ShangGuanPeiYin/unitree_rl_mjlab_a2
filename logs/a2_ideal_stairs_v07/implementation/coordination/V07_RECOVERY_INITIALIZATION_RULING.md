# v0.7 恢复初始化语义澄清 Qv07-001

主决策线程01a1d49c-5f5d-7890-ac95-9234b307139a已明确裁定，原SDS SHA303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca不变。本文件作为最终验收的补充绑定文件。

严格继承C1已批准顺序：**恢复来源checkpoint完整RNG → 原schedule一次 → 原ppo.reset一次**。允许变化槽内部初始化后再全量new_episode的额外随机消耗；不得加seed、回卷或拼接RNG，不声称物理位精确。

首迁移generation72→73仅一次。v07后续load保留已存generation、generation_updates、谨慎退级评估队列及保持时钟，不重新获得1000保护期。

用当前23tile课程顺序安装正确phase/substage/stage_iteration/512角色，禁止训练tile_override。首/二load均记录四时点RNG、reset槽次数、最终观测/奖励/终止缓存；临时reset不计学习、计费或课程结果。模型/完整Adam/normalizer/学习计费保持；若发现额外未清统计副作用，保存证据先上报，不自行调整。
