# TCEI 接入调研与对照协议

记录日期：2026-11-06。状态：已核对作者公开实现，提供
[AutoDL 环境/权重预检](tcei-autodl.md)，尚未接入追踪或产生新GPU指标。
当前进球归因代码固定在 `goal-link-v3-baseline-20261106`。

## 来源与机制

- 论文：[Dual-level Adaptation for Multi-Object Tracking: Building Test-Time
  Calibration from Experience and Intuition](https://arxiv.org/abs/2603.21629)。
- 作者仓库：[1941Zpf/TCEI](https://github.com/1941Zpf/TCEI)，核对版本
  `145d1b8431398156f8d9f854430e306fdee39eaa`。
- 判断依据：作者 README、`TCEI/tcei.py`、`models/runtime_tracker.py`、
  `models/motip/id_decoder.py`、`submit_and_evaluate.py` 和 SportsMOT 配置。
  本记录不声称已经逐公式复现论文或验证其报告数值。

实现建立在MOTIP上。对象层使用DETR对象特征，维护视频内正/负缓存；
ID层使用ID解码器末端的ID特征，维护宽范围正/负缓存。根据预测熵、
特征相似性校准ID logits，再做原有ID分配。推理入口和缓存计算使用
`torch.no_grad()`，没有在线反向传播或优化器更新。

本地MOTIP与该版本的 MOTIP、IDDecoder、TrajectoryModeling 三个类的
构造函数 AST 一致；作者主要增加推理特征输出和运行时校准逻辑。
原 SportsMOT checkpoint 有复用条件，仍需实际 strict-load 验证；
不应先换作者checkpoint后把权重差异当作校准增益。

作者将 `pos_cache/neg_cache` 放在单个视频内部，将
`w_pos_cache/w_neg_cache` 放在多视频循环外，因此经验依赖之前处理的视频。
此处的跨视频ID词元缓存不是永久球员身份库，不能替代KPR和球员档案。

## 实现边界

保持上游 `MOTIP/` 原样。适配应放在 `pipeline/tracking/` 或 `adapters/`
中，保持帧输入、TrackRecord输出、旧基准入口不变，并提供默认关闭的开关。
需要取得原DETR对象特征、最终ID特征及关联前logits；具体钩子或运行时
包装方案通过等价性测试后再确定。没有这三项的离线轨迹文件不能完整实现TCEI。

优先复用 MOTIP 环境，作者建议的 Python3.12/PyTorch2.4 与已有路线接近；
先核实环境和算子，不修改球检测的 ball 环境。

需要明示的代码问题，而非静默修正：缓存相似度 min-max 在相同值时除零；
空/单元素缓存；固定类别50与 `.cuda()`；负缓存ID映射（源码已有疑问注释）；
熵计算的对数底不一致；配置 `lambda_object` 在融合表达式中未实际使用。
保留作者逻辑的版本与数值/映射修订版本分开记录，防止将改写称为严格复现。

## 对照顺序

1. **关闭等价性**：相同权重、输入帧、预处理、dtype、阈值、分配协议，
   适配开关关闭后与原版逐帧比较ID、框、分数。不能只看最终总指标接近。
2. **CPU结构/数值检查**：缓存空/单类/重复特征、ID词元复用、序列重置、
   正负缓存边界、无NaN。无GPU时可先完成这部分。
3. **小规模消融**：固定开发序列，跑A0原版、A1对象层、A2 ID层、A3双层。
   参数先采用作者SportsMOT配置；GT仅用于评估，不能更新缓存或选伪标签。
4. **完整已有标注范围**：先SportsMOT篮球val、TeamTrack篮球test与
   TrackID3x3 Indoor，确认后再跑较长的Outdoor。序列列表预先冻结，
   不根据是否提升挑选样本。开发子集与最后确认范围明确区分。
5. **功能回归**：固定进球输入、球检测及归因规则，在同样输入长度下
   比较原版与适配版MOTIP，之后扩展PL-NBA。

## 固定评测条件

- 已有数据集基准采用 **FP32、上游PIL预处理、TF32关闭**；现有功能导出
  使用 `cli.tracking` 默认（FP16、tensor预处理）。两种实验各自配对保持一致。
- 相同 SportsMOT 权重/config、图像列表、分辨率、FPS、阈值、object-max
  分配、漏检容忍、标注转换、忽略区域和TrackEval设置。
- 同一GPU，记录包版本、代码与权重哈希、输出目录、耗时与峰值显存。
  FPS使用双方一致的计时和同步范围，不拿作者不含前10帧的速度直接对比端到端时间。
- 缓存写入只用当前/已处理帧，不用未来帧或GT。明确是否先写当前样本再校准。
- 固定顺序、单GPU起步；局部对象缓存每序列清空。ID经验缓存分别报告
  每序列重置和跨序列保留两种协议。不同数据集/视角域之间不无意继承缓存。
  多GPU会改变每进程经历的视频，不能直接视为同一协议。

## 现有结果参考

| 范围 | 序列 | 帧 | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| SportsMOT 篮球val | 15 | 12557 | 76.94 | 67.79 | 84.19 | 305 |
| TeamTrack 篮球test | 16 | 12706 | 35.11 | 33.81 | 41.57 | 1311 |
| TrackID3x3 Indoor | 42 | 7534 | 62.51 | 58.26 | 79.31 | 769 |
| TrackID3x3 Outdoor | 9 | 103710 | 29.73 | 22.04 | 30.72 | 2361 |

这些历史数值来自已有汇总，完整原始protocol应在服务器读取核对。
SportsMOT仅篮球val，不与论文完整SportsMOT test直接等同比较；
TrackID下载的带标注固定镜头范围不声称为官方test split。

主要看AssA、IDF1、HOTA、IDSW、Frag，同时记录DetA、MOTA、FP/FN和速度。
校准重点在身份关联，未重训检测器；Outdoor高FP问题不能预先承诺由此解决。
不能只统计新ID数量少了就判定好，因为误合并也会减少ID。

## 回到进球功能时的正确比较

当前10段由整段MOTIP缓存截取窗口。如果下一版改成5秒冷启动，必须同时
重跑原版5秒作为A组，不能直接用历史整段缓存作为对照。
改变追踪器后局部T号会变化；人工表的T6/T4/T13不是模型无关真值。
应按源视频帧、人工人物位置与身份核对，不能用新旧T号相同作为正确标准。

球检测/进球时刻可复用；依赖人物框和ID的动作、局部裁剪、身份特征及
归因需重算或严格重新映射，不能把旧T号上的动作和KPR特征直接套给新T号。
保持下游归因规则不变，记录事件正确率/覆盖率、局部轨迹纯度、可用身份
图片率、运行时间，才可判断MOT指标提升是否真正帮助高光系统。
