# TeamTrack 篮球 MOTIP 迁移评估

本入口评估 SportsMOT 预训练 MOTIP 在 TeamTrack 的迁移效果。它采用原论文相同的五项指标和仓库自带 TrackEval；TeamTrack 的结果不能称为复现 SportsMOT 测试集的 72.6 HOTA / 83.5 DetA / 63.2 AssA / 92.4 MOTA / 77.1 IDF1。

## 输入与范围

数据根目录包含 `selection.csv`，至少有 `view,split,video` 三列。支持 `basketball_side`、`basketball_side_2`、`basketball_top`，拆分为 `val`、`test`。每行对应：

```text
<dataset>/<view>/<split>/videos/<name>.mp4
<dataset>/<view>/<split>/annotations/<name>.csv
```

上传名单应有 17 条 val、16 条 test。程序按实际 selection.csv 的名单处理，结果记录实际序列数。先用 val 检查标注对齐，再用固定设置测试 test；两者各自生成分视角与总体指标，不混成一份正式测试得分。

四行 CSV 表头必须为 TeamID、PlayerID、Attributes、frame。框属性按名称读取，顺序不限，保留像素 xywh；身份用 `(TeamID, PlayerID)` 映射为序列内唯一整数。排除 BALL。完整缺失框（四项都是空或 NaN）跳过；部分缺失、非正尺寸、重复属性、非连续或非一基帧号报错。超出图像边界的正尺寸框保留并计数，不擅自裁剪。

首次准备会完整解码视频，要求 CSV 行数、实际解码帧数和容器声明帧数一致。首、中、末帧叠加 GT 图保存在 alignment 下。帧数一致只能证明长度对齐，还需人工看图检查坐标和时间是否对齐。空白帧也参与统计，不以最后一个检测框确定视频长度。

## 安装与运行

在控制环境中安装 CPU 依赖（不改变 MOTIP 的 PyTorch 环境）：

```bash
python -m pip install -r requirements-teamtrack.txt
python -m cli.check_teamtrack
```

以下命令在仓库根目录执行；路径均为示例。私有 env 文件提供 `BASKET_MOTIP_PYTHON`、`BASKET_MOTIP_CHECKPOINT`。`--stage all` 会通过该 Python 启动独立的 GPU 推理子进程，其余步骤在当前控制环境运行。

```bash
python -u -m cli.benchmark_teamtrack \
  --env-file .env \
  --dataset-root /data/teamtrack-basketball \
  --output-dir /outputs/teamtrack-baseline \
  --split val --stage prepare
```

查看 `val/alignment/` 后运行全链路：

```bash
python -u -m cli.benchmark_teamtrack \
  --env-file .env \
  --dataset-root /data/teamtrack-basketball \
  --output-dir /outputs/teamtrack-baseline \
  --split val
```

把 `--split val` 换成 `--split test` 运行正式测试。可以在循环中顺序运行两个 split，避免在同一块 GPU 上竞争。不要对同一个输出目录同时启动多个任务。

- `prepare`：转换 GT、完整解码计数、导出对齐图；不占 GPU。
- `infer`：仅推理，需直接使用 MOTIP 环境的 Python，且先完成 prepare。
- `evaluate`：仅计算指标，在控制环境运行，需完整推理结果。
- `all`（默认）：按顺序执行三步。相同命令可恢复中断的工作。

恢复时校验数据和 GT、模型权重、源码、运行配置和已完成预测的 SHA256。只复用完整序列；中断序列从头重跑，因为没有序列内 tracker 状态快照。输入或推理协议发生变化时使用新输出目录。正式评估缺少任何一个序列就失败，不悄悄按成功子集出总分。

## 推理协议

使用 `r50_deformable_detr_motip_sportsmot.pth` 与对应配置（带轨迹增强版本），FP32、object-max、miss_tolerance=60、det=0.3、newborn=0.6、id=0.2、area=0，短边 800、长边上限 1440。

预处理复用上游 `MOTIP/data/seq_dataset.py` 的 PIL Resize → ToImage → FP32 → Normalize 顺序；视频由 OpenCV 解码再转 RGB PIL，不提取成有损 JPEG。保留原帧率、整段连续追踪；每条序列重置 tracker，同一 split 复用模型。直接导出 raw tracker 框和 ID，不做业务质量过滤、GT 过滤、球员数 top-k 限制、KPR/Qwen 或跨视频身份合并。模型在球员之外检测到的裁判等对象也可能贡献 FP。

4K 普通视角与鱼眼视角在同一长边限制下有效人物分辨率不同，因此分别报告。改变分辨率、阈值或权重属于新实验；先在 val 做选择，再冻结设置运行 test。多视角片段可能来自同一比赛，不应被解释为独立比赛样本。

## 输出和口径

每个 split 目录都有：

| 文件 | 内容 |
|---|---|
| `dataset.json` | 名单、源文件校验值、帧数、尺寸、GT 数量及异常框数量 |
| `gt/<sequence>/gt/gt.txt` | 标准 MOT GT（球员类=1，参与评估=1） |
| `gt/<sequence>/identity_map.json` | MOT ID 与 TeamID/PlayerID 的对应关系 |
| `alignment/<sequence>/*.jpg` | 首、中、末帧 GT 叠加图 |
| `inference_protocol.json` | 权重 SHA256、参数、依赖版本和推理源码校验值 |
| `inference_environment.json` | Git 提交、GPU 和 CUDA 版本 |
| `predictions/motip/data/<sequence>.txt` | 原始预测的标准 MOT 文件 |
| `completed/<sequence>.json` | 完成标记、处理帧数、预测数量、耗时和校验值 |
| `metrics/summary.csv` | 每序列、每视角、整个 split 的主要指标和计数 |
| `metrics/hota_thresholds.csv` | 每个范围在 19 个 IoU 阈值下的指标和 HOTA TP/FP/FN |
| `metrics/sequences/<sequence>/clear_frames.csv` | 每帧及累计 TP/FP/FN/IDSW、recall、precision |
| `metrics/sequences/<sequence>/clear_matches.csv` | 每个 TP/FP/FN 的原始 GT ID、tracker ID、匹配 IoU 和 ID 切换标记 |
| `metrics/trackeval_raw.json` | TrackEval 的全部原始返回字段，包括数组 |
| `metrics/metric_protocol.json` | 评估协议及 TrackEval 源码校验值 |
| `evaluation_status.json` | 只有 completed 状态及匹配的 summary 校验值表示此次评估完整结束 |

CSV 中的得分、recall、precision 都是百分数（100 表示完全正确）；原始 JSON 中得分为比例。计数是整数。没有 GT/预测的空分母按 TrackEval 的 max(1, denominator) 约定处理。

`summary.csv` 的 scope 为 `sequence`、`view`、`overall`；overall 的 sequence 为 `COMBINED_SEQ`，view 为 all。总分调用 TrackEval `combine_sequences`，按各指标定义合并，**不平均各视频百分比**。

- HOTA、DetA、AssA：对 IoU=0.05、0.10、…、0.95 的 19 个值取均值。HOTA_TP/FP/FN 随阈值变化，单独导出，不能和 CLEAR 计数互换。
- MOTA、TP、FP、FN、recall、precision、IDSW、Frag：TrackEval CLEAR，IoU=0.5。recall=TP/(TP+FN)，precision=TP/(TP+FP)，MOTA 计入 FP、FN、IDSW，可为负值。
- IDF1、IDTP、IDFP、IDFN、IDR、IDP：TrackEval Identity，IoU=0.5，以整个序列的身份分配计算。IDF1 与 AssA 不是逐帧检测指标，不伪造“逐帧 IDF1/HOTA”。

逐帧明细复现仓库自带 CLEAR 的匹配规则（优先上一时刻关联、再 IoU，包括其空帧状态处理），逐序列与官方 CLEAR 的 TP/FP/FN/IDSW 总计交叉校验，不一致就报错。它描述的是原始 MOTIP 输出，不是业务流程中 quality、identity 等阶段的准确率。

转换 GT 没有提供 MOT17 distractor/ignore 类，因此使用 `DO_PREPROC=False`，所有有效球员框参与评估。不能使用 GT 身份提前删除多余预测。其余指标由项目现有的上游 TrackEval 实现直接计算。

## 验证范围与参考

CPU 自测包含 CSV 身份映射、BALL 排除、缺失框、帧号错误、真实小视频解码与长度不一致拒绝、复用/文件变更保护，以及真实 TrackEval 下的完美轨迹、漏检、误检、ID 切换、空预测、空 GT 和加权汇总。CPU 自测不证明 TeamTrack 上的实际准确率；真实结果需在模型环境运行后取得。

- [MOTIP 官方评估入口](https://github.com/MCG-NJU/MOTIP/blob/main/submit_and_evaluate.py)
- [MOTIP 模型与论文结果](https://github.com/MCG-NJU/MOTIP/blob/main/docs/MODEL_ZOO.md)
- [TeamTrack 数据说明](https://github.com/AtomScott/TeamTrack)
