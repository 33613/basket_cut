# SportsMOT 篮球原始帧对照测试

先检验官方入口与项目适配器是否一致，再判断 SportsMOT 权重在源数据上的篮球效果。默认处理本地包中排序前5条官方val篮球序列；初步对照通过后，再在新输出目录运行全部序列。

## 数据与安装

数据根目录下保留 `SportsMOT/val/<sequence>/{img1,gt/gt.txt,seqinfo.ini}`，以及官方 `basketball.txt`、`val.txt`。名单可以存放在根目录的子目录中；若有多份同名名单，其内容必须一致。外层可以有一层打包目录，程序自动定位。篮球序列必须同时属于两个官方名单。原始帧应连续命名为 `000001.jpg` 等。

在控制环境安装 `requirements-teamtrack.txt`（包含原始帧验证、CPU TrackEval依赖），在现有MOTIP环境安装 `requirements-motip-reference.txt`。上游CLI即使禁用W&B也会import wandb，所以需要该依赖；运行时禁用W&B记录。无需另装PyTorch，不修改上游源码。

以下命令从仓库根目录运行，路径为示例：

```bash
python -m cli.check_sportsmot
python -u -m cli.benchmark_sportsmot \
  --env-file .env \
  --dataset-root /data/sportsmot-basketball-val \
  --output-dir /outputs/sportsmot-parity-5 \
  --limit 5 --mode parity
```

控制Python从私有env读取 `BASKET_MOTIP_PYTHON` 和 `BASKET_MOTIP_CHECKPOINT`，启动独立GPU子进程。GPU工作者限定单GPU的cuda:0，两条路径依次执行，避免争用显存。

也可以一条命令依次完成前5条对照与全部篮球val的官方入口评估。前一步发生错误或发现差异时，自动停止，保留结果，不执行后一步：

```bash
python -u -m cli.benchmark_sportsmot_suite \
  --env-file .env \
  --dataset-root /data/sportsmot-basketball-val \
  --output-root /outputs/sportsmot-native
```

两阶段分别保存在 `<output-root>/parity-5`、`<output-root>/official-all`。查看根目录 `suite_status.json` 判断整批进度；completed表示两个阶段均成功，stopped表示在stage所指阶段失败。两阶段分别保存完整结果，官方前5条会在全量阶段再计算一次，避免跨实验隐式复用。

## 首轮对照做什么

1. 用官方篮球名单与val名单交集确定下载序列，按序列名排序取前N条。验证所有所选JPEG的尺寸、RGB模式和可解码性，校验帧数、编号、GT范围、框尺寸及身份重复；计算逐帧/GT/seqinfo SHA256，保存数据清单。
2. 官方路径每条序列直接调用未修改的 `MOTIP/submit_and_evaluate.py`，使用 `submit` 模式。通过只读符号链接提供原始图像和标注，不重编码、不转换视频，不裁剪GT或预测框。
3. 适配器路径使用同样的原始JPEG、FP32和PIL预处理，读取方式为OpenCV JPEG→RGB PIL。模型可复用，每条序列重新初始化tracker。两个路径都关闭matmul及cuDNN TF32，阈值为det=0.3、newborn=0.6、id=0.2，object-max，miss_tolerance=60，短边800、长边上限1440。
4. 在适配器路径的首、中、末帧，将实际预处理张量与上游 `SeqDataset` 返回值比较，同时记录PIL/OpenCV解码像素差异。
5. 每个路径都用仓库自带TrackEval计算HOTA/CLEAR/Identity及逐帧CLEAR明细。采用官方GT预处理开关 `DO_PREPROC=True`。当前入口要求SportsMOT的GT行全部为有效球员（标记=1、类别=1）；遇到ignore/其他类别会停下要求核对协议，不擅自改标注。
6. 比较每帧预测数量、框坐标、全序列ID对应关系及主要指标。允许两条路径仅对整个序列的ID作一致重编号；不允许同一条轨迹中途变化。官方输出分数列为1，适配器保留实际置信度，因此不比较该列，也不额外做分数过滤。

默认对照容差：框坐标最大绝对差0.01像素，采样预处理张量最大绝对差1e-6，主要分数差0.01个百分点，TP/FP/FN/IDSW/IDTP/IDFP/IDFN计数要求一致。这些是定位实现差异的工程容差，不是准确率达标线。任何不一致都会保留明细，程序非零退出，先检查差异再扩展实验。

## 输出

```text
<output>/
  dataset.json
  experiment_protocol.json
  official/val/
    inference_protocol.json
    environment.json
    logs/<sequence>.log
    logs/<sequence>.command.json
    completed/<sequence>.json
    predictions/motip/data/<sequence>.txt
    metrics/summary.csv
    metrics/hota_thresholds.csv
    metrics/sequences/<sequence>/clear_frames.csv
    metrics/sequences/<sequence>/clear_matches.csv
  adapter/val/                     # 相同结果结构；没有官方CLI日志
  summary.csv                     # 两个路径的序列/篮球分组/总体指标
  metric_deltas.csv                # adapter减official
  parity.json                     # 框、ID对应和预处理差异
  status.json                     # 完成状态与parity_passed
  reference-input/                 # 指向原始序列的符号链接
  upstream-output/                 # 官方CLI直接输出，保留供追溯
```

全部分数/recall/precision在CSV中按百分数表示，计数为整数。HOTA取0.05..0.95的19个IoU阈值平均；CLEAR/Identity阈值为0.5。总体调用TrackEval合并序列，不平均各视频百分比。

相同命令可重跑：读取清单并核对数据、权重、代码与输出校验值，复用已完成序列；中断的单条序列从头重跑。不能在同一输出目录并发运行。切换数据范围、权重、代码或参数时使用新输出目录，不覆盖旧实验。

查看完成状态和总体指标：

```bash
python -m cli.benchmark_sportsmot --output-dir /outputs/sportsmot-parity-5 --stage status
```

`status=completed` 表示计算结束；`parity_passed=true` 才表示两个入口通过本轮对照。标记文件计数仅用于展示进度，不能替代输出校验。每序列官方进度和报错在 `official/val/logs/`，总运行日志由调用者重定向保存。

## 全部篮球val

初步对照通过后，使用新目录只跑官方路径，`--limit 0` 表示包中全部官方篮球val序列：

```bash
python -u -m cli.benchmark_sportsmot \
  --env-file .env \
  --dataset-root /data/sportsmot-basketball-val \
  --output-dir /outputs/sportsmot-basketball-val-all \
  --limit 0 --mode official
```

排序前5条可能来自同一原始视频，只能用于入口检查；全部篮球val才用于较完整的篮球子集评估。每序列ID属于局部身份，跨序列相加不是独立球员人数。篮球val成绩与论文全SportsMOT test成绩的划分、运动类别和统计范围不同，不预设应达到72.6 HOTA。

当前CPU自测验证原始帧/名单/GT检查、恢复时的输入变更检测、官方真实参数解析器、预测框与ID重编号对照，以及真实TrackEval的汇总与输出保护。实际GPU原始入口及适配器推理需在已配好模型的GPU机器完成，CPU自测不证明两条路径已在真实模型上等价。
