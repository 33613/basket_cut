# basket_cut

按回合切分的篮球短视频处理流水线。上游项目保持原样，项目代码通过稳定的文件接口连接它们：

```text
video.mp4
  -> MOTIP tracking
  -> tracks.jsonl
  -> KPR track identity evidence
  -> action backend
  -> actions.jsonl
```

## 目录约定

- `MOTIP/`：上游 MOTIP 源码，不在其中写项目代码。
- `MMAction2/`：上游 MMAction2 源码，不在其中写项目代码。
- `KPR/`：服务器上单独 clone 的 KPR 上游源码，已被 Git 忽略，不在其中写项目代码。
- `pipeline/`：本项目可修改的跟踪、身份、动作和公共接口代码。
- 数据、权重、运行结果保存在 `/root/autodl-tmp`，不提交到 Git。

## 1. 导出 MOTIP 轨迹

在服务器的 MOTIP 环境中，从仓库根目录运行：

```bash
conda activate /root/autodl-tmp/envs/motip

python -m pipeline.tracking.export_motip_tracks \
  --input /root/autodl-tmp/data/basket_cut/pl_nba_smoke/VIDEO.mp4 \
  --checkpoint /root/autodl-tmp/models/motip/r50_deformable_detr_motip_sportsmot.pth \
  --output-dir /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO \
  --overwrite
```

输出包括：

- `video_meta.json`：视频和运行参数；
- `tracks.jsonl`：每帧每人的结构化轨迹；
- `track_summary.json`：每条轨迹的长度、覆盖率和检测分数；
- `tracks_vis.mp4`：轨迹可视化。

可先用 `--max-frames 120` 做短测试。

## 2. 用 MOTIP 框运行 MultiSports 动作模型

在服务器的 MMAction2 环境中运行：

```bash
conda activate /root/autodl-tmp/envs/mmaction2

python -m pipeline.action.run_multisports \
  --input /root/autodl-tmp/data/basket_cut/pl_nba_smoke/VIDEO.mp4 \
  --tracks /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/tracks.jsonl \
  --config MMAction2/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.py \
  --checkpoint /root/autodl-tmp/models/action/slowfast_multisports/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb_20230320-af666368.pth \
  --label-map MMAction2/tools/data/multisports/label_map.txt \
  --output /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/actions.jsonl
```

这个入口不再运行 Faster R-CNN，人物 proposals 全部来自 MOTIP。每个人每个采样时刻都会保存篮球类别的 top-k 分数，阈值以下的候选也不会被悄悄丢弃，便于分析模型为什么判断错误。

两步必须分别在各自的 Conda 环境运行。中间的 JSONL 是接口，因此无需同时把 MOTIP 和 SlowFast 装入显存，也不要求两个上游项目使用完全一致的依赖版本。

`tracks.jsonl` 中的 `det_score` 是 MOTIP 检测置信度。当前上游运行接口没有返回 ID 分类的关联分数，所以 `association_score` 暂时为 `null`；不要把检测置信度误当成身份关联置信度。`track_summary.json` 额外提供轨迹长度、观测覆盖率和检测分数统计，可用于第一轮质量筛选。

## 3. 独立生成对照视频

可视化不会重新运行模型，可以反复调整显示方式：

```bash
conda activate /root/autodl-tmp/envs/motip

python -m pipeline.visualization.render_results \
  --input /root/autodl-tmp/data/basket_cut/pl_nba_smoke/VIDEO.mp4 \
  --tracks /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/tracks.jsonl \
  --actions /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/actions.jsonl \
  --output /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/result.mp4 \
  --show-top-candidate \
  --overwrite
```

`--show-top-candidate` 会把未达到阈值的最高分动作显示为 `top?`，只用于诊断，不能视为最终识别结果。去掉该参数时，只显示达到动作阈值的标签。

## 4. 按需下载 SHOT 数据

先在 MOTIP 环境安装仅用于下载和评估的依赖：

```bash
conda activate /root/autodl-tmp/envs/motip
cd /root/autodl-tmp/project/basket_cut
pip install -r requirements-eval.txt
```

先查看下载计划，不产生下载流量：

```bash
python -m pipeline.data.download_shot --dry-run
```

默认只选择官方示例 `ATLvsNJ-10-view1-3`，并使用 `tracking-eval` 配置，只下载：

```text
ATLvsNJ-10-view1-3.mp4
ATLvsNJ-10-view1-3-track_with_gt.txt
```

确认后开始下载：

```bash
python -m pipeline.data.download_shot
```

下载量由三层参数控制：

- `--profile video`：只下载 MP4；
- `--profile tracking-eval`：只下载 MP4 和对齐后的参考轨迹，推荐；
- `--profile full`：下载所选样本的所有帧、姿态、视线等文件，不建议首次使用；
- `--sample PATH`：精确指定一个样本，可重复传入；
- `--max-samples N`：多次传入 `--sample` 时只保留前 N 个；
- `--dry-run`：只显示精确文件清单。

下载多个已选片段时，重复传入 `--sample`；下面仅演示参数形式，运行前先从 SHOT 文件页复制真实路径：

```bash
python -m pipeline.data.download_shot \
  --sample view1/Drive_Dunk/ATLvsNJ-10-view1-3 \
  --sample VIEW/TACTIC/ANOTHER_SAMPLE \
  --max-samples 2 \
  --profile tracking-eval \
  --dry-run
```

去掉最后的 `--dry-run` 才会真正下载。数据保存在：

```text
/root/autodl-tmp/data/basket_cut/SHOT/view1/Drive_Dunk/ATLvsNJ-10-view1-3/
```

## 5. 使用 SHOT 评估 MOTIP 轨迹

先对下载的视频运行 MOTIP：

```bash
conda activate /root/autodl-tmp/envs/motip
cd /root/autodl-tmp/project/basket_cut

python -m pipeline.tracking.export_motip_tracks \
  --input /root/autodl-tmp/data/basket_cut/SHOT/view1/Drive_Dunk/ATLvsNJ-10-view1-3/ATLvsNJ-10-view1-3.mp4 \
  --checkpoint /root/autodl-tmp/models/motip/r50_deformable_detr_motip_sportsmot.pth \
  --output-dir /root/autodl-tmp/outputs/basket_cut/shot/ATLvsNJ-10-view1-3 \
  --overwrite
```

再把 MOTIP 输出与 SHOT 对齐后的参考轨迹比较：

```bash
python -m pipeline.evaluation.evaluate_tracks \
  --reference /root/autodl-tmp/data/basket_cut/SHOT/view1/Drive_Dunk/ATLvsNJ-10-view1-3/ATLvsNJ-10-view1-3-track_with_gt.txt \
  --prediction /root/autodl-tmp/outputs/basket_cut/shot/ATLvsNJ-10-view1-3/tracks.jsonl \
  --output /root/autodl-tmp/outputs/basket_cut/shot/ATLvsNJ-10-view1-3/tracking_metrics.json \
  --events-output /root/autodl-tmp/outputs/basket_cut/shot/ATLvsNJ-10-view1-3/tracking_events.csv \
  --iou-threshold 0.5
```

评估器会自动读取与 `tracks.jsonl` 同目录的 `video_meta.json`，并使用其中的
`processed_frames` 限制评估区间。例如视频实际只解码了 58 帧、参考轨迹却有
72 帧时，只评估 `[0, 57]`，不会把视频中不存在的 14 帧误算成漏检。需要故意
评估两侧帧号并集时才使用 `--ignore-video-meta`；也可以用 `--max-frames N`
显式覆盖自动值。

评估过程不是比较两边的 ID 数字是否相等。每一帧先计算所有参考框与预测框的 IoU，使用匈牙利算法寻找整体代价最小的对应关系；IoU 小于 0.5 的候选不能匹配。随后在整段时间上检查同一个参考人物是否持续对应到同一个预测 ID。

重点指标：

- `precision / recall`：误检和漏检情况；
- `matched_mean_iou`：成功匹配框的平均 IoU，越高越好；
- `IDF1`：人物身份在整段视频上的连续性，越高越好；
- `num_switches`：同一参考人物被切换到另一个预测 ID 的次数，越低越好；
- `num_fragmentations`：参考轨迹丢失后又重新出现的次数，越低越好；
- `MOTA`：综合漏检、误检和 ID Switch，越高越好；
- `MOTP`：本工具中是匹配框的平均 `1-IoU` 距离，越低越好。

`tracking_events.csv` 保存逐帧 `MATCH`、`MISS`、`FP` 和 `SWITCH` 事件，可用来定位具体出错帧。SHOT 的 `track_with_gt.txt` 是与人工关键帧身份对齐并经过检查的参考轨迹，适合当前小规模验证，但仍应称为 reference，而不是把它当作绝对无误的逐帧人工真值。

## 6. 加入 KPR 轨迹身份证据

KPR 与 MOTIP 使用独立环境。KPR 接收人物裁剪图，输出分部位 embedding 和
部位可见度；本项目再把多帧人物特征汇总成轨迹级原型。第一版不使用关键点
提示，因为 KPR 官方接口允许无提示推理；跑通后再接姿态模型提供正、负关键点。

### 6.1 在服务器安装上游 KPR

```bash
cd /root/autodl-tmp/project/basket_cut
git clone https://github.com/VlSomers/keypoint_promptable_reidentification.git KPR

conda create -p /root/autodl-tmp/envs/kpr \
  python=3.10 \
  pytorch=1.13.0 \
  torchvision=0.14.0 \
  pytorch-cuda=11.7 \
  -c pytorch -c nvidia -y

conda activate /root/autodl-tmp/envs/kpr
cd /root/autodl-tmp/project/basket_cut/KPR
pip install -r requirements.txt
python setup.py develop
pip install gdown
```

这是 KPR 官方给出的兼容环境，不要安装进 MOTIP 环境。服务器驱动可以向下
兼容该环境自带的 CUDA runtime。

### 6.2 下载 Hugging Face 多数据集权重

```bash
mkdir -p /root/autodl-tmp/models/kpr
mkdir -p /root/autodl-tmp/cache/huggingface

unset HF_ENDPOINT
source /etc/network_turbo

export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_DOWNLOAD_TIMEOUT=600

hf download trackinglaboratory/keypoint_promptable_reid \
  kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar \
  --local-dir /root/autodl-tmp/models/kpr
```

该 checkpoint 联合使用 DanceTrack、SportsMOT、PoseTrack21、OccludedDuke 和
Market 数据训练。项目使用 `configs/kpr/multidataset_sports_test.yaml` 作为
推理配置；`model.load_config=True`，模型结构和 KPR 参数从 checkpoint 内嵌
配置恢复。

下载后先验证官方发布的 SHA-256、checkpoint 结构、内嵌配置和所有浮点 tensor：

```bash
cd /root/autodl-tmp/project/basket_cut
conda activate /root/autodl-tmp/envs/kpr

python -m pipeline.identity.inspect_kpr_checkpoint \
  --output /root/autodl-tmp/outputs/basket_cut/kpr_checkpoint_report.json
```

成功时输出中的 `valid` 和 `sha256_matches_published_value` 都应为 `true`。
本项目记录的官方 SHA-256 是：

```text
c7f3a74d86a0bb56940b2703508a50f1d3dbee4d755049272ef5caa18457db3f
```

### 6.3 对当前 SHOT 片段提取轨迹身份特征

先用每条轨迹 2 个样本进行真实视频前向测试：

```bash
cd /root/autodl-tmp/project/basket_cut
conda activate /root/autodl-tmp/envs/kpr

python -m pipeline.identity.run_kpr_reid \
  --input /root/autodl-tmp/data/basket_cut/SHOT/view1/Drive_Dunk/ATLvsNJ-10-view1-3/ATLvsNJ-10-view1-3.mp4 \
  --tracks /root/autodl-tmp/outputs/basket_cut/shot/ATLvsNJ-10-view1-3/tracks.jsonl \
  --kpr-root /root/autodl-tmp/project/basket_cut/KPR \
  --config /root/autodl-tmp/project/basket_cut/configs/kpr/multidataset_sports_test.yaml \
  --checkpoint /root/autodl-tmp/models/kpr/kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar \
  --output-dir /root/autodl-tmp/outputs/basket_cut/shot/ATLvsNJ-10-view1-3/kpr_smoke \
  --samples-per-track 2 \
  --batch-size 4 \
  --overwrite
```

成功后把输出目录改为 `kpr`，并把 `--samples-per-track` 和 `--batch-size`
分别改为 8，运行正式测试。三个路径参数已有上述默认值，从项目根目录运行时
可以省略，但首次测试建议显式保留，便于核对日志。

首次运行不要设置 `--candidate-threshold`。KPR 官方也提示跨数据域 ReID 的效果
不保证稳定，因此距离阈值要用 SHOT 参考身份标注校准，不能凭感觉指定。

输出包括：

- `kpr_samples.jsonl`：实际送入 KPR 的帧、轨迹框和裁剪框；
- `kpr_samples.npz`：逐人物裁剪的 embedding 和可见度；
- `kpr_track_prototypes.npz`：每条轨迹的聚合 embedding、可见度和距离矩阵；
- `kpr_track_pairs.jsonl`：任意两条轨迹的 KPR 距离、时间重叠和可合并性；
- `kpr_summary.json`：轨迹内外观一致性统计。

距离已按官方 Demo 归一化到约 `[0, 1]`：越接近 0，外观越像；但它不是概率。
同时存在的两个人默认不能合并，所以只有不重叠、间隔不超过 90 帧的轨迹才标记
为 `temporally_compatible`。后续阈值校准完成后，可以追加
`--candidate-threshold VALUE` 生成候选标记，仍不会自动修改 MOTIP 的原始 ID。

KPR 解决的是“两个轨迹是否可能属于同一个人”，不是直接输出球员姓名。要得到
真实身份，还需要建立带姓名或球衣号码标签的 gallery，再把轨迹原型与 gallery
特征比较。

## 同步到云服务器

本地修改并推送后，在云端执行：

```bash
cd /root/autodl-tmp/project/basket_cut
git pull --ff-only origin main
```

数据、模型和输出不要放进 Git。建议继续使用：

```text
/root/autodl-tmp/data/basket_cut
/root/autodl-tmp/models/{motip,kpr,action}
/root/autodl-tmp/outputs/basket_cut
```
