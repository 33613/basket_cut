# basket_cut

按回合切分的篮球短视频处理流水线。上游项目保持原样，项目代码通过稳定的文件接口连接它们：

```text
video.mp4
  -> MOTIP tracking
  -> tracks.jsonl
       |-> KPR identity archive ---------|
       |-> MMAction2 action predictions -|
  -> actions_with_identity.jsonl
  -> MMAction2 temporal adapter
  -> events.jsonl [id, event, start, end, raw_score]
```

## 目录与职责

上游仓库保留在项目根目录，不改动其源码，也不再额外套一层
`third_party/`：

- `MOTIP/`：上游 MOTIP 源码。
- `MMAction2/`：上游 MMAction2 源码。
- `KPR/`：服务器上单独 clone 的 KPR 源码，已被 Git 忽略。

本项目自己的代码按职责分层：

- `pipeline/`：只保留 `tracking/`、`identity/`、`action/` 三个核心模块；
  负责模型调用和单模块业务逻辑，不解析命令行，不负责可视化。
- `contracts/`：跨模块稳定数据契约，包括 `tracks.jsonl` 等 JSON/JSONL
  的读写和结构校验。
- `workflows/`：应用级编排；组合核心模块、分析和产物。
- `cli/`：所有命令行入口；只做参数解析、调用 workflow/service 和
  JSON 结果输出。
- `analysis/`：辅助分析能力，目前包含轨迹评估和结果可视化；不属于
  三个模型模块。
- `tools/`：数据集与模型权重的下载、校验等开发工具。
- `adapters/`：第三方输出到项目稳定契约的转换；当前包含 MMAction2 稀疏
  动作预测点到时间事件的聚合，不属于三个模型基线。
- `web/`：研究调试台的 FastAPI 后端和无框架前端；通过独立子进程
  调用三个模型环境。
- `configs/`：项目接入层的配置文件。

依赖方向保持单向：

```text
cli -> workflows -> pipeline
                 -> adapters
                 -> analysis
web -> cli subprocesses / artifact readers
pipeline / adapters / workflows / analysis -> contracts
tools 独立负责准备外部资源
```

时间事件构建、MultiSports 参考数据转换和产品评估见
[`docs/EVENTS_EVALUATION.md`](docs/EVENTS_EVALUATION.md)。

评估计分可在无卡模式独立自检：`python -m cli.check_evaluation`。
安装 `requirements-eval.txt` 后加 `--with-tracking` 验证 IDF1/串 ID/空轨迹。
多视频验收使用 `cli.evaluate_event_batch`：按清单汇总 TP/FP/FN，失败视频不会
被静默跳过。人物位置 F1 不代表 KPR 身份检索准确率，完整操作见上面的文档。
AutoDL 换机与数据安全检查见 [`docs/AUTODL_MIGRATION.md`](docs/AUTODL_MIGRATION.md)。

`pipeline` 不反向依赖 `cli`、`workflows`、`analysis` 或 `web`。数据、
权重、运行结果保存在 `/root/autodl-tmp`，不提交到 Git。

## 1. 导出 MOTIP 轨迹

在服务器的 MOTIP 环境中，从仓库根目录运行：

```bash
conda activate /root/autodl-tmp/envs/motip

python -m cli.tracking \
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

python -m cli.action \
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

python -m cli.render_results \
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
python -m cli.download_shot --dry-run
```

默认只选择官方示例 `ATLvsNJ-10-view1-3`，并使用 `tracking-eval` 配置，只下载：

```text
ATLvsNJ-10-view1-3.mp4
ATLvsNJ-10-view1-3-track_with_gt.txt
```

确认后开始下载：

```bash
python -m cli.download_shot
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
python -m cli.download_shot \
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

python -m cli.tracking \
  --input /root/autodl-tmp/data/basket_cut/SHOT/view1/Drive_Dunk/ATLvsNJ-10-view1-3/ATLvsNJ-10-view1-3.mp4 \
  --checkpoint /root/autodl-tmp/models/motip/r50_deformable_detr_motip_sportsmot.pth \
  --output-dir /root/autodl-tmp/outputs/basket_cut/shot/ATLvsNJ-10-view1-3 \
  --overwrite
```

再把 MOTIP 输出与 SHOT 对齐后的参考轨迹比较：

```bash
python -m cli.evaluate_tracking \
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
提示，因为 KPR 官方接口允许无提示推理；适配层在这种模式下不传
`prompt_masks`，由 checkpoint 对应的模型创建通道数兼容的空提示。跑通后再接
姿态模型提供正、负关键点。

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

本项目当前只注册一个推荐权重别名 `multidataset-sports`。它联合使用
DanceTrack、SportsMOT、PoseTrack21、OccludedDuke 和 Market1501 训练，既包含
体育域数据，也是 Hugging Face 模型仓库目前提供的多数据集 checkpoint。
`none` 与 `keypoints` 两种提示模式使用同一份权重，不要重复下载。

无卡实例可以完成下载和 SHA-256 校验：

```bash
cd /root/autodl-tmp/project/basket_cut
conda activate /root/autodl-tmp/envs/kpr

python -m cli.download_kpr --list
python -m cli.download_kpr
```

下载器默认保存到 `/root/autodl-tmp/models/kpr`，并检查项目记录的官方
SHA-256。也可以继续使用等价的 Hugging Face CLI：

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

项目使用 `configs/kpr/multidataset_sports_test.yaml` 作为推理引导配置，并用
`model.load_config=True` 恢复 checkpoint 内嵌的 KPR 模型参数。引导配置使用
上游已注册的 `market1501` 名称，但不会读取 Market1501 数据；它只负责在加载
checkpoint 之前初始化 KPR 配置树。

下载后先验证官方发布的 SHA-256、checkpoint 结构、内嵌配置和所有浮点 tensor：

```bash
cd /root/autodl-tmp/project/basket_cut
conda activate /root/autodl-tmp/envs/kpr

python -m cli.inspect_kpr_checkpoint \
  --output /root/autodl-tmp/outputs/basket_cut/kpr_checkpoint_report.json
```

成功时输出中的 `valid` 和 `sha256_matches_published_value` 都应为 `true`。
本项目记录的官方 SHA-256 是：

```text
c7f3a74d86a0bb56940b2703508a50f1d3dbee4d755049272ef5caa18457db3f
```

### 6.3 对当前 SHOT 片段提取轨迹身份特征

#### Prompt mode 怎么选

- `--prompt-mode none`：默认。只使用 RGB 人物裁剪；没有可靠姿态结果时选择它。
- `--prompt-mode keypoints`：需要每个被采样的 Track/Frame 都有一个 COCO-17
  正关键点提示，可另外提供同一裁剪内其他人的负关键点。适合一个人物框中混入
  多个人或严重人物遮挡的情况。

关键点提示不是越多越好。错误骨架会明确地把 KPR 指向错误人物，所以第一版
身份归档先用 `none` 建立基线；准备好姿态模型后，再在同一批样本上对比
`none/keypoints`。提示模式不改变 KPR checkpoint，额外需要的是姿态模型输出。

`keypoints` 模式接受 JSONL，每行格式为：

```json
{"track_id": 7, "frame_idx": 24, "coordinate_space": "frame", "keypoints_xyc": [[123,80,0.95],[121,77,0.92],[126,77,0.91],[118,80,0.88],[130,80,0.87],[114,105,0.96],[135,105,0.95],[108,135,0.90],[141,135,0.89],[104,165,0.84],[145,165,0.82],[117,166,0.94],[133,166,0.93],[116,210,0.91],[135,210,0.90],[115,250,0.86],[137,250,0.85]], "negative_kps": []}
```

`keypoints_xyc` 必须有 17 个 `[x, y, confidence]`，顺序采用 COCO-17。
`coordinate_space` 可为 `frame` 或 `crop`。推荐保存原视频帧坐标，程序会根据
实际 KPR padded crop 自动转换。`negative_kps` 为可选的 `N x 17 x 3` 数组。
严格模式发现缺帧、重复 Track/Frame、错误形状或 checkpoint 通道不匹配时会
立即停止，而不会悄悄混用有提示和无提示特征。

无卡实例可以先执行数据预检。下面命令不加载 KPR，也不产生 embedding；它会
验证视频/轨迹/提示接口并保存实际抽取的人物图和检查清单：

```bash
python -m cli.identity \
  --input VIDEO.mp4 \
  --tracks tracks.jsonl \
  --output-dir identity_prepare \
  --prompt-mode none \
  --samples-per-track 8 \
  --prepare-only \
  --overwrite
```

检查 `kpr_prepare_summary.json`、`kpr_track_sampling.jsonl`、
`kpr_sampling_manifest.jsonl` 和 `identity_media/samples/`。获得 GPU 后用相同输入
去掉 `--prepare-only`，并将输出目录换成正式的 `identity_archive`。

先用每条轨迹 2 个样本进行真实视频前向测试：

```bash
cd /root/autodl-tmp/project/basket_cut
conda activate /root/autodl-tmp/envs/kpr

python -m cli.identity \
  --input /root/autodl-tmp/data/basket_cut/SHOT/view1/Drive_Dunk/ATLvsNJ-10-view1-3/ATLvsNJ-10-view1-3.mp4 \
  --tracks /root/autodl-tmp/outputs/basket_cut/shot/ATLvsNJ-10-view1-3/tracks.jsonl \
  --kpr-root /root/autodl-tmp/project/basket_cut/KPR \
  --config /root/autodl-tmp/project/basket_cut/configs/kpr/multidataset_sports_test.yaml \
  --checkpoint /root/autodl-tmp/models/kpr/kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar \
  --output-dir /root/autodl-tmp/outputs/basket_cut/shot/ATLvsNJ-10-view1-3/kpr_smoke \
  --prompt-mode none \
  --samples-per-track 2 \
  --batch-size 4 \
  --overwrite
```

成功后把输出目录改为 `identity_archive`，并把 `--samples-per-track` 改为 8；
`--archive-exemplars 4` 会为每个人保留一张封面和最多四张代表图。路径参数已有
默认值，从项目根目录运行时可以省略，但首次测试建议显式保留。

`keypoints` 模式在同一条命令中改为：

```bash
  --prompt-mode keypoints \
  --keypoints /root/autodl-tmp/outputs/basket_cut/pose/VIDEO/keypoints.jsonl
```

首次运行不要设置 `--candidate-threshold`。KPR 官方也提示跨数据域 ReID 的效果
不保证稳定，因此距离阈值要用 SHOT 参考身份标注校准，不能凭感觉指定。

输出包括：

- `kpr_track_sampling.jsonl`：所有原始轨迹的入选数、排除原因和实际过滤阈值；
- `kpr_sampling_manifest.jsonl`：KPR 前向前就保存的采样、裁剪和 prompt 检查表；
- `kpr_samples.jsonl`：实际送入 KPR 的帧、轨迹框和裁剪框；
- `kpr_samples.npz`：逐人物裁剪的 embedding 和可见度；
- `kpr_track_prototypes.npz`：每条轨迹的聚合 embedding、可见度和距离矩阵；
- `kpr_track_pairs.jsonl`：任意两条轨迹的 KPR 距离、时间重叠和可合并性；
- `kpr_summary.json`：轨迹内外观一致性统计；
- `identity_map.jsonl`：原始 MOTIP `track_id` 到规范 `person_id` 的映射；
- `identities.jsonl`：每个人的封面、代表图、时间范围和身份档案；
- `identity_archive_manifest.json`：给编排器/前端读取的产物索引；
- `identity_media/samples/`：每个 KPR 样本的紧裁剪和带框上下文图；
- `identity_media/identities/Pxxxx/`：人物封面与代表图。

`archive_no_merge` 模式中一条原始轨迹对应一个确定的 `person_id`，例如 Track 7
对应 `P0007`；其他轨迹是否通过过滤不会改变这个编号。这是身份归档，不会自动
合并轨迹，也不会虚构姓名。封面选择综合检测分数、可见部位、清晰度、人物尺寸
和到轨迹 prototype 的距离；保存的
`archive_quality_score` 是透明的展示质量启发式，不是身份置信度。

距离已按官方 Demo 归一化到约 `[0, 1]`：越接近 0，外观越像；但它不是概率。
同时存在的两个人默认不能合并，所以只有不重叠、间隔不超过 90 帧的轨迹才标记
为 `temporally_compatible`。后续阈值校准完成后，可以追加
`--candidate-threshold VALUE` 生成候选标记，仍不会自动修改 MOTIP 的原始 ID。

KPR 解决的是“两个轨迹是否可能属于同一个人”，不是直接输出球员姓名。要得到
真实身份，还需要建立带姓名或球衣号码标签的 gallery，再把轨迹原型与 gallery
特征比较。

### 6.4 把动作结果归档到人物

动作识别和 KPR 可以并行运行。两者都完成后，在任意包含本项目代码的 Python
环境中执行 CPU-only join，无需重新加载模型：

```bash
python -m cli.link_events \
  --actions /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/actions.jsonl \
  --identity-map /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/identity_archive/identity_map.jsonl \
  --output /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/actions_with_identity.jsonl \
  --overwrite
```

输出保留原 `track_id`，并增加 `raw_track_id`、`person_id`、`identity_label` 和
`identity_status`，为后续按人物剪辑和检索事件提供稳定接口。默认丢弃没有人物
映射的动作记录；需要保留时添加 `--allow-unmapped`。

独立渲染结果时可以显示规范人物 ID：

```bash
python -m cli.render_results \
  --input VIDEO.mp4 \
  --tracks tracks.jsonl \
  --actions actions_with_identity.jsonl \
  --identity-map identity_archive/identity_map.jsonl \
  --output result_with_identity.mp4 \
  --show-top-candidate \
  --overwrite
```

## 7. 启动可视化研究调试台

`web/` 是一个面向当前研究进度的检查界面，不是只展示最终视频的
演示页。它保留并展示：

- 多视频拖拽上传和单 GPU 顺序队列；
- MOTIP 轨迹、KPR 人物档案、动作结果和最终叠加视频；
- 每个阶段的执行状态、完整命令、实时日志和原始 JSONL；
- 以人物图片为入口的片段内事件索引和跨片段汇总视图。

Web 进程不导入三个模型环境，而是分别调用
`/root/autodl-tmp/envs/{motip,kpr,mmaction2}/bin/python`，所以不会把它们的
依赖强行安装到一起。首次在 AutoDL 上创建轻量 Web 环境：

```bash
cd /root/autodl-tmp/project/basket_cut

conda create -p /root/autodl-tmp/envs/basket-web python=3.10 -y
conda activate /root/autodl-tmp/envs/basket-web
pip install -r requirements-web.txt

uvicorn web.backend.app:app \
  --host 0.0.0.0 \
  --port 6006
```

在 AutoDL 实例的“自定义服务”中映射 `6006` 端口后，即可用本地浏览器
打开。也可以通过 SSH 端口转发把服务器的 `127.0.0.1:6006` 转到
本机同名端口。长时运行时建议把 `uvicorn` 放到 `screen` 会话中。

默认模型和环境路径与本 README 中的 AutoDL 目录一致。不一致时可以在
启动前设置：

```bash
export BASKET_MOTIP_PYTHON=/path/to/motip/bin/python
export BASKET_KPR_PYTHON=/path/to/kpr/bin/python
export BASKET_ACTION_PYTHON=/path/to/mmaction2/bin/python
export BASKET_MOTIP_CHECKPOINT=/path/to/motip.pth
export BASKET_KPR_CHECKPOINT=/path/to/kpr.pth.tar
export BASKET_ACTION_CHECKPOINT=/path/to/slowfast.pth
```

当前 `archive_no_merge` 生成的 `Pxxxx` 仅在单个片段内有效。页面会汇总
不同片段的人物与事件，但会明确标记为 `CLIP-LOCAL IDs`，在 KPR 距离
校准或真实球员 gallery 接入前，不会自动把两个片段中的人判为同一人。

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
