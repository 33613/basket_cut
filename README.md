# basket_cut

篮球短视频的人物追踪、身份归档与动作事件分析。三个模型通过 JSONL 文件连接，
保留原始输出与派生结果，便于检查、替换和评估。

```text
视频 → MOTIP → 原始轨迹 → 质量检查 → 稳定轨迹
                                  ├→ KPR → 人物归并 ─┐
                                  └→ MMAction2 ─────┴→ 事件区间 → 可视化
```

事件输出：`[id, event, start, end, raw_score]`。时间单位为秒，区间为 `[start, end)`。
原始分数、KPR 距离、轨迹数量都不是准确率。当前身份归并仅限单视频。

## 目录

- `pipeline/{tracking,identity,action}`：三个独立模块；不解析命令行。
- `contracts`：通用数据契约、路径默认值和几何校验。
- `workflows`：模块编排；`cli`：运行入口。
- `analysis`：评估、回归检查和可视化；`adapters`：模型输出适配。
- `tools`：数据与权重准备；`web`：检查界面；`configs`：示例配置。
- `MOTIP`、`MMAction2`：保留的上游源码；`KPR`：独立安装的上游源码。
- `runtime`：默认数据、权重、结果目录，不提交到 Git。

## 快速检查

Python 3.10+，从仓库根目录运行；以下两项只依赖标准库：

```bash
python -m cli.check_quality
python -m cli.check_evaluation
python -m cli.check_privacy
```

GPU 推理依赖分别按 [MOTIP 安装说明](MOTIP/docs/INSTALL.md)、
[MMAction2 安装说明](MMAction2/docs/en/get_started/installation.md) 和
[KPR 官方项目](https://github.com/VlSomers/keypoint_promptable_reidentification) 配置。
建议分别安装，避免模型依赖冲突。KPR 放在根目录的 `KPR/`：

```bash
git clone https://github.com/VlSomers/keypoint_promptable_reidentification.git KPR
```

MOTIP 权重见 [Model Zoo](MOTIP/docs/MODEL_ZOO.md)，动作权重见
[MultiSports 配置](MMAction2/configs/detection/slowfast/README.md)。KPR 下载及校验：

```bash
python -m pip install huggingface_hub
python -m cli.download_kpr
python -m cli.inspect_kpr_checkpoint
```

默认资源目录为 `runtime/{data,models,outputs}`；可用 `BASKET_RUNTIME_ROOT`
指定其他目录，或给 CLI 传完整路径。不要求固定用户名、机器或环境名称。
部署设置参考 [.env.example](.env.example)，实际 `.env`、数据和权重不会提交。

## 单视频

在对应模型环境分别运行；检查与归并不需要 GPU：

```bash
python -m cli.tracking --input runtime/data/clip.mp4 \
  --checkpoint runtime/models/motip/r50_deformable_detr_motip_sportsmot.pth \
  --output-dir runtime/outputs/demo/tracking

python -m cli.track_quality --tracks runtime/outputs/demo/tracking/tracks.jsonl \
  --video-meta runtime/outputs/demo/tracking/video_meta.json \
  --output-dir runtime/outputs/demo/quality

python -m cli.identity --input runtime/data/clip.mp4 \
  --tracks runtime/outputs/demo/quality/tracks.jsonl \
  --output-dir runtime/outputs/demo/identity_raw

python -m cli.resolve_identity --tracks runtime/outputs/demo/quality/tracks.jsonl \
  --video-meta runtime/outputs/demo/tracking/video_meta.json \
  --archive-dir runtime/outputs/demo/identity_raw \
  --quality-dir runtime/outputs/demo/quality \
  --output-dir runtime/outputs/demo/identity

python -m cli.action --input runtime/data/clip.mp4 \
  --tracks runtime/outputs/demo/quality/tracks.jsonl \
  --config MMAction2/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.py \
  --checkpoint runtime/models/action/slowfast_multisports.pth \
  --label-map MMAction2/tools/data/multisports/label_map.txt \
  --output runtime/outputs/demo/action/actions.jsonl

python -m cli.link_events --actions runtime/outputs/demo/action/actions.jsonl \
  --identity-map runtime/outputs/demo/identity/identity_map.jsonl \
  --output runtime/outputs/demo/action/actions_with_identity.jsonl

python -m cli.aggregate_events --input runtime/outputs/demo/action/actions_with_identity.jsonl \
  --video-meta runtime/outputs/demo/tracking/video_meta.json \
  --output runtime/outputs/demo/action/events.jsonl
```

`none` 与 `keypoints` 提示模式使用同一份 KPR 权重。默认 `none`；使用关键点时，
给 `cli.identity` 传 `--prompt-mode keypoints --keypoints PATH`，输入契约见
`pipeline/identity/archive.py`。不把检测分数当身份关联分数。

## 已有结果与验收

质量规则、人工确认格式、无推理复用以及初版验收见
[质量与身份验收](docs/QUALITY_ACCEPTANCE.md)。
同人/碎片/混人复核、归并档案与 30 条验收流程见
[短视频轨迹验收](docs/TRACK_REVIEW.md)。
带真值的数据准备与批量评估见 [评估协议](docs/EVALUATION.md)。
可以少量调试，但不能仅凭无真值样例或训练域验证集宣称泛化性。

## 检查网页

```bash
python -m pip install -r requirements-web.txt
python -m cli.import_web_results --result-dir runtime/outputs/demo --name demo
python -m uvicorn web.backend.app:app --host 127.0.0.1 --port 6006
```

打开 `http://127.0.0.1:6006`。已有项目索引持久化，重启服务不需要重跑模型。
导入结果只读；原视频必须在允许的数据根目录，结果在允许的输出根目录。
浏览器编码不兼容时安装 FFmpeg，并在导入时加 `--prepare-media`；预览可能无音频。
独立环境的 Python 路径、资源根目录均可配置，见 `.env.example`。
使用 `--env-file .env` 时先安装 `python-dotenv`；CLI 可通过环境变量或显式参数配置。
服务没有账号认证，默认仅监听回环地址，不应直接开放到公网。

网页回归检查：`pip install -r requirements-web-test.txt` 后运行 `python -m cli.check_web`。
轨迹复核自检：`python -m cli.check_track_review`。完整网页处理会生成按时间排列的
原始轨迹证据；复核记录单独持久保存，不修改导入的模型输出。
模型推理及质量规则在实际视频上的效果，需要另做验收；自检不代表模型正确率。

## 数据与隐私

只提交通用代码、合成回归样例与公开配置，不提交运行记录、私人部署信息、
账号凭据、模型权重和视频。提交前运行 `cli.check_privacy`。
该检查针对工作树；删除文件不会自动清除旧 Git 历史。
上游许可证与公开作者署名保留；数据和权重的授权以原项目为准。
