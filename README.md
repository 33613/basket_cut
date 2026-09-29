# basket_cut

按回合切分的篮球短视频处理流水线。上游项目保持原样，项目代码通过稳定的文件接口连接它们：

```text
video.mp4
  -> MOTIP tracking
  -> tracks.jsonl
  -> action backend
  -> actions.jsonl
```

## 目录约定

- `MOTIP/`：上游 MOTIP 源码，不在其中写项目代码。
- `MMAction2/`：上游 MMAction2 源码，不在其中写项目代码。
- `pipeline/`：本项目可修改的跟踪、动作和公共接口代码。
- `tests/`：不依赖 GPU 的接口测试。
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

## 同步到云服务器

本地修改并推送后，在云端执行：

```bash
cd /root/autodl-tmp/project/basket_cut
git pull --ff-only origin main
```

数据、模型和输出不要放进 Git。建议继续使用：

```text
/root/autodl-tmp/data/basket_cut
/root/autodl-tmp/models/{motip,action}
/root/autodl-tmp/outputs/basket_cut
```

## 接口测试

```bash
python -m unittest discover -s tests -v
```
