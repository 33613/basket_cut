# 球轨迹与 PL-NBA 标注

`cli.ball_track` 输出篮球/篮筐检测与连续球段，供[出手者关联](link-player.md)使用。这个模块本身不输出球员ID，也不单独确认进球。

## 环境与权重

单独使用 `requirements-ball.txt` 对应的环境。已验证组合为 Python3.10、torch2.6.0+cu124、torchvision0.21.0、RF-DETR1.10.1。篮球 RFDETRSmall 的类别为 ball、holder、rim、scorer、player、referee、ball-in-basket。

[权重来源](https://github.com/feifeifei85/OpenHoopVision/blob/ec969037b2c02b983a77b260c50cb9eb36fe9fc1/weights/rf-detr-basketball.pth)，文件129499889字节，SHA-256：

```text
69409cbefc4b92c3cd194fb7f877c6e58e52548909e05e3b5467fe0e2fe188c5
```

分发权重时保留来源和许可。加载成功不等于文件哈希或准确率已验证；本项目只复用检测权重，不运行来源项目的整套事件流程。

## 检测和缓存重放

```bash
/path/to/env/ball/bin/python -m cli.ball_track \
  --input /data/video.mp4 --checkpoint /models/rf-detr-basketball.pth \
  --output-dir /outputs/ball-run --start-s 6 --end-s 11 --goal-time-s 10
```

逐原始帧检测，保存 `ball.jsonl`（全部检测、球段、原始帧/时间映射）、`ball_vis.mp4`（H.264预览）、`summary.json`（来源/模型哈希、设置和覆盖统计）。程序检查CFR帧时钟，不隐式把VFR当作固定FPS。

`observed=false` 表示预测点，不能作为接触手或穿筐的视觉证据。原始检测一直保留，便于改变球关联策略后离线重放。

连续性约束使用画面对角线1%的位置基准，加球框尺寸和运动不确定性；尺寸比例超过2.5的候选排除，速度用近期位移的中位数估计。拒绝点和预测点不更新观测历史，新球段至少需要3个连续观测帧；未确认段保留为 tentative。

确认时可回填同一短段的早期观测，表示时序支持而非真值。持续且合理的背景误检仍可能通过；不跨长遮挡或镜头切换拼接，也不强加不适合扣篮/反弹的抛物线约束。

```bash
/path/to/env/ball/bin/python -m cli.ball_track \
  --input /data/video.mp4 --reuse-detections /outputs/ball-run \
  --output-dir /outputs/ball-replay --min-confirm-frames 3
```

重放继承窗口和检测设置，校验源视频/权重/JSONL哈希、连续源帧和时间，不加载检测模型。检测分辨率变化、缓存外窗口或低于原检测阈值的重放需要重新推理。使用新目录保留旧结果。

## PL-NBA

[公开仓库](https://github.com/holhouse/PL-NBA-Dataset) 中的 `PL-NBA-JSON.zip`，核对版本 `0b3d5013b5828004062ed27b10c9eddfca313711`。ZIP SHA-256 为 `391948eef9fd8b4cecfc58a6088edf1253b0ce4bc5531edd7147d15a8f94ee48`。

事件提供 `event_label/player/result/timestamp_video.start/end`。根据事件内部 `result=made` 筛选，文件名可能与内容不一致。时间是动作区间，没有单独精确入筐帧；README未明确冒号时间单位。60fps视频不能证明标注是“秒:帧”，应对真实画面确认后再转换，也不能把动作结束直接当作进球时刻。

```bash
python -m cli.pl_nba_annotations \
  --annotation-zip /data/PL-NBA-JSON.zip --match-id 20250501_LAL_MIN \
  --video-ids 3 6 20 43 50 68 11 30 37 63 \
  --output-dir /outputs/pl-nba-labels
python -m analysis.ball_selfcheck
```

标注读取默认保留原始时间格式；具体转换选项见 `--help`，未核实时不猜测单位。

本地只上传标注进球视频时，可用标准库脚本匹配文件名并生成rsync清单：

```bash
python3 tools/datasets/select_pl_nba.py \
  --inventory /outputs/pl-nba-labels/made_events.jsonl \
  --input-dir /path/to/local/match-videos --output-dir /path/to/new-selection
```

匹配原标注文件名或精确数字片段ID，缺失/歧义会报告并停止，不输出不完整上传清单。生成 `upload_files.txt`、原视频SHA256清单和 `timing_review.csv`，不复制/修改原视频。时间表并列显示“秒:帧（60fps候选）”与“秒:百分秒”两种解释，`goal_time_s` 留空；候选时间不是确认进球时刻。确认时间单位、逐事件进球时间和标签冲突后，再进入出手者评估。
