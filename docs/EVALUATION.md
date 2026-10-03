# 评估协议

只保留三个重点：追踪 IDF1、身份归并 pair-F1、人物/类别/时间都合格的事件 F1。
没有对应真值不输出准确率；计数、检测分数和 KPR 距离不能替代真值。

## 自检

```bash
python -m cli.check_quality
python -m cli.check_evaluation
python -m pip install -r requirements-eval.txt
python -m cli.check_evaluation --with-tracking
```

## 追踪

SHOT 参考轨迹可用 `cli.download_shot --dry-run` 按需选取；默认只取一个示例。
下载完成、导出预测后：

```bash
python -m cli.evaluate_tracking --reference runtime/data/reference-tracks.txt \
  --prediction runtime/outputs/demo/tracking/tracks.jsonl \
  --video-meta runtime/outputs/demo/tracking/video_meta.json \
  --output runtime/outputs/demo/analysis/tracking_metrics.json
```

与 `quality/tracks.jsonl` 使用相同视频、范围和阈值做对照。评估边界显式限定，
试跑少量帧不能代表全片效果。SHOT 的对齐参考并非完全人工逐帧标注，应说明来源。
identity 的 P ID 不直接当作真值 ID。身份归并单独评估，见
[验收说明](QUALITY_ACCEPTANCE.md)。

## MultiSports 事件

安装 Hugging Face 下载工具并遵守数据集授权，下载公开的 trainval 真值：

```bash
hf download MCG-NJU/SportsAction data/trainval/multisports_GT.pkl \
  --repo-type dataset --local-dir runtime/data/MultiSports
python -m cli.prepare_multisports_reference \
  --annotation runtime/data/MultiSports/data/trainval/multisports_GT.pkl \
  --list-videos --split validation --limit 30
```

不要用 `multisports_test.pkl` 作真值：官方测试集没有公开 gttubes。
用 `prepare_multisports_reference --help` 选择实际下载的视频 key 并转换参考事件。
模型在 MultiSports 上训练过；validation 是同域评估，不等于外部篮球数据泛化。
数据仅提供动作人物 tube，不提供跨动作或跨视频的稳定真人身份。

```bash
python -m cli.evaluate_events --reference runtime/data/reference-events.jsonl \
  --prediction runtime/outputs/demo/action/events.jsonl \
  --tracks runtime/outputs/demo/tracking/tracks.jsonl \
  --video-meta runtime/outputs/demo/tracking/video_meta.json \
  --actor-mode tube --profile multisports \
  --output runtime/outputs/demo/analysis/event_metrics.json \
  --errors-output runtime/outputs/demo/analysis/event_errors.jsonl
```

事件通过条件：类别一致、temporal IoU >=0.5，时间交集内同一原始轨迹的
平均人物框 IoU >=0.5，且至少 80% 标注帧的 IoU >=0.5；缺框按 0。
目前不允许从多个原始轨迹逐帧挑最好框，因此是保守的动作执行者定位指标，
不是归并后完整人物身份评分。官方排除 save/jump_ball，当前 profile 同样排除。
一对一匹配；重复输出为 FP，遗漏为 FN。这个产品 F1 不是官方 mAP。

空预测是真实漏检；空真值是明确负样本。缺文件、损坏或 video_id 不匹配是失败，
不能当成空预测。无 GT 且无预测时 F1 为 null，不当作满分。

## 批量

清单 JSON 的 `videos` 数组每项至少包含唯一的 `name`、`reference`、`prediction`、
`video_meta`，tube 评估还需 `tracks`。路径使用实际资源位置；参考和预测必须对齐。
参数结构见 `cli.evaluate_event_batch --help` 和 `analysis/evaluation/batch.py`。

```bash
python -m cli.evaluate_event_batch --manifest runtime/data/evaluation.json \
  --output-dir runtime/outputs/evaluation --actor-mode tube --profile multisports
```

按 TP/FP/FN 汇总，不简单平均每视频 F1；失败视频不静默跳过。
冻结开发集调好的配置，在未参与调参的比赛上测试，明确标注同域、域外和真值覆盖范围。
