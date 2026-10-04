# 短视频轨迹与身份验收

## 三个问题，不混成一个“准确率”

1. **轨迹内是否同人**：完整检查该 ID 的起止区间，尤其是交叉、遮挡、转身与镜头切换。记录 `pure`（始终同一球员）、`mixed`（包含另一人）、`non_player`（非球员/误检）、`uncertain` 或 `unreviewed`。
2. **同人是否多 ID**：只给完整检查且同人的轨迹填写片段内标签，例如“白队#15”。相同标签的轨迹组成一组；额外 ID 数为 `组内 ID 数 - 1`。纯轨迹也可能是碎片，这两个判断不互斥。时间跨度重叠的同人 ID 单独提示可能重复检测，不直接称作前后断裂。
3. **归并档案是否一致**：档案含已确认混人/非球员轨迹，或完整标注的来源轨迹属于不同球员，即为冲突；仍有未确认来源时标记未验证。全部来源完整复核且标签相同才称“一致”。

网页重点显示：**完整复核覆盖率、球员轨迹同人率、同人额外 ID 数**。混人、非球员和未知数量同时保留；原始与质量过滤后的计数分别显示。

同人率 = 完整复核的同人球员轨迹 /（完整复核的同人 + 混人球员轨迹）。非球员另计，不隐藏。不确定、未复核和仅抽样不进入这个分母；分母为零显示无结果，不能当成 100%。按轨迹计数，不是按帧、按真人或按视频计数。

**抽样图片只能发现反例，不能证明整条轨迹正确。** 复核范围默认 `sampled`，看完完整区间才选 `full_track`。自动告警、KPR 距离、过滤数量、档案数量都不是准确率。此验收也不衡量未检出的球员、框定位、动作或跨片段身份的正确性；需要这些结论时使用有对应真值的数据集。

## 运行与复核

新版网页的完整流程：MOTIP → 质量检查 → KPR → 身份归并 → 动作预测 → 人物连接 → 事件区间 → 渲染 → 时序复核证据。上游模型文件不修改，原始轨迹在 `tracking/`，过滤轨迹在 `quality/`。

在网页的“同人 / 碎片 / 混人复核”页：

1. 查看按时间排列的裁剪，点击图片打开带目标框的上下文。
2. 点击“播放区间”，在原视频上单独叠加该原始 T ID 的逐帧框；缺失观测帧不补框。完整看完再判断范围。
3. 选择同人/混人等结论。同人填写队伍＋号码；看不清就留空或选不确定，不能凭封面猜。
4. 保存复核。记录独立存放在网页实验目录的 `reviews/`，刷新、重启后仍保留。只读导入结果也能保存复核，原输出不被改动。轨迹文件变化后旧复核失效；并发编辑会提示冲突。
5. “复核汇总”合并各视频的分子、分母，不平均各视频百分比。缺失/失败视频不隐藏；点击“导出复核”冻结当前记录与原始轨迹。

已有输出可以补充证据，无需 GPU（需 OpenCV，可使用跟踪环境的 Python）：

```bash
python -m cli.prepare_track_review \
  --input runtime/data/clip.mp4 \
  --tracks runtime/outputs/baseline/clip/tracking/tracks.jsonl \
  --video-meta runtime/outputs/baseline/clip/tracking/video_meta.json \
  --quality runtime/outputs/baseline/clip/quality/quality_tracks.jsonl \
  --output-dir runtime/outputs/baseline/clip/analysis/track_review \
  --overwrite
```

若输入是旧版过滤后的派生结果，不能用它评估原始追踪；应回到保留完整原始 `tracks.jsonl` 的基线目录。新版 refinement 已保留原始轨迹。

导出的 ZIP 解压后可离线重算：

```bash
python -m cli.evaluate_track_reviews \
  --manifest runtime/reviews/manifest.json \
  --output runtime/reviews/recomputed_metrics.json
```

无复核的视频仍计入覆盖率分母。缺失必需文件时写出失败列表并以非零退出码结束；不冒充全批次评估成功。ZIP 不包含视频或模型。

## 确认归并与人物事件

人物档案一张卡片对应一个当前 P ID，列出所有原始 T ID、逐条来源封面、时间范围与复核状态；没有事件的档案也保留。合并档案不展示伪造的“整体平均 KPR 距离”。

点击“导出确认归并”下载 `merge_review.json`。只导出保留轨迹中**完整复核同人且有身份标签**的赋值；混人、非球员和不确定轨迹阻止自动归并。整个混人轨迹不能归给某一个号码。此导出不会自动改变模型结果。

应用修正时生成独立目录，并重新连接和聚合缓存动作，不能只改档案卡片：

```bash
python -m cli.refine_results \
  --input-dir runtime/outputs/baseline/clip \
  --output-dir runtime/outputs/confirmed/clip \
  --review runtime/reviews/merge_review.json

python -m cli.render_results \
  --input runtime/data/clip.mp4 \
  --tracks runtime/outputs/confirmed/clip/quality/tracks.jsonl \
  --identity-map runtime/outputs/confirmed/clip/identity/identity_map.jsonl \
  --actions runtime/outputs/confirmed/clip/action/actions_with_identity.jsonl \
  --output runtime/outputs/confirmed/clip/visualization/result.mp4
```

归并文件绑定原始轨迹指纹；用不同轨迹的文件会拒绝处理。将新目录导入网页查看。与当前身份映射不一致的旧事件不展示，并提示重新连接/聚合，避免旧事件被误认为属于新档案。

使用同一份复核标签修正结果后再检查，只能报告**修正后验收**，不能称独立模型准确率。已发现的混人轨迹不会被自动切开修好，必须保留问题记录；跨片段真人识别也尚不在此流程内。

## 第一批 30 条

- 固定输入与参数，整段处理，先不调归并距离。可在新网页实验拖入 30 条并“批量处理”，串行使用 GPU。
- 尽量覆盖不同比赛/机位、拥挤与遮挡场景，不全选同一比赛的容易片段。现有 5 条留作开发回归，另外 25 条提前选定，不能看完效果再挑。
- 先完整复核前 5 条，确认流程和分母无误，再处理剩余 25 条。若资源受限，可先做抽样排错，但报告必须说明抽样，不给全轨迹准确率。
- 原始模型结果与人工修正结果分开。报告可一句话概括：“30 条中已完整复核 X/Y 条轨迹，同人率 Z%，发现 M 条混人；已确认 P 人对应额外 Q 个 ID。”没完成复核就只报告覆盖范围。
- 这是一批开发验收数据，不足以证明泛化。调好规则后再用未参与调参、不同比赛的留出数据复测。

CPU 自检：`python -m cli.check_track_review`、`python -m cli.check_quality`、`python -m cli.check_web`（网页依赖见 `requirements-web-test.txt`；OpenCV 用于真实视频证据自检）。
