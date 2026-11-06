# 独立 MOTIP / KPR 数据包

`cli.export_goal_test` 对标注进球短视频导出追踪、KPR 特征和人物复核材料。它不负责自动选出手者；自动关联使用 [link_player](link-player.md)。位于 `link_player` 分支，不依赖旧实验分支。

## 输入与环境

输入目录含视频和 `annotations.csv`：

```csv
video_id,filename,goal_time_s,jersey_number
3,3.mp4,10,27
37,37.mp4,10,15
```

时间从输入文件第一帧起算。号码为人工评估标签，不是预测。控制环境需要 FFmpeg，各模型路径由 `.env` 的 `BASKET_MOTIP_PYTHON/BASKET_MOTIP_CHECKPOINT/BASKET_KPR_PYTHON/BASKET_KPR_CHECKPOINT` 指定，见 [GPU 配置](gpu.md)。

## 运行与交接

```bash
python -u -m cli.export_goal_test --env-file .env \
  --input-dir /data/goal-videos --annotations /data/goal-videos/annotations.csv \
  --output-dir /outputs/goal-test --match-id game-001
python -m cli.export_goal_test --output-dir /outputs/goal-test --stage status
```

使用新的输出目录；同一协议可恢复，参数变化需另开目录。用 `--help` 查看阶段和导出设置。

交接包中的视频、追踪和身份特征都有来源清单。复核者通过轨迹浏览页填写 `shooter_selection.csv`，后续导出目标轨迹、特征和图片。T号只属于本次追踪；同号跨视频不是同一个人，轨迹内也可能换人，应填写有效源帧区间。

```bash
python -m cli.export_goal_test --output-dir /outputs/goal-test --stage shooter \
  --selections /outputs/goal-test/shooter_selection.csv
python -m cli.export_goal_test --output-dir /outputs/goal-test --stage pack
```

下游可复用视频、`track/identity/target` 数据建立球员库；将局部轨迹与永久 `player_id` 关联后，才能按球员检索其进球片段。不要把评估号码直接写成模型身份结论。

已生成的数据包可作为 `cli.link_player --tracking-root` 的人物追踪来源；兼容性取决于原视频、模型和追踪哈希校验。改变追踪模型后重新生成依赖框/ID的特征和裁剪。
