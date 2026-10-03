# 质量检查与身份归并

## 职责与限制

- `pipeline/tracking/quality.py`：短轨迹、逐帧重叠候选、运动突变检查。
- `pipeline/identity/resolution.py`：KPR 证据与人工确认到片段内人物档案。
- `workflows/refinement.py`：复用已有输出，过滤动作 proposals，重新连接与聚合。
- `analysis/evaluation/identity.py`：独立真值下的身份归并 pair-F1。

原文件不变。默认至少 3 次观测且实际观测时间 >=0.1s 才进入稳定轨迹。
这会漏掉非常短暂的真实人物；必须检查过滤记录，不把减少档案数当准确率提升。
`quality_observations.jsonl` 保存每个观测的保留决定；`quality_tracks.jsonl`
保存轨迹状态与原因。运动突变仅标记 `needs_review`，不自动拆分或改写身份。
高 IoU 只是疑似重复框：默认只报告；加 `--suppress-duplicates` 才进行逐帧抑制。
人物相互遮挡也会产生高重叠，启用前检查误过滤。

身份归并默认不自动合并。`--max-distance` 显式启用 KPR complete-link 归并：
合并后的任意两条轨迹都必须具有合格的距离证据、无禁止关系、时间间隔合格；
同时出现且框不一致的轨迹不能合并。疑似混人轨迹暂不自动归并。
默认少于两张 KPR 样本的轨迹也暂不自动归并，可用 `--min-samples` 调整。
默认轨迹内最大距离 0.4 触发复核，重现间隔默认 1.5s；均为可调整的启发式，
不是已校准参数。不会自动读取球衣号码或推断真实姓名。

归并输出：`identity_map.jsonl`、`identities.jsonl`、`resolution_pairs.jsonl`、
`resolution_summary.json`、封面与代表图。P ID 仅在本片段有效；`human_confirmed`、
`auto_merged`、`unresolved`、`needs_review` 区分证据来源，`confidence` 保持 null。
旧版档案没有 video_id 时给出兼容警告；使用前确认输入来自同一片段。

## 复用已有结果，不重跑模型

输入为包含 `tracking/`、`identity/`、`action/` 的单片段结果目录。
新版原始 KPR 档案位于 `identity_raw/` 时会优先读取它。
输出必须是另一个空目录，不能覆盖基线。

```bash
python -m cli.refine_results \
  --input-dir runtime/outputs/baseline/clip \
  --output-dir runtime/outputs/refined/clip
```

这条命令用标准库完成质量过滤、读取已保存 KPR 距离、重连动作、重新聚合事件。
被拒绝的观测对应的缓存动作也被过滤。动作模型没有重新推理，不能称为新动作模型效果。
未归并的 P ID 保留，不伪造“已识别真人”。失败时不发布成功标记，使用新的空目录重试。

确认候选后，再建另一份结果：

```bash
python -m cli.refine_results \
  --input-dir runtime/outputs/baseline/clip \
  --output-dir runtime/outputs/reviewed/clip \
  --review configs/review.local.json
```

实际确认文件不提交。格式为：

```json
{
  "video_id": "与 video_meta.json 相同",
  "assignments": [
    {"raw_track_ids": [1, 4], "identity_label": "team-A:#32"}
  ],
  "cannot_link": [[1, 2]]
}
```

同一 identity_label 的人工组会合并。未知轨迹、重复赋值、不同确认身份、
同时出现的不同框、cannot_link 冲突会报错，不静默忽略。若一条轨迹内部混人，
不能把它整体按封面号码归档；需要单独复核/后续拆分功能。
可另用 `--max-distance 0.2` 测试自动归并，但 0.2 只是实验起点，不是推荐准确率阈值。

新结果的叠加视频应重新渲染，不能复制旧叠加：

```bash
python -m cli.render_results --input runtime/data/clip.mp4 \
  --tracks runtime/outputs/reviewed/clip/tracking/tracks.jsonl \
  --identity-map runtime/outputs/reviewed/clip/identity/identity_map.jsonl \
  --actions runtime/outputs/reviewed/clip/action/actions_with_identity.jsonl \
  --output runtime/outputs/reviewed/clip/result.mp4
python -m cli.import_web_results --result-dir runtime/outputs/reviewed --name reviewed
```

网页完整流程自动执行质量检查和归并。归并距离留空时不自动合并，
更改参数后勾选“强制重跑已有阶段”，或使用上面的新目录复用命令。
旧导入仍按原结果显示，不会被后台自动修改。

## 初版验收

1. 先运行 `python -m cli.check_quality`、`cli.check_evaluation`；安装评估依赖后
   增加 `cli.check_evaluation --with-tracking`。这是计分与接口自检，不是模型评估。
2. 冻结少量已有片段为开发集。保留 baseline、仅 quality、人工确认、自动归并
   四种结果；记录所有阈值，不覆盖旧输出。重点检查短轨迹、重叠框和混人样例。
3. 审核过滤掉的观测，确认有无漏掉真实球员；查看运动异常和归并失败原因。
   不能只看剩余档案数或仅挑成功画面。
4. 人工确认用于建立可用结果，但不能把同一确认文件当自动身份评估真值。
   自动归并另用独立标注，冻结配置后再评价。
5. 身份真值 JSONL 每行：`{"video_id":"clip","raw_track_id":1,"identity_label":"A"}`。
   同一真人的多个轨迹共享 label。至少要有正对和负对；混人轨迹先标记问题，
   不能强行赋一个 label。`cli.evaluate_identity --reference PATH --prediction PATH --output PATH`
   输出 pair-F1、错误归并 FP、漏归并 FN 和未计分范围。没有正对且没有误合并时 F1 为 null，
   不把全部拆开算成满分；未输出的同人轨迹会形成 FN。
6. 带逐帧身份真值时比较过滤前后的 IDF1；带动作 tube 真值时比较事件 F1。
   总结误过滤/误合并/漏归并，修正实现问题后冻结参数。
7. 然后才运行约百条带真值的未参与调试片段。按比赛划分开发/测试，报告全量结果、
   失败率与困难场景，不逐条调参数。MultiSports 只能验证人物动作定位和事件，
   没有跨动作真人 ID，不能据此证明 KPR 身份泛化性。

执行大批量前的门槛：自检全部通过、数据与帧时钟对齐、失败不被丢弃、
开发集已检查误过滤及误归并、配置冻结、测试片段未参与调参。
更多同一比赛片段并不等于独立泛化验证；外部比赛/拍摄条件才支持域外结论。
