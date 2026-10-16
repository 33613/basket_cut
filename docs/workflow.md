# 长视频、球员库与动作剪辑

## 运行目录

```text
<match-run>/
  segmentation/
    protocol.json               # 原视频哈希、模型和采样/提示词签名
    status.json                 # 粗看/复核进度，失败原因
    responses/*.json            # 每次请求、原始回答；无效回答也保留
    segmentation.json           # 自动候选片段与未决边界
    clips_manifest.json         # 导出片段 ID、哈希、原视频时间范围
    clips/CL-*.mp4              # 原分辨率 H.264 CFR 片段
    clips/CL-*.json              # 编码、量化误差与来源记录
  results/
    batch_manifest.json         # clip_id、来源及处理签名
    batch_summary.json
    library/
      players.sqlite3           # KPR 特征/可见性、号码、观察、原始事件、修正
      library_manifest.json
      indexes/<revision>-v3/    # 当前身份归属和事件索引
    CL-*/
      tracking/                 # clip_id 作为 video_id 写入轨迹
      quality/
      identity_raw/             # 每条原轨迹的 KPR 样本、原型与模型合同
      identity/                 # 档案、号码读数/缓存、登记凭据
      action/                   # 原始预测点、事件、身份关联
      visualization/
      analysis/track_review/
      pipeline.log
```

窗口临时视频在推理后删除，完整请求及原始回答保留。CLI 断点复用成功请求；模型、视频、代码或参数变化会拒绝混用旧签名。正式片段与身份入库后不改写，重新切分使用新运行目录。

## Qwen 两阶段切分

默认每窗口 20 秒、相邻重叠 6 秒、抽 16 帧，合并重复候选后在候选中心前后各 4 秒抽 32 帧复核。模型先输出时间范围，再确认局部切点；程序把局部秒数转换回原视频。参数是功能初值，尚未通过真实比赛验证时间准确率或显存上限。

规则：对手明确获得控制才认定球权转换；进攻篮板继续同一回合。触球、出手或哨声本身不能证明换球权。镜头切换可作为独立追踪片段边界，但标为 `camera_cut`，不伪装成回合结束。模糊候选保持未决，不强行切分；原视频仍由连续、不重叠的片段完整覆盖。处理片段不保证与进攻回合一一对应。

提示词在 `pipeline/segmentation/prompts.py`。`--context` 可提供五人制/三人制、球队球衣和拍摄视角；模型仍需依据可见证据判断。输出 JSON 的格式、范围与布尔字段由程序校验，失败响应不计为完成。粗看遗漏的边界不会自动被局部复核找回，仍需要实拍验证。

只分析并保留人工检查机会：

```bash
.venv/bin/python -m cli.segment_video --env-file .env \
  --input runtime/data/game-001.mp4 \
  --output-dir runtime/outputs/game-001/segmentation --dry-run

/path/to/qwen3/bin/python -u -m cli.segment_video --env-file .env \
  --input runtime/data/game-001.mp4 \
  --output-dir runtime/outputs/game-001/segmentation \
  --stage analyze --window-s 20 --overlap-s 6 \
  --coarse-frames 16 --review-frames 32
```

查看状态不加载模型，可使用控制解释器：

```bash
.venv/bin/python -m cli.segment_video \
  --output-dir runtime/outputs/game-001/segmentation --stage status
```

`segmentation.json` 保存自动结果。人工修订另存 JSON，必须完整、按顺序覆盖原视频，区间不重叠，例如 60 秒视频：

```json
{"segments": [{"start_s": 0, "end_s": 22.4}, {"start_s": 22.4, "end_s": 60}]}
```

**首次导出之前**可通过 `--review-file runtime/review.json` 采用修订。省略此项则使用自动候选。导出只需 CPU：

```bash
.venv/bin/python -m cli.segment_video --env-file .env \
  --output-dir runtime/outputs/game-001/segmentation --stage export
```

编码保留源分辨率，使用准确解码寻址而非关键帧 stream copy。VFR 输入转为 CFR，源时间线映射与一帧量化误差保存在 sidecar；不修改原视频。

程序检测实际 FFmpeg 的参数，优先使用 `-fps_mode`，旧版本回退到 `-vsync`。加载 Qwen 权重前先对输入执行一秒 CPU 解码、缩放和 H.264 编码检查；缺少工具、参数不支持或解码失败会提前停止。无效的 `OMP_NUM_THREADS` 会在模型导入前修正为最多 4 线程，合法的显式设置保持不变。

## 追踪、身份与动作

```bash
.venv/bin/python -u -m cli.process_match --env-file .env \
  --input runtime/data/game-001.mp4 --match-id game-001 \
  --output-dir runtime/outputs/game-001 --stage process --limit 1
```

单片段通过后去掉 `--limit 1`。`cli.process_match` 默认 `--stage full` 自动先分析、导出，再运行全部片段；无默认 30 片段截断。分阶段运行便于先复核切分，再投入全部模型时间。

每个片段重建 MOTIP 状态，原始数字轨迹 ID 仅片段内有效。KPR 与号码证据通过同一片段/轨迹键提交给球员库，跨片段匹配获得持久化 `player_id`。动作模型沿原轨迹推理；事件入库后依据轨迹关联当前球员，未知或归属冲突的事件也保存，不错误归到部分已知轨迹。

该方案接受切片边缘丢失追踪历史的影响，由球员库负责跨片段身份；切分与后续身份错误需分别评估。动作片段边缘沿用现有首尾补帧逻辑并标记 `temporal_padding`。

## 按球员检索与导出

先从 `results/library/people.jsonl` 或前端取得 `PL-*` ID。打印该球员的动作、轨迹、片段与原视频时间：

```bash
.venv/bin/python -m cli.player_events \
  --run-dir runtime/outputs/game-001/results --player-id PL-example
```

加 `--event basketball_2point_shot`、`--clip-id CL-example` 或 `--min-score 0.5` 筛选。分数是动作模型原始输出，不是准确率。

```bash
.venv/bin/python -m cli.player_events --env-file .env \
  --run-dir runtime/outputs/game-001/results --player-id PL-example \
  --export-dir runtime/exports/game-001 --scope event \
  --before-s 1 --after-s 1
```

`--scope clip` 导出命中事件的整个片段，同片段去重；`--media-source original` 按来源时间直接从原视频剪辑。默认从已切片视频导出。保存剪辑请求、源哈希、当前球员库修订和输出哈希；重跑不重复编码相同结果。

网页导入 `results/` 后可查看每个片段的来源和 Qwen 边界证据，并在球员动作列表直接下载。长视频切分从 CLI 启动，网页 GPU 队列仍处理单片段；边界修订当前使用上述 JSON 文件。

## 无 GPU 验证

```bash
.venv/bin/python -m cli.check_match
.venv/bin/python -m cli.check_batch
.venv/bin/python -m cli.check_person_library
.venv/bin/python -m cli.check_qwen
.venv/bin/python -m cli.check_web
```

CPU 检查使用合成模型结果并实际调用 FFmpeg。真实长视频的 Qwen 请求、联合 GPU 链路、切分召回率和身份/动作准确率均需在空闲 GPU 上另行测试。
