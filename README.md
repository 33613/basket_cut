# CourtVision · 篮球视频与同场球员库

一场比赛的长视频经过 Qwen3-VL 分窗口粗看、边界复核后切成片段，再逐片段运行 MOTIP。轨迹的 KPR 外观特征与 Qwen 球衣号码证据一起更新同场球员库；SlowFast 动作按片段和轨迹关联到球员，支持检索、复核与视频剪辑。

```mermaid
flowchart TD
    S[原视频 source_video_id] --> C[Qwen 粗看窗口 → 边界复核 → 片段 clip_id]
    C --> T[MOTIP：每片段独立追踪]
    T --> Q[质量筛选，保留原始轨迹]
    Q --> K[KPR 人物特征]
    Q --> N[Qwen 多帧号码证据]
    K --> P[(同场比赛球员库 player_id)]
    N --> P
    Q --> A[SlowFast 轨迹动作 → 事件区间]
    A --> J[clip_id + track_id → player_id]
    P --> J
    J --> W[按球员、动作、片段检索 → 视频剪辑]
```

## ID 与时间

| 字段 | 含义 |
| --- | --- |
| `match_id` | 一场比赛；一份球员库只属于这一场 |
| `source_video_id` | 原视频内容 SHA-256 派生的 `SRC-*` |
| `clip_id` | 原视频与切分起止时间派生的 `CL-*`；切点改变则 ID 改变 |
| `track_id` / `raw_track_ids` | MOTIP 在片段内的数字 ID；不同片段可以重复 |
| `track_uid` / `track_uids` | `clip_id:T0000`，用于唯一定位轨迹 |
| `local_person_id` | 片段内档案 ID `P*`，保留旧接口与审计证据 |
| `player_id` | 同场持久化身份 `PL-*`，兼容字段为 `global_person_id` |
| `event_id` | 动作事件 ID，关联片段、原轨迹和当前球员 |

事件 `start/end` 是片段内秒数；`source_start_s/source_end_s` 是原视频秒数。来源时间从原视频首个视频呈现时间戳开始计算，不是比赛计时器。导出片段明确转为 CFR，供现有 `frame_idx/fps` 后端使用，切点量化误差随元数据保存。

## 运行

在仓库根目录执行。控制环境与各 GPU 环境独立，私人解释器和权重路径参考 [.env.example](.env.example)；部署见 [GPU 环境](docs/gpu.md)。控制机需要 `ffmpeg`、`ffprobe`。

```bash
bash tools/setup_control.sh
.venv/bin/python -m cli.check_runtime --env-file .env
.venv/bin/python -u -m cli.process_match \
  --env-file .env --match-id game-001 \
  --input runtime/data/game-001.mp4 \
  --output-dir runtime/outputs/game-001
```

`--dry-run` 查看计划，不加载模型。`--stage segment` 只切分；`--stage process --limit 1` 先处理一个已导出片段，移除 `--limit` 后继续全部。切分参数、人工修订、断点恢复和检索剪辑见 [完整流程](docs/workflow.md)。

已有短视频仍可使用 `cli.process_batch --input-dir ... --output-dir ... --match-id ...`。同一批次的历史成员不会因为本次只选部分片段而删除。单张 GPU 串行运行，Qwen 切分子进程退出后才启动 MOTIP；每片段执行：

`tracking → quality → action → aggregate → identity → resolution → jersey → players → link → render → review`

## 球员库与网页

KPR 与 Qwen 使用各自的质量采样：人物图提取外观，躯干图识别号码。号码可以拒识，`0` 与 `00` 分开；至少两个时间间隔满足要求的支持帧才形成候选，读数冲突保留。同号不能单独证明同人。质量不足、外观不一致、候选接近时待定观察不进入图库。

KPR 权重固定，增量更新的是数据库和外观图库。球员 ID 创建后保持稳定，人工合并保留别名；不跨比赛合并。同一片段同时出现的不同轨迹不能合并。默认匹配距离 `0.2`、新身份距离 `0.5`、候选距离差 `0.05` 是待验证的工程初值。轨迹内部混人仍需要人工检查。

```bash
.venv/bin/python -m cli.serve_web --env-file .env --host 0.0.0.0 --port 6006
```

网页“导入结果”选择长视频运行目录下的 **`results/`**。流程页显示原视频、片段时间和 Qwen 边界证据；球员库可核查外观/号码来源、调整身份、检索动作，并下载动作前后各一秒或整个对应片段。号码、模型分数和处理成功均不代表准确率。服务通过平台受控端口访问。

SQLite 是身份事实来源，JSON 是按修订号发布的可浏览索引。人工修正后检索和导出读取当前归属，无须重新运行模型。已注册的特征、号码或动作证据不能被静默覆盖；改变模型或切点后使用新的输出目录，保留旧数据库和证据。

## 文件管理与验证

| 目录 | 职责 |
| --- | --- |
| `pipeline/segmentation`、`pipeline/media.py` | Qwen 切分、时间映射、CPU 视频导出 |
| `pipeline/tracking` | MOTIP、轨迹质量 |
| `pipeline/identity` | KPR、号码证据、持久化球员库 |
| `pipeline/action`、`adapters/mmaction2` | 原有 MultiSports SlowFast 动作方案 |
| `contracts`、`workflows`、`cli` | 数据契约、执行编排、命令行入口 |
| `web` | 检查、身份修正、检索与下载 |
| `analysis` | CPU 自检、数据集评估 |
| `MOTIP`、`MMAction2` | 保留的上游实现 |

权重、环境、私人配置、原视频和输出均不提交 Git。部署目录可继续分为 `model/`、`tool/`、`env/`、`basket_cut/`、`data/`、`output/`。

`cli.check_match` 实际执行合成视频的窗口切分、边界合并、来源映射、身份检索和 FFmpeg 导出；模型回答使用明确标注的合成输入。其他 CPU 检查由 `tools/setup_control.sh` 执行。真实 Qwen 切分及 MOTIP/KPR/SlowFast 联合推理需要在有空闲 GPU 时另行验证。

保留独立的 [SportsMOT 原始帧评估](docs/sportsmot.md) 和 [TeamTrack 评估](docs/teamtrack.md)，不把业务切分混入基准评估。模型来源：[MOTIP](https://github.com/MCG-NJU/MOTIP)、[KPR](https://github.com/VlSomers/keypoint_promptable_reidentification)、[MMAction2](https://github.com/open-mmlab/mmaction2)、[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)。
