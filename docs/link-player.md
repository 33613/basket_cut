# 进球时刻到出手球员

本入口接收外部进球时刻，输出局部出手轨迹与身份识别交接包。跨视频身份归并、KPR 方案和球员库由下游处理。

## 输入与时钟

```csv
event_id,video_id,filename,goal_time_s
goal-3,3,3.mp4,10
goal-20,20,20.mp4,3.5
```

也支持 JSON 对象列表。`event_id` 可省略，由比赛、视频和时间派生；同一视频可有多个事件。`goal_time_s` 是当前文件第一帧起的秒数，范围 `[0, 视频时长)`，不是比赛计时器。`jersey_number/team_id/person_gt_id` 只保留为评估答案，不参与选人。不规则帧时钟会被拒绝，避免框与标注错位。

## 实际证据流程

1. 球证据模式默认截取 `[goal-4, goal+1)`，越界裁到原片。保存窗口帧到源帧的映射。
2. MOTIP 输出人物框和局部轨迹；可新跑窗口，也可校验复用已有整段追踪。复用整段结果不等于5秒冷启动追踪。
3. SlowFast 提供轨迹投篮动作分数。复用动作缓存时保留原推理上下文。
4. RF-DETR 检测球、篮筐、holder、scorer；球关联使用位置、尺寸、稳健速度和多帧确认。预测点不作为视觉接触证据。
5. 进球前3秒内保留多个有观测支持的到筐球段，检查球与人物上半身的接触、相对运动稳定性和后续分离。扣篮允许手随球到筐；排除同球路上的更早控球者。
6. 几何球分数80%、动作分数20%是启发式，非概率。持续、唯一匹配的 holder 可消解几何歧义；scorer 单独不能指定球员。强证据冲突时保留待复核。
7. 从目标局部轨迹裁原尺寸 PNG，按时间分箱，以清晰度、尺寸和检测分数选图；输出256×512预览、轨迹视频和来源清单。`player_id` 留给下游赋值。

人物框接触不是手腕关键点证据，T号也不保证整条轨迹为同一人。需要复核目标局部段的身份纯度和图像可用性。

## 运行

```bash
python -u -m cli.link_player --env-file .env \
  --input-dir /data/goal-videos --annotations /data/goal-videos/annotations.csv \
  --output-dir /outputs/goal-link-run --match-id game-001 \
  --ball-python /path/to/env/ball/bin/python \
  --ball-checkpoint /models/ball/rf-detr-basketball.pth
python -m cli.link_player --output-dir /outputs/goal-link-run --stage status
```

阶段为 `prepare/track/action/ball/link/export/pack`，控制入口使用各模型独立解释器。省略球参数时兼容原动作方案，其回看与歧义策略不同。

- `--tracking-root`：复用独立 MOTIP/KPR 导出中的追踪结果。
- `--reuse-run`：复用旧运行中校验通过的窗口、人物和动作数据，生成新的归因结果。
- `--ball-cache-run`：按事件 ID 复用旧运行的球证据；支持同一视频多次进球。
- `--ball-cache VIDEO_ID=DIRECTORY`：指定单视频球缓存。

修改输入、模型、选人设置或复用来源时用新目录。同一协议可断点恢复；已完成阶段经哈希校验后跳过。不要绕过运行锁同时写同一目录。

### PL-NBA 区间输入（无需逐条填写精确进球时刻）

```bash
python -m cli.pl_nba_link_input --review-csv /data/selection/timing_review.csv --output /data/selection/link-intervals.json
```

使用此 JSON 作为 `--annotations`，必须启用 `--ball-checkpoint`。默认取原动作区间的两种未确认时间解释（秒:百分秒、秒:60fps帧）的并集，前留0.25秒、末尾增加1.5秒飞行余量。原始字符串和不确定性保留，不将动作结束点伪装为精确进球时刻。超出视频的范围在准备阶段裁剪；完全无交集报错。

在整个区间内寻找经观测支持的近筐球段，保留同一球ID的不同近筐访问；每个近筐候选只向前搜索 `lookback_s` 秒的人球证据。没有合格球路或存在人物歧义时仍输出待复核/无候选，不回退为动作强制选人。窗口覆盖区间及其前后上下文，可能长于5秒；这属于区间辅助归属测试，不是已知精确时间的5秒基线。

兼容字段 `goal_time_s` 在区间模式下只是区间上界参考，`goal_time_semantics` 明确标识；`estimated_rim_time_s` 只是候选球路首次受支持的近筐时间，不证明穿筐或实际进球。复核页面、交接清单保留区间信息。所有事件仍须纳入成功率分母；未确认时制、飞行余量不足或多次攻筐可能导致失败。不要复用精确时间旧运行的动作/人物缓存。

持球角色匹配要求检测分数≥0.35、人物 IoU≥0.5，且领先次优 IoU≥0.1；至少3个不同源帧，跨度≥0.05秒，相邻间隙≤0.1秒。仅使用搜索区间内、候选出手之前的支持。多个角色输出来自同一检测器，不作为独立概率重复投票。

### 人物轨迹质量筛选

`--min-track-observed-s 0.3` 在原 MOTIP 输出后复用质量检查，过滤实际观测时长不足0.3秒的轨迹；也可设为0.5。时长为窗口内观测帧数/FPS，不用首尾跨度充当持续观测。默认0保持旧基线行为。

原始 `tracks.jsonl` 不改写，派生结果与审计保存在 `quality/`。动作推理和自动选人使用派生轨迹，复核视频保留原始人物框；突跳与重叠做审计标记，不强制消除。`link.json` 和身份交接清单记录阈值、保留数量和被排除ID。短轨迹过滤不保证局部身份纯净，也不保证裁剪图片可用。

## 结果与复核

`protocol.json` 保存输入、模型、选项和来源；`events/<event_id>/link.json` 保存候选、原因、球路和角色证据；`review.html` 展示人物与上下文视频，`review.csv` 用于人工决策。自动 `candidate/needs_review/no_candidate` 与运行完成状态分开记录。人工复核不覆盖自动证据。

```bash
python -m cli.export_link_review --output-dir /outputs/goal-link-run
```

生成同级 `goal-link-run-review.zip` 与 SHA-256，包含离线页面、人物/球叠加视频、目标轨迹、选出的原尺寸图片和空白复核表；不包括无损窗口和全部模型缓存。它用于复核，完整身份交接应使用主运行导出的清单与数据包。

解压打开 `review.html`，检查出手者、局部轨迹纯度、身份图片和动作完整性。画面高亮表示局部T号，不保证全窗口同人。修改离线复核表不会自动更改服务器结果。

## 实现与回归

### 按球员输出展示视频

`cli.goal_demo` 接在核对后的结果之后，复用 main 的 KPR 适配、`PlayerRegistry` SQLite 库、球员查询和原视频裁剪，不重跑追踪/动作/球检测。默认 `reviewed` 模式按人工 `player_key` 建库，无需 GPU；选 `kpr` 才提取 KPR 特征，未填写身份的目标走现有匹配，人工分组优先。不清晰且无身份的事件独立保留，不假造特征或强行归组。

在现有复核 CSV 中填写 `shooter_correct=1`，必要时增加以下列（UTF-8 CSV，Excel 可导出）：

| 列 | 含义 |
|---|---|
| `reviewed_track_id` | 人工修正的 T 号；支持原表的 `轨迹号码` 列 |
| `player_key` | 本场同人统一键，如 `27g`、`15y`；支持原表以 `球员号码` 开头的列。只填号码不足以区分两队，需队伍标记或独立键 |
| `player_name` | 可选展示名，不影响分组 |
| `source_segments` | 混人/换ID时填 `T2:源帧起点-终点;T4:源帧起点-终点`，边界均包含，必须在已追踪窗口内 |
| `review_start_frame/review_end_frame` | 单T号人工修正的源帧边界，替代自动边界 |
| `clip_start_s/clip_end_s` | 可选的原视频剪辑秒数，默认保留完整候选处理窗口 |
| `include` | 可选，0 排除、1 纳入；空白时只纳入确认正确或明确修正的目标 |

`track_pure=0` 必须给出修正的源帧段/边界，否则跳过并记录原因。该帧范围不同于旧 `review.csv` 的事件窗口局部帧号。`predicted_track_id` 和原始 `link.json` 不改写。样本图从可信局部段重新抽取；不把整条可能混人的轨迹送入 KPR。

```bash
python -u -m cli.goal_demo --env-file /path/to/basket.env \
  --link-run /outputs/goal-run --review-csv /data/review.csv \
  --output-dir /outputs/player-demo-001 --identity-mode reviewed
# 可选 KPR：改为 --identity-mode kpr --kpr-root /path/to/KPR
python -m cli.goal_demo --output-dir /outputs/player-demo-001 --stage render --player-id PL-xxx
python -m cli.player_events --run-dir /outputs/player-demo-001 --player-id PL-xxx
```

阶段可独立运行 `prepare/features/library/render`，便于以后替换身份模块。修改复核表或身份模式时用新 demo 目录；同输入可续跑。`plan.json` 和 `review_rows.json` 记录采用的目标和跳过原因，`library/` 是现有格式的球员库，`videos/PL-*.mp4` 是每个人的合并集锦，`videos/all-players.mp4` 按球员分组拼成一个完整展示视频，`index.html` 可本地播放、下载。KPR 阶段只在独立 KPR 解释器中运行；控制环境需要 numpy/OpenCV/FFmpeg。

视频只显示指定球员的框和身份标签、真实观测且时间确认的球点及短轨迹；不画其他人物、篮筐、分数或调试文字，不连接不同球段/长缺口。画框仅限可信轨迹段，不延伸到未验证的同号轨迹。原片音轨保留，无音轨时补静音，统一 H.264/AAC、默认1280×720/30fps后按视频编号排序拼接；可用 `--width 1920 --height 1080` 调整输出。原尺寸身份图片不随视频缩放。默认窗口不一定覆盖完整攻防回合，可在 CSV 调整剪辑范围，但窗口以外没有追踪时不补造人物框/球点。

KPR 模式使用 `quality_visible_multiframe_v2`：抽图保留所有人物框用于遮挡评分，但仅抽人工确认的目标/帧；至少24×64、检测分数0.5、其他人体框覆盖超过45%的图片不送入特征库。筛选失败的中间帧只作展示兜底，不能成为身份参考。保留最多4张实际图片的部位特征，不平均正背面；移除没有相近同伴的孤立外观样本，保护至少两张相隔0.15秒的支持图，并保存 `features.json/template_audits`。这些尺度、距离及遮挡阈值是工程默认值，不是经过NBA标注校准的置信度。

比较仍使用KPR共同可见部位距离。查询和图库各取最近邻距离中的第二小值，两边取最大，需同时通过距离0.2和领先次优球员0.05的默认条件；只有一张支持图时不自动归并。人工球队/号码冲突先排除，号码0和00保留区别。自动匹配事件不立即加入参考图库，避免错误链式扩散；人工分组及有多图支持的初始新候选提供参考。不清晰/不确定事件仍保留在各自集锦中，人工归组不受影响。旧平均向量不能与新版混用，需新 demo 目录重新抽图/提特征，原追踪、动作和球缓存可直接使用。

参考：[KPR，ECCV 2024](https://github.com/VlSomers/keypoint_promptable_reidentification) 的目标消歧和可见部位比较；[BPBreID，WACV 2023](https://github.com/VlSomers/bpbreid) 的部位表示；[PRTReID，MMSports 2023](https://github.com/VlSomers/prtreid) 的身份/球队/角色联合建模。这里复用已有KPR权重并改进工程匹配，不是复现这些论文的新训练结果，也未换用足球PRTReID权重。当前仍是无关键点提示模式；多人拥挤时通过筛图减轻问题，不能宣称获得了带关键点提示的完整消歧能力。

```bash
python -m unittest analysis.goal_demo_selfcheck analysis.person_library_selfcheck
```

`pipeline/player_link/events.py` 负责事件与窗口，`ball_evidence.py` 负责球与人物关系，`roles.py` 负责角色匹配，`export.py` 负责裁剪、交接和复核，`workflows/link_player.py` 编排各阶段。

```bash
python -m unittest analysis.ball_selfcheck analysis.link_player_selfcheck analysis.link_ball_selfcheck
```

现有十段用于规则开发；历史候选T号与人工表一致不等于独立测试准确率。基线代码由 Git 标签 `goal-link-v3-baseline-20261106` 保留。更换追踪器后T号会变化，应按原视频帧和人物核对；动作、裁剪与身份特征依赖旧框/旧ID，不能直接套给新轨迹。球检测与进球时刻可在校验后复用。

扩展到PL-NBA时，先用 [标注筛选工具](ball-tracking.md#pl-nba) 固定事件清单并核对时间。球员姓名只作评估标签，错误标签应单独裁定。分别报告开发样例与新增样例；自动分配覆盖率、所分配事件中的正确率、以及全体有效事件中“出手者正确且轨迹同人且图片可用”的比例分开统计，未选中或待复核不能从分母静默删除。
