# link_player：进球时间 → 出手球员轨迹

这条分支负责事件与人物轨迹关联。进球时刻由人工标注或其他同学的模块提供；球员跨视频身份、KPR 方案比较和球员库由下游接收交接包处理。不会先运行球员库初始化、Qwen 回合切分，也不要求号码识别成功。

## 输入与模块

CSV（兼容已有 `annotations.csv`）或 JSON 对象列表：

```csv
event_id,video_id,filename,goal_time_s
goal-3,3,3.mp4,10
goal-20,20,20.mp4,3.5
```

`event_id` 可省略，将按比赛、视频、进球时间派生。同一视频可以有多个事件，需要不同 event_id。`goal_time_s` 从当前输入文件第一帧开始计算，不是比赛计时器；范围是 `[0, 视频时长)`。原始 20 号罚球使用 3.5 秒。`jersey_number/team_id/person_gt_id` 可作为评估标签保留，但**不会用于选人**。不支持 VFR 隐式换算，发现时间不规则会拒绝执行，避免标注和框错位。

| 文件 | 职责 |
|---|---|
| `pipeline/player_link/events.py` | 外部事件接口、参数校验、原视频帧窗口规划 |
| `pipeline/player_link/evidence.py` | 投篮证据筛选、候选排序、歧义处理、轨迹连续段 |
| `pipeline/player_link/export.py` | 人工修订、人物裁剪、交接包、浏览页与打包 |
| `workflows/link_player.py` | 准备媒体、MOTIP 适配/复用、动作进程、阶段校验与续跑 |
| `cli/link_player.py` | 参数、运行锁、顺序调度与状态查询 |
| `pipeline/tracking`、`pipeline/action` | 复用 main 的模型适配；上游 MOTIP/MMAction2 源码不修改 |

原来的十片段数据导出入口 `cli.export_goal_test` 保留，供同学的 KPR/图库测试继续使用；新流程不依赖 KPR 环境。没有新增初始化库、数据库、服务框架或重复的模型封装。

## 处理顺序

1. 校验视频 SHA256、逐帧时间和事件时间；为每个事件保存协议。
2. 按原分辨率、原帧率生成进球前最多 8 秒、后 0.5 秒的 FFV1 无损窗口，保存原视频帧偏移。时间范围采用原视频帧的半开区间。
3. MOTIP 正向处理完整窗口，或从已经验证的十片段导出结果中截取轨迹。复用模式保留旧局部 ID；重新跟踪时每个窗口 ID 独立，不能直接对比数值。
4. SlowFast 对实际观察到的人物框提取全部篮球动作分数，保留投篮证据；不拿缺失帧附近的框伪造精确出手位置。
5. 优先搜索进球前 4 秒，证据不足、候选分数接近或靠近搜索左边界时扩展到 8 秒。无需重新跟踪，局部 ID 不因扩展改变。
6. 排序候选，输出唯一的出手者假设或待复核。截取峰值证据附近同 ID 的连续轨迹段，按间隔和位置突变拆分；这不等于已解决所有 ID 切换。
7. 导出全部轨迹/动作证据、候选、原尺寸人物图、裁剪视频和身份识别交接清单。人工复核与自动结果分开。

默认动作参考帧率为可配置的 25 FPS，`--action-step 2`；在 60 FPS 输入上中心点约每 5 帧一次，模型帧间隔也按比例换算，避免同一配置在 60 FPS 上只覆盖很短时间。此参考帧率是本方案的工程初值，不声称已完成最优采样实验。旧 `cli.action` 未指定 `--sampling-fps` 时保持原来的逐帧采样方式。

## 候选判据与限制

仅使用标签表的 `basketball_3-point_shot`、`basketball_2-point_shot`、`basketball_free_throw`。默认分数至少 0.35，同一轨迹至少两个中心点支持，支持点间隔不超过 0.35 秒；取一段证据中最高三点的均值排序。相邻中心点不是独立的投篮事件。比较不同轨迹时默认分数差要求 0.10。以上都是**待本批视频校准的工程阈值，不是身份概率或官方最优参数**。

不会仅取时间上最后一个举手者；也不会把高分的较早投篮强行覆盖另一个人稍后的有效投篮候选。出现后者时记录 `later_supported_shot_by_another_actor`，交给复核。无投篮支持、多候选接近、画外出手、遮挡或回放均可能无法确定。

自动状态：

- `candidate`：达到支持/分差规则，提供 `shooter_track_id` 和候选轨迹，但仍需复核。
- `needs_review`：有投篮证据但人选不明确；不强行给最终 ID，保留前五候选和全轨迹视频。
- `no_candidate`：没有足够支持；交接清单仍保留事件，ID 为空。

目前**没有篮球检测、球与手分离或落网验证**。动作区间不是精确球离手时刻，也不能证明已找到“导致这次进球的最后一次出手”。`ball_verified=false`、`requires_review=true` 会随自动交接包保留。分支完成的是可运行、可检查的动作证据基线，真实正确率需用这 10 个视频核验。

## 服务器运行

先让当前十片段导出完成，再切换代码分支，避免在旧任务运行中改变模块文件。

```bash
cd /root/autodl-tmp/basket_cut
git fetch origin link_player:refs/remotes/origin/link_player
git switch --track origin/link_player
```

若服务器已经有本地分支，则 `git switch link_player` 后 `git pull --ff-only`。不要覆盖服务器未提交的改动。

先检查窗口规划（不加载 GPU 模型，不创建输出目录）：

```bash
/root/autodl-tmp/env/control/bin/python -m cli.link_player \
  --env-file /root/autodl-tmp/tool/manifests/basket-autodl.env \
  --input-dir /root/autodl-fs/data/goal-demo-input \
  --annotations /root/autodl-fs/data/goal-demo-input/annotations.csv \
  --output-dir /root/autodl-fs/link-player-demo-001 --dry-run
```

复用已完成的 `/root/autodl-fs/test` 中的 MOTIP（只读，不需要再跑 MOTIP/KPR）：

```bash
mkdir -p /root/autodl-tmp/tool/manifests
cd /root/autodl-tmp/basket_cut
nohup env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/autodl-tmp/env/control/bin/python -u -m cli.link_player \
  --env-file /root/autodl-tmp/tool/manifests/basket-autodl.env \
  --input-dir /root/autodl-fs/data/goal-demo-input \
  --annotations /root/autodl-fs/data/goal-demo-input/annotations.csv \
  --output-dir /root/autodl-fs/link-player-demo-001 \
  --match-id pl-nba-goals-001 \
  --tracking-root /root/autodl-fs/test \
  > /root/autodl-tmp/tool/manifests/link-player-demo-001.log 2>&1 < /dev/null &
```

若没有可复用结果，删除 `--tracking-root` 一行，在新的输出目录执行；MOTIP 和 SlowFast 分别在既有环境顺序运行，退出后释放显存。需要 `.env` 里的 `BASKET_MOTIP_*` 和 `BASKET_ACTION_*` 有效，不需要 Qwen 或 KPR。复用模式也验证权重、配置、视频内容和轨迹文件摘要，不能指向未完成/修改过的数据。

```bash
cd /root/autodl-tmp/basket_cut
/root/autodl-tmp/env/control/bin/python -m cli.link_player \
  --output-dir /root/autodl-fs/link-player-demo-001 --stage status
tail -n 40 /root/autodl-tmp/tool/manifests/link-player-demo-001.log
```

同一协议可重复执行恢复，已完成阶段校验后跳过。修改输入、事件时间、模型、选人参数或复用来源时使用新输出目录。`--stage prepare/track/action/link/export/pack` 可单独执行，模型阶段由控制入口调度独立解释器；不要绕过运行锁同时写一个目录。

## 结果与人工复核

```text
link-player-demo-001/
  protocol.json                  输入事件、视频摘要、参数、来源
  status.json                    完成统计，不是准确率
  review.html                    各事件候选分数、人物图、带轨迹与动作的 H.264 视频
  review.csv                     人工复核模板
  reviews.json                   已应用修订，续跑不会默默撤销
  handoff.jsonl                  交给身份识别同学的汇总入口
  events/<event_id>/
    window.mkv                   无损回看窗口，包含全体人物
    window.json                  窗口与原视频帧偏移
    tracks.jsonl                 窗口零起始帧、原尺寸 XYXY、局部 ID
    video_meta.json
    actions.jsonl                每个轨迹/中心点的全部篮球动作分数
    actions.summary.json
    link.json                    自动结果，人工修订不覆盖
    evidence.mp4                 人物框、轨迹 ID、动作分数可视化
    handoff/
      manifest.json              事件、轨迹范围、来源、候选/人工状态、player_id=null
      tracks.jsonl               选定人物轨迹，增加原视频帧和时间字段
      frames.jsonl               裁剪视频帧→窗口/原视频帧，缺测明确标记
      crops/*.png                原尺寸、无文字叠加的逐观察帧人物图
      trajectory.mp4             256×512 等比例补边裁剪预览；缺测黑帧保留时间
  checksums.sha256
```

不确定事件没有 `trajectory.mp4/crops`，但保留清单、候选和全体人物轨迹，便于人工判断。交接清单推荐最多 12 张时间分散的图，默认不施加之前导致全部拒收的质量门槛；接收方可以从逐帧图中进一步筛选清晰样本。可视化视频不用于特征提取。原视频路径是来源记录，下载包中的 `window.mkv` 和 PNG 可独立读取，不依赖服务器绝对路径。

`review.csv` 的 `decision` 填 `confirm` 并指定 `track_id`，或填 `reject`；`start_frame/end_frame` 可限制混人轨迹，**是事件窗口内的零起始、两端包含帧号**，不是原视频帧号。原视频帧 = 窗口帧 + `window_start_source_frame`。确认只表明人工已复核，不表示模型自动正确；`reset` 可撤销该人工修订，空白保持已有修订。

```bash
cd /root/autodl-tmp/basket_cut
/root/autodl-tmp/env/control/bin/python -m cli.link_player \
  --env-file /root/autodl-tmp/tool/manifests/basket-autodl.env \
  --output-dir /root/autodl-fs/link-player-demo-001 \
  --stage export --review-file /root/autodl-fs/link-player-demo-001/review.csv
/root/autodl-tmp/env/control/bin/python -m cli.link_player \
  --output-dir /root/autodl-fs/link-player-demo-001 --stage pack
```

Mac 下载最终包（端口按当前实例）：

```bash
mkdir -p "$HOME/Downloads/link-player-demo"
scp -P 16310 root@connect.bjb1.seetacloud.com:/root/autodl-fs/link-player-demo-001.tar "$HOME/Downloads/link-player-demo/"
scp -P 16310 root@connect.bjb1.seetacloud.com:/root/autodl-fs/link-player-demo-001.tar.sha256 "$HOME/Downloads/link-player-demo/"
cd "$HOME/Downloads/link-player-demo"
shasum -a 256 -c link-player-demo-001.tar.sha256
tar -xf link-player-demo-001.tar
open link-player-demo-001/review.html
```

## 验证

`python -m analysis.link_player_selfcheck`：真实视频解码、无损切窗、来源帧映射、人物 PNG/H.264、人工修订持久化、tar 校验读回、跨进程调度和续跑。人物轨迹和动作分数是合成测试输入；不宣称本地 CPU 检查完成了 GPU 模型准确率验证。另运行 `analysis.goal_test_export_selfcheck`，保证既有十片段导出仍可使用。
