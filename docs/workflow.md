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
    bootstrap_tracklets/        # 六窗口初始化，与回合片段分开计数
      protocol.json             # 通用时间分层、画质阈值、原视频/检测模型签名
      selection.json            # 候选探查和最终 6 个起点
      windows/WIN-*.mkv         # 原分辨率 FFV1 无损 CFR 窗口
      WIN-*-tracks.jsonl        # 每窗口独立的连续追踪
      media/*.png               # 原尺寸人物裁剪，无缩放
      manifest.json             # 窗口/裁剪哈希、源时间与轨迹关联
      assessments.json          # Qwen 角色、清晰无遮挡、队服、号码、朝向
      team_map.json             # 显式队伍映射或谨慎推断的匿名 KIT 队伍
      features.npz              # 仅合格清晰人物图的 KPR 特征
      feature_contract.json
      registration.json         # 入库数、淘汰数和修订号
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

`cli.process_match` 默认每窗口 20 秒、重叠 6 秒、16 帧、8192 视频视觉预算。先记录四个时刻的有效进攻状态，死球持球方不会直接更新进攻方；同一窗口和跨窗口的状态变化产生待复核候选。复核查看完整候选前后各 2 秒，使用 16 帧，超过 40 秒的宽候选保留待人工检查。8192 是总视频预算，增加帧数会降低每帧可用空间分辨率。局部时间转换回原视频；这些设置尚未得到完整比赛准确率验证。

回合起点约定为**新进攻方开始获准发球的准备，或取得活球控制，以先出现者为准**。死球捡球、交球给裁判、同队连续罚球或进攻篮板不产生新回合。状态提示词和独立复核位于 `pipeline/segmentation/possession.py`。不会强行把候选中点用作切点；未缩小或无法判断的候选保留未决。`cli.segment_video` 默认仍保留旧的 `boundaries` 模式用于对照；使用 `--coarse-method states --video-tokens 8192 --review-frames 16` 选择新流程。两种模式和参数都记入签名，不混用缓存。

触球、出手或哨声本身不能证明换球权。旧 `boundaries` 模式还可提出 `camera_cut`；新状态模式当前聚焦回合，不单独检测镜头切换或回放，正式转播应用仍需检查这些情况。模糊候选保持未决；原视频仍由连续、不重叠的片段完整覆盖。处理片段不保证与进攻回合一一对应。

提示词在 `pipeline/segmentation/prompts.py`。`--context` 可提供五人制/三人制、球队球衣和拍摄视角；模型仍需依据可见证据判断。输出 JSON 的格式、范围与布尔字段由程序校验，失败响应不计为完成。粗看遗漏的边界不会自动被局部复核找回，仍需要实拍验证。

无候选时要求输出 `{"boundaries": []}`，解析器也接受字段完全合格的直接数组（包括 `[]`）。`null`、缺字段和越界时间不会被当作空结果。格式失败时在同一视频窗口上最多再请求两次，每次提示词、原始回答及错误保存在 `responses/<window>.json` 的 `attempts` 中；重试耗尽会明确停止。格式处理或提示词更新后使用新运行目录，不覆盖旧请求证据。

如果人工确认存在换球权，但粗看全部返回空列表，先诊断一小段，避免重复跑整场。以下命令选择与原视频 120–180 秒相交的原始粗看窗口，对每个窗口分别请求画面描述与原来的边界提示词；模型只加载一次：

```bash
/path/to/qwen3/bin/python -u -m cli.diagnose_segmentation --env-file .env \
  --segmentation-dir runtime/outputs/game-001/segmentation \
  --output-dir runtime/diagnostics/game-001-first \
  --start-s 120 --end-s 180
```

默认保留原始采样帧数和边界提示词，对比原始回答；最多诊断 8 个窗口。诊断输出独立保存 `diagnostic.json`、每个窗口的 `window.mp4`、两种提示词及回答。`model-input.jpg` 从实际送入模型的 Qwen3 像素张量还原，`input-summary.json` 记录实际帧数、处理后尺寸、token 数和帧差，`input-text.txt` 保留时间戳文本并压缩重复视觉占位符。序号是采样顺序，不能当作秒数。这些文件用于核查视觉输入与任务理解，不是人工标注或准确率评估。

若实际输入画面及描述正常、边界回答仍为空，再检查边界定义及提示词。`--frames 32` 会保留原有总像素预算，因此增加帧数也会降低单帧分辨率，不能称为纯采样密度对照。采样实验需另行控制空间分辨率和总 token 开销。诊断不更新正式片段、缓存或球员库；CPU 自检只验证张量还原与诊断链路，真实视频效果需在 GPU 上检查。

若实际输入是罚球而描述成跳球，可先用同一窗口做四次简短推理，比较静态图片/视频及 SDPA/eager 注意力实现：

```bash
/path/to/qwen3/bin/python -u -m cli.probe_qwen_vision --env-file .env \
  --diagnostic-dir runtime/diagnostics/game-001-first \
  --window-id coarse-000010 --frame-index 8 \
  --output-dir runtime/diagnostics/game-001-vision-probe
```

输入窗口须仍保留在服务器诊断目录中。图片从实际处理后的视频第 8 号采样帧还原（不是第 8 秒），并关闭二次空间缩放，核对图片/视频的空间网格相同；提示词不提供期望事件标签。每种模态的输入张量在两种注意力实现之间复用，输出保存 `attention-comparison.json`，图像保存 `sampled-frame.png`。命令核实实际注意力配置，并在结束/异常时恢复原 SDPA；不保存修改后的模型配置或权重。该实验可以提示视频路径或注意力实现是否影响结果，单个窗口不能证明根因或模型整体能力。

如果 eager 未改善，下一次可在新输出目录添加 `--attention-modes sdpa --video-tokens 8192`，保持同一 16 帧及提示词，仅提高空间预算，执行两次图片/视频推理。此值是整个视频的视觉 token 上限，实际空间尺寸和 token 数取决于源分辨率及网格取整；不会凭空增加原视频细节。`probe-protocol.json` 分别记录原始/实际像素预算，模型来源签名仍指向原始已校验模型。更高预算会增加显存和耗时；真实 GPU 运行后才能确认显存是否足够。该选项仅用于诊断，不改变生产处理器预算。

若提高空间预算后能识别罚球，应继续检查换球权候选，而不是据此判定切分通过。在 `cli.diagnose_segmentation` 的小范围命令中添加 `--video-tokens 8192 --queries boundary`，保持原来的窗口、16 帧采样及保存的原始边界提示词，仅请求边界结果；120–180 秒通常对应 5 次推理。诊断记录基础模型来源和实际运行像素预算，成功或失败后恢复处理器预算，原有正式切分目录不变。若仍返回空列表，再单独改进候选任务与边界定义。

若高分辨率的直接边界请求仍全为空，可在另一个输出目录改为 `--video-tokens 8192 --queries state`，保持采样、窗口及空间预算不变，改测结构化状态观察。每个窗口的 0%、25%、50%、75% 时刻分别返回比赛状态（活球、罚球、死球、不确定）、进攻方明暗球衣和可见证据，无法判断时允许 `unknown`；空列表、缺失或重复时间点会被标为格式错误。无需识别号码、比分或球队名称。`analysis/possession_states.py` 保留此前的实验对照定义；生产状态方案另在 `pipeline/segmentation/possession.py`，增加发球准备、死球约束及约定切点。

这个实验入口从已知进攻方变化，或进攻方不明的罚球→活球变化提出宽时间范围；相邻状态为 `unknown` 不会被补写成某支球队。所有候选均是 `uncertain`、`needs_review=true`，保存局部/原视频范围及对应观察点，终端打印 `[state candidates]`。这些实验候选不自动导出片段，不写球员库，也不等于人工确认的回合转换，用于和生产方案对照。诊断新生产运行的 `boundary` 查询会使用其保存的状态提示词、对应解析器和原运行预算，不会把状态 JSON 错当成边界 JSON。

复核上述状态候选可运行（默认每个候选前后各加 2 秒，16 帧、8192 视觉 token 上限）：

```bash
/path/to/qwen3/bin/python -u -m cli.review_segmentation_candidates --env-file .env \
  --diagnostic-dir runtime/diagnostics/game-001-states \
  --output-dir runtime/diagnostics/game-001-state-reviews
```

命令先打印候选及源视频范围，模型只加载一次，最多默认 8 次请求。复核提示词不提供粗看猜测的球队方向或文字证据；这仍是同一个模型的新请求，不是独立模型或人工验证。每条回答必须为 `confirmed`、`rejected`、`uncertain`，分别保留前后控球证据和理由。确认需要不同的已知控制方及候选内部的过渡范围；排除需要同一已知控制方，模糊情况保留不确定。格式错误单独保存，不当作排除。`candidate-review.json` 保存模型回答和原视频过渡范围，原始窗口及实际输入图保留供人工检查；即使模型确认也保留 `needs_human_review=true`，不会导出正式片段或写球员库。

模型返回 `confirmed` 但原样复述整个候选范围时，额外标记 `localization_status=unchanged_candidate_range`、`range_narrowed=false`；模型缩小范围也只标为 `model_narrowed_unverified`，不表示定位正确。死球阶段捡球、把球交回裁判等持球动作不代表有效回合转换。切分基线需区分“新方开始准备发球”和“场内队员接球控制”：若处理片段需保留完整发球过程，采用前者；若研究场内控制时刻，单独记录后者。关键帧检查参考不应冒充独立人工标注的测试真值。

只分析并保留人工检查机会：

```bash
.venv/bin/python -m cli.segment_video --env-file .env \
  --input runtime/data/game-001.mp4 \
  --output-dir runtime/outputs/game-001/segmentation --dry-run

/path/to/qwen3/bin/python -u -m cli.segment_video --env-file .env \
  --input runtime/data/game-001.mp4 \
  --output-dir runtime/outputs/game-001/segmentation \
  --stage analyze --window-s 20 --overlap-s 6 \
  --coarse-frames 16 --review-frames 16 --coarse-method states --video-tokens 8192
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

全流程 `--stage full` 首先运行 `cli.bootstrap_tracklets`，默认 6 个 4 秒窗口（`--bootstrap-windows 6`、`--bootstrap-window-s 4`）。适用于任意足够长的比赛视频，不固定 15 分钟或某支球队。把源时间划为 6 层，每层 4 个稀疏中心帧只用于检测画质探查；选择清晰人物最多且质量更好的短窗口，再逐帧连续追踪。每个窗口只重置一次完整 MOTIP 跟踪器；远隔窗口不共享轨迹 ID。某层没有合格人物时从其他清晰时段补足；仍找不到 6 个互不重叠的清晰窗口时明确停止，不用模糊人物填满。每个完成窗口有断点记录，恢复不会重新追踪已完成窗口。

初始化人物图默认要求高度至少 128px、宽度至少 40px、检测分数至少 0.65、不贴画面边缘、框重叠不超过 0.15、Laplacian 方差至少 30。全部是可诊断的工程阈值。每条足够长的轨迹最多选 4 张相隔至少 0.5 秒的清晰全身图，直接入口 `--samples-per-track` 可调；Qwen 进一步拒绝裁判、非球员、多人混杂、显著遮挡和模糊。KPR 只提取通过筛选的图。两帧以上一致的队伍与号码构成强身份候选，号码冲突暂不进入初始库，单次读数不能作为锚点。未知号码但清晰的轨迹可辅助匹配已有锚点，尚不能关联的保持待定。单独清晰球衣颜色不等同明确队伍：只有恰好两个颜色各获至少两条轨迹支持时才自动建立匿名队伍；其他情况需显式配置或保持待定。自动证据仍非人工真值。

每人最多 6 个模板，每个已知朝向最多 2 个；只有单一朝向时不强求多角度。强同队同号可跨 KPR 外观距离关联，不同队伍/号码和同时间不同身体的矛盾会阻止合并。后续片段读取同一 `team_map.json`，使用完整人物图判断队伍、号码和角色，再更新原身份库。初始化短窗口不冒充回合，不产生动作事件。

自动探查选点不理想时，写一个 JSON 文件如 `[20, 150, 280, 410, 540, 670]`（仅为格式示例，需按输入时长和实际画质选择），传 `--bootstrap-windows-file starts.json`。也可用 `--bootstrap-team-map teams.json` 明确映射，如 `{"black":"home","white":"away"}`。基础尺寸与清晰度阈值可在直接调用 `cli.bootstrap_tracklets --help` 时调整。首次测试请使用新输出目录，保留旧照片库。`registration.json` 的 `ready_for_incremental` 表示是否已建立身份锚点；没有锚点时 `--stage full` 会先停止，让初始化结果可供排查。

单独初始化：

```bash
.venv/bin/python -u -m cli.process_match --env-file .env \
  --input runtime/data/game-001.mp4 --match-id game-001 \
  --output-dir runtime/outputs/game-001 --stage initialize
```

各模型通过独立子进程顺序执行，释放显存后进入下一个模型。初始化和片段共享 `results/library/players.sqlite3` 与同一 KPR 特征合同。已登记的窗口、人物图、特征与号码不能改写；模型、选窗或证据变化请新建运行目录。图库匹配/模板策略可按后文维护命令显式修订，不重新提取特征。`--skip-bootstrap` 可用于不需要初始化图库的对照。

```bash
.venv/bin/python -u -m cli.process_match --env-file .env \
  --input runtime/data/game-001.mp4 --match-id game-001 \
  --output-dir runtime/outputs/game-001 --stage process --limit 1
```

单片段通过后去掉 `--limit 1`。`cli.process_match` 默认 `--stage full` 自动先初始化图库、分析、导出，再运行全部片段；无默认 30 片段截断。有未决边界、或长视频只导出整段时会先停止下游处理。仅做探索性链路测试可显式传入 `--allow-unreviewed`，不会把切分结果改标为准确；`--stage process` 使用已导出的片段，不会自动补跑初始化。

## 初始化画质与只读复核

原下载命令的 `-c copy` 不重新编码，也不主动缩放。已知测试文件是 1280×720、25FPS；本地元数据不能证明上游是否另有更高清码流。Qwen 回合预览采用缩放和 CRF23，回合导出用原分辨率 H.264 CRF18，均与现在的 FFV1 初始化窗口区分。PNG 人物裁剪保留原尺寸，网页放大无法恢复源视频没有的细节。

```bash
.venv/bin/python -m cli.inspect_video_quality --input runtime/data/game-001.mp4
.venv/bin/python -m cli.inspect_video_quality --input runtime/data/game-001.mp4 \
  --derived runtime/outputs/game-001/results/bootstrap_tracklets/windows/WIN-01.mkv
.venv/bin/python -m cli.maintain_players --operation preview \
  --run-dir runtime/outputs/game-001/results \
  --output-dir runtime/outputs/game-001/diagnostics/six-window-preview
```

预览会导出可下载本地查看的 HTML。网页导入 `results/` 后可查看原尺寸模板、角色与队伍/号码证据，下载无损 MKV 短窗口。统计区分初始化轨迹与业务片段。CPU 自检 `cli.check_tracklet_bootstrap` 使用真实 FFmpeg 视频和明确标记的合成模型，验证逐像素无损、身份门槛、视角上限、恢复、维护与前端；不代表真实模型准确率。

## 六窗口零入选复核

六窗口初始化若 `accepted_clear_samples=0`，先导出全部候选图的只读报告。它不依赖非空 KPR 特征或球员库，不加载 GPU，也不改变清晰度门槛：

```bash
.venv/bin/python -m cli.audit_tracklet_bootstrap \
  --run-dir runtime/outputs/game-001/results \
  --output-dir runtime/outputs/game-001/diagnostics/tracklet-quality-audit \
  --context-samples 6
```

输出 `index.html`、原尺寸 PNG、最多六个原分辨率场景帧、逐图和逐轨迹 CSV、`report.json`、原始回答及同名 `.tar.gz`。HTML 点击图片可查看原尺寸；场景红框仅由网页叠加，不改原图像素。报告分别统计原始角色、遮挡、清晰、颜色、号码可读性和实际拒绝阶段，保留被质量门槛屏蔽的原始字段。`clear=false` 全部拒绝时，需要将原图与模型回答对照，不能直接放宽门槛，也不能据此判断 KPR 失败。下面的 `cli.audit_bootstrap` 仅适用于历史照片库。

## 历史照片库只读复核

当照片观察很多、实际图库仅有少数样本时，先保留原库并诊断，暂不放宽匹配阈值。旧版初始化复用了增量登记规则：待定身份不参与后续匹配；“可比较距离”还包含被同帧冲突排除的身份，可能阻止新身份建立。v2 已改用候选分组和有限模板机制；旧库通过后文维护命令预览/迁移，不能由旧版登记阻塞直接判断 KPR 模型效果差。

```bash
.venv/bin/python -m cli.audit_bootstrap \
  --run-dir runtime/outputs/game-001/results \
  --output-dir runtime/outputs/game-001/diagnostics/bootstrap-audit
```

只读取现成照片、特征、号码和 SQLite 快照，不调用 GPU、不改原库，使用新诊断目录。输出 `index.html`、`summary.json`、`observations.csv`、`pairs.csv` 和同名 `.tar.gz`；下载压缩包解开后，可直接本地打开 HTML 查看裁剪、跨帧近邻及拒绝原因。包内包含原特征和配置，但不含模型权重或完整视频。

先检查裁剪里是否混入裁判/观众、严重遮挡或过小人物，再检查每张图的跨帧近邻是否外观一致。旧库统计区分匹配距离超限、候选距离差不足、没有无冲突候选、新身份距离门槛未达到；v2 读取当前维护状态和真实拒绝原因，不把原始入库占位状态当成当前决定。两者均报告待定样本之间的近邻。

页面还提供多个阈值下的完整连接分组对照，组内任意两张图均满足距离上限，并禁止同帧不同检测合并。它只生成独立诊断假设，不写入球员库，不将号码读数作为合并依据；组数少不代表准确率高，同帧限制也不能替代人工真值。CPU 验证入口为 `cli.check_bootstrap`。

## 当前视频回合评估

先检查完整视频并标注全部回合转换，包含没有转换的负例时段，不能只查看模型候选。默认前 300 秒开发、之后验证；这是同一视频的内部留出，不是独立比赛测试集。`cli.process_match` 在切分后自动创建空的 `possession-reference.json` 并记录待标注状态，不覆盖已有参考；其他入口可用下面的命令单独初始化。模板不会把模型预测或此前少数目视参考当作真值：

```bash
.venv/bin/python -m cli.evaluate_possessions \
  --segmentation-dir runtime/outputs/game-001/segmentation \
  --annotations runtime/outputs/game-001/possession-reference.json --initialize
```

填写各范围 `boundaries_s`（原视频秒数），无法判定的连续区间写入 `ignore_ranges_s`。完整检查该范围后设 `complete=true`，人工核对后设顶层 `human_verified=true`。评估只使用明确完成的覆盖范围，空的未标注模板不会产生虚假的零召回或完美准确率。

```bash
.venv/bin/python -m cli.evaluate_possessions \
  --segmentation-dir runtime/outputs/game-001/segmentation \
  --annotations runtime/outputs/game-001/possession-reference.json \
  --tolerance-s 2 --max-candidate-width-s 10
```

输出 `evaluation.json` 和 UTF-8 BOM `evaluation.csv`。粗看候选与模型实际切点分别计算一对一 TP/FP/FN、precision、recall、F1、FP/分钟、匹配中点 MAE、候选宽度。JSON/CSV 的 precision、recall、F1 是 0–1 比例，网页显示百分比；时间单位为秒。超宽候选不能靠覆盖整场获得 TP；重复预测只能匹配一次，其余记 FP。默认 2 秒容差、10 秒候选宽度是可调工程口径；HOTA/MOTA 等跟踪指标需要人物框真值，不能从回合边界计算。相机切点不计入回合预测。

`cli.process_match --stage status` 打印当前完成/失败、初始化、入库与评估状态；不加载 GPU。导入 `results` 后默认进入球员照片展示，侧栏显示同样的统计与已完成标注的回合评估。人物与动作没有真值时明确显示未评估。逐片段 `batch_status.json` 保存阶段耗时、缓存复用与退出码。

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

## 图库初始化、增量复查与撤销

图库 v2 将原始观察、候选身份和实际匹配模板分开。独立照片先做质量/重复/重叠目标筛选，再进行带号码冲突、同帧互斥和组内兼容约束的分组；单张照片保持待定，跨时刻支持或合格多帧轨迹才产生候选。先建立严格外观种子，已有号码共识或明确同队证据才允许有条件地放宽距离；两次单帧同号输出不能独立证明同人。未知号码不会被丢弃。

每人默认最多 8 个模板，按清晰度、尺寸、曝光、已有视角标签和外观差异选择，替换低价值模板，保留来源。片段有 `identity_raw/kpr_samples.npz` 与对应 JSONL 时实际使用逐帧图片特征；旧结果只有轨迹原型时标记 `track_prototype`，不伪装成多视角图片。KPR 权重不进行在线训练。正常增量只暴露当前选定模板，待定样本每次新入库时重新参与候选分组；人工/显式全量复查可重新选择历史观察，处理拆分或错误污染。

原始模型证据不可覆盖。低质量、重复和人工淘汰是匹配库中的逻辑删除：停止作为模板、保留原因和可追溯记录，不自动删除源视频/原始裁剪/NPZ，否则旧检查、回滚与重新取样会失去依据。重叠几何仅筛出可疑多人框，不能自动保证目标纯净或识别所有裁判。默认不强制聚成 10 人，也不把候选组数当准确率。

已有结果先生成独立 CPU 预览，不加载 GPU、不更改原库：

```bash
.venv/bin/python -m cli.maintain_players \
  --run-dir runtime/outputs/game-001/results \
  --operation preview --output-dir runtime/diagnostics/gallery-v2-001
```

终端打印前后数量；预览目录包含 `memory_report.json`、独立 SQLite/JSON 索引、`index.html` 和缩略图。预览索引引用原始结果，不要作为新的比赛导入。可通过 `--strong-distance`、`--supported-distance`、`--gallery-limit` 做独立对照；默认 `0.12/0.2/8`。缩略图和流程统计不等于身份 precision/recall。

核对后应用到原库，必须使用当前修订号。若 `process_match` 仍持有比赛锁（包括暂停状态），先正常停止该任务；独立预览不需要停止它。

```bash
CV_RUN=runtime/outputs/game-001/results
CV_REV=$(.venv/bin/python -c 'import json,sys; from pathlib import Path; print(json.loads((Path(sys.argv[1])/"library/library_manifest.json").read_text())["review_revision"])' "$CV_RUN")
.venv/bin/python -m cli.maintain_players --run-dir "$CV_RUN" \
  --operation reprocess --expected-revision "$CV_REV"
```

应用前保存 `library/backups/<revision>.sqlite3`；状态和模板变更在事务内完成。后续普通入库沿用库内策略；通过维护命令显式更改策略，不会因恢复处理的默认参数覆盖已校准设置。最多 2000 个观察、4096 个工作模板的 CPU 规划预算可防止无界分配，超出时事务失败，保留原库；需要分批核查或显式调整 `MemoryPolicy`，不能静默丢观察。

`--operation rollback_memory --expected-revision <当前修订>` 撤销最近一次可撤销更新；原始观察和事件仍保存，查询按恢复后的身份重新关联。已有后续修订时，旧修订不能覆盖新结果。`--operation reject_observation --node-id <观察ID>` 淘汰误检/混人观察，`reinstate_observation` 解除人工淘汰并重新检查；两者同样要求当前修订号。

前端默认显示已建候选/人工确认球员，待定和淘汰观察单独筛选；提供重新处理、淘汰和撤销操作，详情展示当前实际模板。照片还可以人工标记为非球员。自动候选不是人工确认，待定/淘汰观察的事件保留为未关联，不强挂球员。

实现借鉴 [KPR（ECCV 2024）](https://arxiv.org/abs/2407.18112) 的可见部位匹配、[SpCL（NeurIPS 2020）](https://arxiv.org/abs/2006.02713) 的保守伪标签思路和 [iCaRL（CVPR 2017）](https://arxiv.org/abs/1611.07725) 的有限样本记忆原则。这里的规则是工程实现，没有运行 SpCL/iCaRL 训练或声称复现论文效果；多人目标关键点提示仍需独立的姿态证据和 GPU 对照。

## 无 GPU 验证

```bash
.venv/bin/python -m cli.check_match
.venv/bin/python -m cli.check_batch
.venv/bin/python -m cli.check_person_library
.venv/bin/python -m cli.check_gallery
.venv/bin/python -m cli.check_qwen
.venv/bin/python -m cli.check_web
```

CPU 检查使用合成模型结果并实际调用 FFmpeg。真实长视频的 Qwen 请求、联合 GPU 链路、切分召回率和身份/动作准确率均需在空闲 GPU 上另行测试。
