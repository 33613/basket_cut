# basket_cut

篮球短视频的人物追踪、身份归档、动作事件分析与可视化检查。

```text
视频 → MOTIP → 原始轨迹 → 质量检查 → KPR 原始档案 → 人物归并
                                  └→ MMAction2 → 人物事件 → 区间
```

事件输出为 `[id, event, start, end, raw_score]`，时间单位为秒、区间为 `[start, end)`。
原始分数、OCR 读数、轨迹数量和档案减少量都不是准确率。
片段内身份与可选的比赛级候选身份分开保存，不改写原始轨迹。

## 结构与环境

- `pipeline/{tracking,identity,action}`：独立模块；`cli`：运行接口；`workflows`：编排。
- `contracts`：数据契约与配置；`analysis`：评估、检查与可视化。
- `tools`：数据和权重准备；`web`：检查界面；`configs`：通用配置。
- `runtime/{data,models,outputs}`：默认资源目录，均不提交。上游源码不修改。

Python 3.10+。GPU 模块分别按 [MOTIP](MOTIP/docs/INSTALL.md)、
[MMAction2](MMAction2/docs/en/get_started/installation.md)、
[KPR](https://github.com/VlSomers/keypoint_promptable_reidentification) 官方说明安装到独立环境。
KPR 源码放在根目录 `KPR/`。权重见 [MOTIP Model Zoo](MOTIP/docs/MODEL_ZOO.md) 和
[MultiSports 配置](MMAction2/configs/detection/slowfast/README.md)；KPR 可用 `cli.download_kpr` 下载。

复制 `.env.example` 为私有 `.env`，填写数据、结果、权重和各环境 Python 的实际路径。
`cli.serve_web`、`cli.process_batch`、`cli.import_web_results` 接受 `--env-file .env`，
无需额外安装 dotenv；也可以设置环境变量。不要求特定机器、用户名或环境名称。

## 批量处理

先用 `--list-only` 查看预计解压大小；只解压选中的视频，不展开整个数据集：

```bash
python -m cli.extract_videos --archive runtime/data/videos.tar \
  --output-dir runtime/data/subset --limit 30 --max-gb 6 --list-only
python -m cli.extract_videos --archive runtime/data/videos.tar \
  --output-dir runtime/data/subset --limit 30 --max-gb 6

python -m cli.process_batch --env-file .env \
  --input-dir runtime/data/subset --selection runtime/data/subset/video_selection.json \
  --output-dir runtime/outputs/batch30 --limit 30 --target full --dry-run
python -m cli.process_batch --env-file .env \
  --input-dir runtime/data/subset --selection runtime/data/subset/video_selection.json \
  --output-dir runtime/outputs/batch30 --limit 30 --target full
```

先试 1 条，再运行固定 30 条；后续扩大批次用新的输出目录。任务串行、失败保留、
同配置重跑会从未完成阶段继续。输入、权重、代码或参数变化会拒绝静默复用；不要随意
`--force` 覆盖用于评估的基线。`--target identity` 跳过动作，只跑追踪、身份与检查。
`--merge-distance` 必须明确指定，未给出时不自动归并；距离需要小样本校准，不是概率。

每视频保留 `tracking` 原始轨迹、`quality` 过滤轨迹、`identity_raw` 原始档案、
`identity` 归并档案、`action` 预测与事件、`analysis/track_review` 时序证据。
单模块参数可用 `python -m cli.tracking --help` 等查看。

## 跨片段人物库与事件检索

同场比赛的批次可以添加 `--cross-clip-distance 0.2`，在所有片段完成后建立
`library/`。它独立于 `--merge-distance`（片段内归并）；二者都是未校准的
距离阈值，不是概率。不要把不同比赛混进同一人物库。

```bash
python -m cli.process_batch --env-file .env \
  --input-dir runtime/data/match --output-dir runtime/outputs/match100 \
  --limit 100 --target full --merge-distance 0.2 --cross-clip-distance 0.2

# 只重建跨片段库，不重跑任何模型，也不改片段内结果。
python -m cli.build_person_library --run-dir runtime/outputs/match100 \
  --max-distance 0.2 --overwrite
```

复用 KPR 部位原型与可见性，采用与
[KPR 官方距离实现](https://github.com/VlSomers/keypoint_promptable_reidentification/blob/main/torchreid/metrics/distance.py)
一致的可见性加权欧氏距离 / 2。归组使用 complete linkage：所有成员原型都必须
满足阈值，不沿相似度链无限合并。同片段不同局部档案禁止在比赛级再次归并；
样本不足、外观不一致、轨迹告警或缺少可比部位时保留独立档案。不会强制输出10个人。
新 KPR 输出记录 checkpoint 校验和、prompt 模式与特征布局，拒绝混用不同特征空间；
旧缓存只有在确认权重与设置一致时才显式使用 `--allow-legacy-features`，并保留警告。

`library/people.jsonl` 保存比赛级 `G-*` 候选档案、代表图引用和来源成员；
`identity_map.jsonl` 保留 `(clip_name, local_person_id, raw_track_ids) → global_person_id`；
`match_pairs.jsonl` 保存有界的最近候选与匹配证据；`events.jsonl` 输出比赛级
`id`，同时保留 `local_person_id`、来源事件与片段名。时间仍是原片段内的秒数，
不会虚构整场时间轴。`library_manifest.json` 记录覆盖、参数、警告和源文件签名。
缺少任何计划片段、事件引用无效或来源已改变时拒绝静默生成完整人物库。

导入该批次后，网页“比赛人物库”可按人物、事件类型、原始分数和片段名检索，
点击事件跳转原片段；查看多张来源图、最近匹配距离，拆出误合并成员。
相同的人工“球队＋号码”标签可归组，但不自动将 OCR 号码当成身份真值。
人工修正与重新索引保存到网页索引目录的独立 sidecar，不改模型输出，
有版本冲突和失效检查。恢复自动归并只清空人物库修正，不清空轨迹复核。
人工归组后的检索是修正后验收，不是独立身份识别准确率。

## 号码候选（可选）

在独立 OCR 环境安装 `requirements-ocr.txt`，在 `.env` 配置
`BASKET_OCR_PYTHON`、`BASKET_OCR_MODEL_DIR`。给批量命令添加 `--jersey-ocr`；
首次下载权重需明确加 `--allow-ocr-download`。

使用通用 EasyOCR 读取近似躯干区域，多个独立帧支持才给出候选；冲突或读不清留空，
保留读数、位置和图片证据。不是训练好的球衣号码模型，也不能证明整条轨迹同人。
相同号码可能属于不同球队，OCR 永远不直接改写或归并身份。

## 专用球衣号码基线（JNR）

[Single-Stage Uncertainty-Aware Jersey Number Recognition in Soccer](https://github.com/lukaszgrad/uncertainty-jnr)
提供 ViT-S / ViT-B 的 SoccerNet 权重。先用 ViT-S；论文最佳模型未公开，
足球测试准确率不能当作篮球准确率。该上游采用 CC-BY-SA-4.0，使用或分发须遵守许可。
适配器不修改上游源码，不安装其训练/云日志依赖。独立 Python 3.11 环境：

```bash
conda create -p ./runtime/envs/jnr python=3.11 pip -y
conda activate ./runtime/envs/jnr
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-jnr.txt
git clone https://github.com/lukaszgrad/uncertainty-jnr.git UncertaintyJNR
git -C UncertaintyJNR checkout f19d9cb90e1a67d5ffe44acbf69fb348fe6525e7
python -m cli.download_jnr --model vit-small --output-dir runtime/models/jnr
python -m cli.inspect_jnr --device cuda
```

在 `.env` 配置 `BASKET_JNR_PYTHON`、`BASKET_JNR_ROOT`、`BASKET_JNR_CONFIG`、
`BASKET_JNR_CHECKPOINT`。直接调用 `cli.inspect_jnr` / `cli.jersey_jnr` 不自动加载
`.env`，自定义资源路径请用命令参数；批处理加载 `.env`。
权重默认安全加载、严格匹配所有层。只有确认来自作者的文件，且安全加载确实拒绝
训练元数据时，才加 `--trust-checkpoint`（批量为 `--jnr-trust-checkpoint`）。
Google Drive 下载失败时从作者链接下载后传入 `--checkpoint`，不要找不明镜像。

```bash
python -m cli.process_batch --env-file .env \
  --input-dir runtime/data/smoke5 --output-dir runtime/outputs/jnr5 \
  --limit 5 --target full --jersey-jnr --merge-distance 0.2 --cross-clip-distance 0.2
```

复用 KPR 时序采样的人物原图，按作者的躯干裁剪、缩放和归一化运行模型。
这不是为号码可读性专门选帧；侧面、遮挡、框内多人的样本可能无法读号。
保存 `identity/jersey_{readings,tracks,people}.jsonl`、`jersey_predictions.npz`
（全部100类分数与不确定性）、`jersey_summary.json`（权重/源码校验和、阈值与拒绝原因）。
默认分数≥0.8、不确定性≤0.2、top2差≥0.2，至少2帧且间隔≥0.25秒支持；
这些是初始工程阈值，不是校准的准确率。强读数冲突不会被多数票掩盖。
模型类别仅0–99，无法区分0/00，因此类0留空并标记。没有球队/姓名识别。

`--jersey-jnr` 与 `--cross-clip-distance` 同用时，比赛级库会采用号码冲突约束：
可靠不同号禁止归并，局部号码混乱的档案保留待审；同号不降低KPR阈值、不强制归并。
片段内归并仍由KPR执行，JNR只检查并标记其冲突，不改写原始轨迹与局部ID。
重建库使用 `cli.build_person_library --use-jersey-evidence`；缺少JNR结果或混用权重/阈值会报错。
网页展示原始读数、拒绝帧、号码候选和归并来源，号码可检索但不是实名身份。
先复核5条中的接受读数是否正确、不可读是否留空、归并是否混人，再固定设置处理100条。
没有人工号码真值时只报告覆盖与冲突数量，不能报告号码准确率。

## 网页与验收

```bash
python -m pip install -r requirements-web.txt
python -m cli.import_web_results --env-file .env --result-dir runtime/outputs/batch30
python -m cli.serve_web --env-file .env --port 6006
```

打开 `http://127.0.0.1:6006`。原始/归并档案分开展示，能追溯来源轨迹、图片、事件。
批量审查按视频名、待审/告警/不确定状态筛选，每页 12 条视频、6 条轨迹；
可先检查保留轨迹，再切换已过滤碎片检查误删。分页草稿合并保存，不覆盖其他页，
支持“保存并下一条视频”。多个窗口同时修改时拒绝覆盖旧版本。人物素材图库独立于
事件结果，每页 18 个档案，支持号码候选搜索；不同视频中的同名 ID 不代表同一真人。
记录与复核持久保存在配置的索引目录；重启必须使用同一配置。点击右上角查看实际目录。
导入只建立只读索引；浏览器编码不兼容时安装 FFmpeg，再导入加 `--prepare-media`。
复核页可用配置的 OpenCV Python 在独立缓存生成截图，无需 GPU、不修改原预测。
旧过滤副本缺失原始 ID 时会明确拒绝完整评估，应导入原始基线。

验收只抓三个问题：轨迹是否始终同人、同人是否多 ID、归并档案是否混人。
完整看完轨迹才标记 `full_track`；抽样只能发现反例。保存队伍＋号码等标签后，
完整复核覆盖率、同人率、额外 ID 数、已标注纯轨迹上的身份成对 F1 才有对应分母。
无真值、无复核时数值留空；过滤量不能冒充正确率。基于同一复核标签修正后再评分，
只能叫修正后验收，不能叫独立模型准确率。

```bash
python -m cli.evaluate_run --run-dir runtime/outputs/batch30 \
  --web-output-root runtime/outputs/web
```

MultiSports 的公开验证标注可用于动作区间/动作 tube 评估，不提供整场持续真人身份与
号码真值；测试集元信息不能代替动作 GT。轨迹真值评估用 `cli.evaluate_tracking`；
动作 GT 适配用 `cli.prepare_multisports_reference` 与 `cli.evaluate_event_batch`。
网页也可导出冻结的复核包，用 `cli.evaluate_track_reviews` 离线重算。
100 条处理成功不等于泛化；开发数据与未参与调参的不同比赛留出数据分开。

## 回归与隐私

```bash
python -m cli.check_quality
python -m cli.check_evaluation
python -m cli.check_track_review
python -m cli.check_batch
python -m cli.check_person_library
python -m cli.check_jnr
python -m cli.check_privacy
# 网页/API 检查另外安装 requirements-web-test.txt
python -m cli.check_web
```

自检只说明代码逻辑可运行，不代表真实模型效果。服务未带认证，默认只监听回环地址，
不要直接公开。只提交通用代码；私人配置、运行记录、视频、权重不提交。
隐私扫描只检查工作树，删除文件不会自动清除旧 Git 历史。保留上游许可证与署名。
