# basket_cut

篮球短视频的人物追踪、身份归档、动作事件分析与可视化检查。

```text
视频 → MOTIP → 原始轨迹 → 质量检查 → KPR 原始档案 → 人物归并
                                  └→ MMAction2 → 人物事件 → 区间
```

事件输出为 `[id, event, start, end, raw_score]`，时间单位为秒、区间为 `[start, end)`。
原始分数、OCR 读数、轨迹数量和档案减少量都不是准确率。身份归并只在单视频内进行。

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

## 号码候选（可选）

在独立 OCR 环境安装 `requirements-ocr.txt`，在 `.env` 配置
`BASKET_OCR_PYTHON`、`BASKET_OCR_MODEL_DIR`。给批量命令添加 `--jersey-ocr`；
首次下载权重需明确加 `--allow-ocr-download`。

使用通用 EasyOCR 读取近似躯干区域，多个独立帧支持才给出候选；冲突或读不清留空，
保留读数、位置和图片证据。不是训练好的球衣号码模型，也不能证明整条轨迹同人。
相同号码可能属于不同球队，OCR 永远不直接改写或归并身份。

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
python -m cli.check_privacy
# 网页/API 检查另外安装 requirements-web-test.txt
python -m cli.check_web
```

自检只说明代码逻辑可运行，不代表真实模型效果。服务未带认证，默认只监听回环地址，
不要直接公开。只提交通用代码；私人配置、运行记录、视频、权重不提交。
隐私扫描只检查工作树，删除文件不会自动清除旧 Git 历史。保留上游许可证与署名。
