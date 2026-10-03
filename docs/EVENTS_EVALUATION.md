# 时间事件与产品评估

## 为什么当前选 MultiSports

当前动作后端使用 MMAction2 的 MultiSports SlowFast checkpoint，因此第一版
评估数据也选择 MultiSports 篮球子集：

- 模型词表与现有 `label_map.txt` 完全一致，共 18 个篮球动作；MultiSports
  官方评估排除 `basketball_save` 和 `basketball_jump_ball`，所以本项目参考
  数据转换器默认评估其余 16 类；
- 每个动作实例都有第一帧、最后一帧以及逐帧人物框；
- 开始/结束边界由数据集定义，不需要本项目重新人工标注；
- 可以检查事件类别、时间区间以及执行动作的人物是否正确。

它的限制也必须保留在评估结论中：MultiSports 提供的是每个动作实例的
人物 tube，不提供跨动作实例的真实球员姓名或稳定人物 ID。因此第一版通过
预测轨迹与参考人物 tube 的空间重合度检查“是否找对动作执行者”，不能用它
证明跨片段身份识别。

其他候选数据集暂不作为当前基准：

- FineSports 有篮球动作的空间框和时间边界，但使用 12 个大类、52 个细类，
  与当前 MultiSports checkpoint 不同，适合更换动作模型后接入；
- PL-NBA 有球员姓名、事件类型和时间戳，适合回合级事件理解，但公开标注以
  事件时间戳为主，不是当前动作 tube 的开始/结束定义；
- SHOT 有稳定球员 ID、轨迹和战术类别，但没有逐个原子动作的时间区间。

## 评估协议 v2：先验证计分，再跑真实视频

评估模块位于 `analysis/evaluation/`，调用入口位于 `cli/`，不修改 MOTIP、
KPR 或 MMAction2。`tests/` 继续忽略；随 Git 同步的回归自检保存在
`analysis/evaluation/selfcheck.py`。

```bash
cd /root/autodl-tmp/project/basket_cut
git pull --ff-only
python -m cli.check_evaluation
```

事件自检只依赖 Python 标准库，无需 GPU、权重或数据集。若还要验证 IDF1、
空轨迹、串 ID 和漏掉尾部，使用已有 motip 环境或独立 CPU 评估环境：

```bash
python -m pip install -r requirements-eval.txt
python -m cli.check_evaluation --with-tracking
```

自检通过证明计分规则实现正确，不证明模型在真实视频中效果好。

当前验收只保留三个层次：

- 追踪：IDF1 为主，串 ID 次数辅助定位；需要稳定人物 ID 真值。
- 事件：人物、类别和区间都通过的产品 F1；附 Precision、Recall 和 TP/FP/FN。
- 身份：现阶段归档质量检查，不宣称 KPR 检索准确率。未来有同人/不同人或
  query-gallery 真值后，单独评价人物检索。

事件通过条件：类别一致、temporal IoU >= 0.5；tube 模式在时间交集内的真值
标注帧上检查人物框。**同一条原始轨迹**的平均 IoU >= 0.5，且至少 80% 帧的
框 IoU >= 0.5；缺框按 0 算。不能逐帧从多个 ID 中挑最佳框。该规则是保守的
动作执行者定位标准，不是跨事件身份真值；未来轨迹修复合并需要另定协议。

合法候选采用最大配对数量的一对一匹配（遍历顺序优先较高 IoU，但不保证最大
总 IoU）。重复输出记 FP，遗漏记 FN。这个产品指标不是 MultiSports 官方 mAP。
`--profile multisports` 对真值和预测同时排除 save/jump_ball；自定义范围使用
重复的 `--exclude-label`，未知且未排除的预测类别仍计 FP。

空预测文件是有效的漏检结果；空真值文件表示已确认的负样本，预测会计 FP。
缺文件、JSON 损坏、错配视频是运行失败，不伪装成空预测。没有预测时 Precision
为 null；没有 GT 时 Recall 为 null；两者都为空时 F1 为 null，不能当成满分。
`raw_score` 是模型分数，不是概率校准后的准确率。

## 稳定事件契约

`events.jsonl` 每行至少包含产品需要的五个字段：

```json
{
  "id": "P0007",
  "event": "basketball_pass",
  "start": 2.4,
  "end": 3.12,
  "raw_score": 0.78
}
```

`start` 为包含端，`end` 为不包含端，单位都是源视频秒。文件同时保存
`video_id`、`event_id`、帧范围、原始 MOTIP IDs 和支持预测点数量，供评估与
问题定位使用。`raw_score` 只是区间内 MMAction2 分数的聚合值，不是校准后的
产品正确概率。

## 从动作预测点生成事件

聚合器位于 `adapters/mmaction2/`，属于第三方输出适配，不属于 MOTIP、KPR
或动作基线。它执行：同人物同类别分组、重复点去重、允许短暂漏判、相邻点
合并、采样单元边界展开和分数聚合。

```bash
cd /root/autodl-tmp/project/basket_cut
conda activate /root/autodl-tmp/envs/motip

python -m cli.aggregate_events \
  --input /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/action/actions_with_identity.jsonl \
  --video-meta /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/tracking/video_meta.json \
  --output /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/action/events.jsonl \
  --score-threshold 0.2 \
  --max-missing-steps 1 \
  --min-support 1 \
  --score-reducer mean \
  --overwrite
```

`--max-missing-steps 1` 表示两个正预测之间最多允许缺失一个模型采样点。
第一版保留 `--min-support 1`，因为传球等动作很短；实际阈值应在 MultiSports
验证视频上确定。如果一个视频没有动作分数超过阈值，聚合器会正常写出空的
`events.jsonl` 和计数为 0 的 summary；这表示“未预测出事件”，不是运行失败。

## 准备 MultiSports 参考事件

MultiSports 在 Hugging Face 上是需接受 CC BY-NC 4.0 条款的 gated 数据集。
登录并接受条款后，可以只下载较小的官方标注文件：

```bash
hf auth login
hf download MCG-NJU/SportsAction data/trainval/multisports_GT.pkl \
  --repo-type dataset \
  --local-dir /root/autodl-tmp/data/basket_cut/MultiSports
```

不要使用 `data/test/multisports_test.pkl`：官方 test split 不公开动作真值，
所以该文件没有 `gttubes`。本项目使用 trainval 包中带真值的 validation 列表。

先查看可用篮球视频 key：

```bash
python -m cli.prepare_multisports_reference \
  --annotation /root/autodl-tmp/data/basket_cut/MultiSports/data/trainval/multisports_GT.pkl \
  --list-videos \
  --split validation \
  --limit 30
```

选择一个已经准备好视频/帧的 key，导出项目统一格式：

```bash
python -m cli.prepare_multisports_reference \
  --annotation /root/autodl-tmp/data/basket_cut/MultiSports/data/trainval/multisports_GT.pkl \
  --video 'basketball/VIDEO_KEY' \
  --output /root/autodl-tmp/data/basket_cut/MultiSports/VIDEO_KEY.reference_events.jsonl \
  --overwrite
```

只有在明确希望检查那两个官方未评估类别时，才增加
`--include-unevaluated-labels`。用于正式分数时保持默认即可。

官方 Hugging Face 仓库当前篮球 trainval 包为 `data/trainval/basketball.tar`，不能像 SHOT 那样按
样本路径只下载一条视频。第一次只下载 annotation 可以先完成接口检查；正式
评估时再准备所选 key 对应的官方帧并转成输入视频。转码时必须保持官方的
25 FPS、完整帧序列和画面尺寸；裁剪、抽帧或缩放会破坏 tube 与预测轨迹的
帧号/坐标对应关系。

## 产品化事件评估

只对完整源视频做事件验收：提供 `--video-meta`，检查 processed_frames 等于
frame_count、秒数/帧号符合 FPS。重新导出的 MultiSports GT 还带源视频 FPS、
帧数与分辨率，评估器据此拒绝错误缩放/裁剪/转码。旧 GT 应重新导出。

以下路径是示例，必须替换成实际存在的文件。GT 官方 key 与 pipeline 的 video_id
可能不同；先核对它们确属同一个视频，再显式声明映射。原始 ID 不同且不声明
映射会报错，不能关闭这个保护。可从 video_meta.json 读取预测 video_id。

```bash
python -m cli.evaluate_events \
  --reference /root/autodl-tmp/data/basket_cut/MultiSports/VIDEO_KEY.reference_events.jsonl \
  --prediction /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/action/events.jsonl \
  --tracks /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/tracking/tracks.jsonl \
  --output /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/analysis/event_metrics.json \
  --errors-output /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/analysis/event_errors.jsonl \
  --video-meta /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/tracking/video_meta.json \
  --reference-video-id 'basketball/VIDEO_KEY' \
  --prediction-video-id 'ACTUAL_VIDEO_ID_FROM_META' \
  --profile multisports \
  --temporal-iou-threshold 0.5 \
  --actor-mode tube \
  --actor-iou-threshold 0.5 \
  --actor-min-coverage 0.8
```

输出只突出两个层次：

- `headline.value`（`headline.name=usable_event_f1`）：人物、动作、区间通过协议的主分数；
- `event_and_interval_only.f1`：暂时忽略人物，只检查动作和区间。

两者差距提示动作执行者定位问题，不能归因于 KPR，也不能把 exact 模式用于
未经对齐的任意 P0001/GT ID。`--actor-mode ignore` 的主分数改名为
`event_interval_f1`，不宣称人物正确。

`event_errors.jsonl` 区分 usable、duplicate、wrong_actor、wrong_class、
wrong_interval、false_positive、missed_reference。这些是按协议得到的定位线索，
不是互斥的真实故障原因；同一事件可能同时有人物和时间错误。报告附每类
TP/FP/FN、支持数量、忽略数量和已匹配事件的开始/结束偏差（秒）。不能只看
F1 而忽略边界偏差，也不能把这些自定义分数与论文 mAP 直接比较。

## 固定清单的批量验收

先以 PL-NBA 的 5 条检查链路；找到对应 JSON 并对齐类别/时间后才可量化。
PL-NBA 时间点不可虚构为动作区间。MultiSports 用官方 validation、按来源
视频分开调试/保留评估样本，报告每类事件支持数量。选择 50 条本身不保证
代表性或泛化，当前动作模型同数据集的效果也不是外部泛化证明。

在数据目录创建 `evaluation_manifest.json`，每条记录对应一个已处理视频。
它不同于下载用的字符串 key 名单。相对路径按 manifest 所在目录解析：

```json
{
  "videos": [
    {
      "name": "basketball/v_-6Os86HzwCs_c001",
      "reference": "references/v_-6Os86HzwCs_c001.events.jsonl",
      "prediction": "/root/autodl-tmp/outputs/basket_cut/multisports/v_-6Os86HzwCs_c001/action/events.jsonl",
      "tracks": "/root/autodl-tmp/outputs/basket_cut/multisports/v_-6Os86HzwCs_c001/tracking/tracks.jsonl",
      "video_meta": "/root/autodl-tmp/outputs/basket_cut/multisports/v_-6Os86HzwCs_c001/tracking/video_meta.json",
      "reference_video_id": "basketball/v_-6Os86HzwCs_c001",
      "prediction_video_id": "ACTUAL_VIDEO_ID_FROM_META"
    }
  ]
}
```

以上仍为文件路径模板，实际不存在时不要直接运行。准备完成后：

```bash
python -m cli.evaluate_event_batch \
  --manifest /root/autodl-tmp/data/basket_cut/MultiSports/manifests/evaluation_manifest.json \
  --output-dir /root/autodl-tmp/outputs/basket_cut/evaluation/multisports_v2 \
  --profile multisports --actor-mode tube
```

`summary.json` 汇总所有视频的 TP/FP/FN 后算微平均 F1，绝不平均每个视频的
F1。每个视频报告放在数字子目录，原始名字保留在 summary 中。批量要求
video_meta；任一视频失败，status=incomplete、headline.value=null、退出码=1。
已完成样本只在 completed_subset 中展示，不能冒充完整评估结果。MultiSports
批量验收还要求新版 GT 的源视频属性；旧参考必须重新导出。

阈值/类别范围应只在调试样本上确定，再冻结清单、checkpoint、配置、Git commit
和阈值后跑保留样本。沿用动作分数缓存可重复评估；但若旧推理只保存有限
top-k/阈值以上类别，降低聚合阈值不能恢复未保存的分数，需要重新推理。

## 数据集依据

- [MultiSports 官方项目页](https://deeperaction.github.io/datasets/multisports.html)
- [MultiSports / SportsAction Hugging Face 数据页](https://huggingface.co/datasets/MCG-NJU/SportsAction)
- [FineSports 论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_FineSports_A_Multi-person_Hierarchical_Sports_Video_Dataset_for_Fine-grained_Action_CVPR_2024_paper.pdf)
- [PL-NBA 数据集仓库](https://github.com/holhouse/PL-NBA-Dataset)
- [SHOT 数据页](https://huggingface.co/datasets/muyu111/basketball)
