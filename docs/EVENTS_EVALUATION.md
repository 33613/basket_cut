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
hf download MCG-NJU/SportsAction multisports_GT.pkl \
  --repo-type dataset \
  --local-dir /root/autodl-tmp/data/basket_cut/MultiSports
```

先查看可用篮球视频 key：

```bash
python -m cli.prepare_multisports_reference \
  --annotation /root/autodl-tmp/data/basket_cut/MultiSports/multisports_GT.pkl \
  --list-videos \
  --limit 30
```

选择一个已经准备好视频/帧的 key，导出项目统一格式：

```bash
python -m cli.prepare_multisports_reference \
  --annotation /root/autodl-tmp/data/basket_cut/MultiSports/multisports_GT.pkl \
  --video 'basketball/VIDEO_KEY' \
  --output /root/autodl-tmp/data/basket_cut/MultiSports/VIDEO_KEY.reference_events.jsonl \
  --overwrite
```

只有在明确希望检查那两个官方未评估类别时，才增加
`--include-unevaluated-labels`。用于正式分数时保持默认即可。

官方 Hugging Face 仓库的视频帧包为大型 `rawframes.tar`，不能像 SHOT 那样按
样本路径只下载一条视频。第一次只下载 annotation 可以先完成接口检查；正式
评估时再准备所选 key 对应的官方帧并转成输入视频。转码时必须保持官方的
25 FPS、完整帧序列和画面尺寸；裁剪、抽帧或缩放会破坏 tube 与预测轨迹的
帧号/坐标对应关系。

## 产品化事件评估

评估器只在类别正确、时间重合达到阈值、预测人物轨迹与参考人物 tube 重合
达到阈值时，才把事件记为可用：

```bash
python -m cli.evaluate_events \
  --reference /root/autodl-tmp/data/basket_cut/MultiSports/VIDEO_KEY.reference_events.jsonl \
  --prediction /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/action/events.jsonl \
  --tracks /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/tracking/tracks.jsonl \
  --output /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/analysis/event_metrics.json \
  --errors-output /root/autodl-tmp/outputs/basket_cut/pipeline/VIDEO/analysis/event_errors.jsonl \
  --temporal-iou-threshold 0.5 \
  --actor-mode tube \
  --actor-iou-threshold 0.3
```

输出只突出两个层次：

- `headline.usable_event_f1`：人物、动作、区间全部正确的端到端主分数；
- `event_and_interval_only.f1`：暂时忽略人物，只检查动作和区间。

两者差距主要指向人物追踪/身份问题；两者都低则先检查动作模型或时间聚合。
`event_errors.jsonl` 将每项标成 `usable`、`wrong_actor`、`false_positive` 或
`missed_reference`，用于网页逐条定位问题。

## 数据集依据

- [MultiSports 官方项目页](https://deeperaction.github.io/datasets/multisports.html)
- [MultiSports / SportsAction Hugging Face 数据页](https://huggingface.co/datasets/MCG-NJU/SportsAction)
- [FineSports 论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_FineSports_A_Multi-person_Hierarchical_Sports_Video_Dataset_for_Fine-grained_Action_CVPR_2024_paper.pdf)
- [PL-NBA 数据集仓库](https://github.com/holhouse/PL-NBA-Dataset)
- [SHOT 数据页](https://huggingface.co/datasets/muyu111/basketball)
