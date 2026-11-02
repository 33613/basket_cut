# 同场比赛 10 个进球片段：独立 MOTIP / KPR 测试包

分支 `feature/score-event-player-linking`。这只是数据导出入口，不调用或修改主链路、Qwen、动作识别、球员库及其准入策略。进球检测由人工时间标注替代。此阶段不根据号码猜测轨迹，也不实现自动投篮球员定位。

首批来自用户的标注（时间单位秒，相对各短视频开头）：

| video_id | 进球时间 | 球衣号码 |
|---|---:|---:|
| 3 | 10 | 27 |
| 6 | 12 | 27 |
| 20 | 3.5 | 27 |
| 43 | 10 | 27 |
| 50 | 5 | 27 |
| 68 | 27 | 27 |
| 11 | 7 | 15 |
| 30 | 13 | 15 |
| 37 | 10 | 15 |
| 63 | 12 | 15 |

号码是事件级人工标签，不自动传播到同一视频的所有轨迹。需要确认这些号码属于预期队伍；队伍未知时保留空白，不能用号码单独判定跨队身份。10 个进球是小批量 demo 样本，不是重识别泛化性能基准。

## 1. Mac 准备和上传

在 `~/Downloads/goal-demo-input/` 放入这十个原始视频副本，分别命名为 `3.mp4`、`6.mp4`、`20.mp4`、`43.mp4`、`50.mp4`、`68.mp4`、`11.mp4`、`30.mp4`、`37.mp4`、`63.mp4`。仅复制/改名，不转码、不降分辨率。若使用其他文件名，生成标注模板后修改 `filename` 列即可。

Mac 终端执行（SSH 端口采用当前实例的 16310）：

```bash
ssh -p 16310 root@connect.bjb1.seetacloud.com 'mkdir -p /root/autodl-fs/data/goal-demo-input'
scp -P 16310 "$HOME/Downloads/goal-demo-input/"*.mp4 root@connect.bjb1.seetacloud.com:/root/autodl-fs/data/goal-demo-input/
```

上传中断可重传缺失文件。若本机已安装 rsync，可用支持断点的替代命令，不必两种都执行：

```bash
rsync -avP -e 'ssh -p 16310' "$HOME/Downloads/goal-demo-input/" root@connect.bjb1.seetacloud.com:/root/autodl-fs/data/goal-demo-input/
```

## 2. 服务器取分支，生成标注

```bash
cd /root/autodl-tmp/basket_cut
git fetch origin
git switch --track origin/feature/score-event-player-linking
```

若本地已经存在该分支，改为 `git switch feature/score-event-player-linking`，然后 `git pull --ff-only`。不要强制覆盖服务器已有改动。若 GitHub 网络需要加速，可沿用实例现有 `/etc/network_turbo` 配置，成功取代码后恢复原下载/推理环境。

生成预填的 10 条标注（已存在时拒绝覆盖）：

```bash
/root/autodl-tmp/env/control/bin/python -m cli.export_goal_test \
  --stage template \
  --output-dir /root/autodl-fs/data/goal-demo-input/annotations.csv
cat /root/autodl-fs/data/goal-demo-input/annotations.csv
```

可选列 `team_id`、`person_gt_id` 默认空白；只有人工确认后填写。`person_gt_id` 是评估答案，不是模型预测。一个视频当前对应一个进球事件。

## 3. 后台运行并导出

使用**全新**输出目录 `/root/autodl-fs/test`，不要指向现有生产结果目录。此处的 `test/` 是结果包，不是 Git 项目的测试代码目录。现有环境配置应包含有效的 MOTIP/KPR Python 和 checkpoint 路径。

```bash
mkdir -p /root/autodl-tmp/tool/manifests
cd /root/autodl-tmp/basket_cut
nohup env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/autodl-tmp/env/control/bin/python -u -m cli.export_goal_test \
  --env-file /root/autodl-tmp/tool/manifests/basket-autodl.env \
  --input-dir /root/autodl-fs/data/goal-demo-input \
  --annotations /root/autodl-fs/data/goal-demo-input/annotations.csv \
  --output-dir /root/autodl-fs/test \
  --match-id pl-nba-goals-001 \
  > /root/autodl-tmp/tool/manifests/goal-demo-export.log 2>&1 < /dev/null &
```

```bash
tail -n 50 /root/autodl-tmp/tool/manifests/goal-demo-export.log
cd /root/autodl-tmp/basket_cut
/root/autodl-tmp/env/control/bin/python -m cli.export_goal_test --output-dir /root/autodl-fs/test --stage status
```

执行顺序：复制并校验视频 → MOTIP 正向跟踪每个完整片段 → 原图采样 → KPR 特征 → 打包。MOTIP 和 KPR 在各自独立环境里顺序运行，进程退出释放显存。此入口不会检测其他任务的显存占用，建议在 GPU 无其他任务时启动。

模型/视频/采样设置不变时，重新执行同一命令可跳过已校验的阶段。改变视频、标注或采样设置应使用新输出目录。跟踪未完整完成时会重跑该视频；不会把部分 tracks 当成完成。准备视频复制阶段若中断且尚无 manifest，使用新的空输出目录。

首次打包后日志打印 `/root/autodl-fs/test.tar`，其中 `shooter_status=pending_manual_selection` 是正常状态，说明尚未人工选定进球球员轨迹。

## 4. 下载、复核并指定进球球员轨迹

Mac 终端：

```bash
scp -P 16310 root@connect.bjb1.seetacloud.com:/root/autodl-fs/test.tar "$HOME/Downloads/"
scp -P 16310 root@connect.bjb1.seetacloud.com:/root/autodl-fs/test.tar.sha256 "$HOME/Downloads/"
cd "$HOME/Downloads"
shasum -a 256 -c test.tar.sha256
tar -xf test.tar
open "$HOME/Downloads/test/review.html"
```

查看每个视频的 MOTIP 叠加视频 `track/<video_id>/tracks_vis.mp4`（导出为 H.264 预览）和人物图。原视频没有因此转码，KPR 裁剪始终使用原视频。找到真实投篮球员的轨迹 ID 后，编辑 `test/shooter_selection.csv` 的 `track_id`。可用 `nano "$HOME/Downloads/test/shooter_selection.csv"`，避免电子表格改动字段类型。

```csv
video_id,track_id,start_frame,end_frame,notes
3,7,,,human checked shooter
6,2,20,160,clean subrange only
```

以上 **7、2 是格式示例，不是实际推理结果**。未确定的行保持空白。`start_frame/end_frame` 是可选的原视频零起始、两端包含的帧号。默认导出该 ID 全部观察；若发生 ID 切换，必须限制为同一人物的区间。当前每个事件支持一条轨迹及一个连续区间，跨多个 ID 的情况先留空或选干净子段，不能宣称已经恢复完整人物轨迹。`track_review.csv` 可另外标 clean/mixed/uncertain/non_player 和人工身份，供两位同学评估使用，程序不会根据这些答案自动合并。

把人工选择表传回服务器（只传表，不要回传修改过的模型数据）：

```bash
scp -P 16310 "$HOME/Downloads/test/shooter_selection.csv" root@connect.bjb1.seetacloud.com:/root/autodl-fs/test/shooter_selection.csv
```

服务器执行：

```bash
cd /root/autodl-tmp/basket_cut
/root/autodl-tmp/env/control/bin/python -m cli.export_goal_test \
  --env-file /root/autodl-tmp/tool/manifests/basket-autodl.env \
  --output-dir /root/autodl-fs/test --stage shooter
/root/autodl-tmp/env/control/bin/python -m cli.export_goal_test \
  --output-dir /root/autodl-fs/test --stage pack
```

然后重新下载 `test.tar` 和 `.sha256`。**解压到新的本地目录**，避免旧文件与新包混合：

```bash
mkdir -p "$HOME/Downloads/goal-demo-final"
scp -P 16310 root@connect.bjb1.seetacloud.com:/root/autodl-fs/test.tar "$HOME/Downloads/goal-demo-final/"
scp -P 16310 root@connect.bjb1.seetacloud.com:/root/autodl-fs/test.tar.sha256 "$HOME/Downloads/goal-demo-final/"
cd "$HOME/Downloads/goal-demo-final"
shasum -a 256 -c test.tar.sha256
tar -xf test.tar
```

## 5. 最终目录与特征读取

```text
test/
  video/3.mp4 ...                  原视频字节不变
  annotations.csv                 人工进球时间及号码
  manifest.json                   比赛/视频/事件 ID、输入 SHA256、原始媒体信息
  review.html                     本地轨迹采样图复核
  shooter_selection.csv           人工指定的进球球员轨迹
  track/<video_id>/
    tracks.jsonl                  原视频坐标 XYXY、frame_idx、timestamp_s、track_id、det_score
    video_meta.json
    track_summary.json
    tracks_vis.mp4                全部人物框与局部轨迹 ID
    samples.jsonl                 图片与框/时间/全局唯一 sample_id 对照
    samples/*.png                 所有轨迹的原尺寸采样图，默认最多 12 张/轨迹
    track_review.csv              轨迹人工检查模板
    shooter/                      人工选定后才生成
      selection.json              记录人工选择方法和原始时间范围
      tracks.jsonl                投篮球员选定轨迹的原始观察
      crops/*.png                 选定轨迹每个观察帧的原尺寸人物裁剪
      trajectory.mp4              256×512 等比例补边的裁剪轨迹预览，无音频
      frames.jsonl                预览帧→原视频帧对照，缺测帧用黑画面保留时间
  identity/<video_id>/
    features.npz                  所有采样图的逐图 KPR 特征及部位可见性
    samples.jsonl                 与特征行一一对应，路径相对 test/ 根目录
    feature_contract.json         权重/配置/样本摘要、嵌入类型及归一化定义
    shooter_features.npz          人工所选人物范围内的采样特征子集
    shooter_samples.jsonl         子集行索引；很短的子区间可能没有采样特征
  export_status.json
  checksums.sha256
  README.txt
```

人物采样按轨迹时间等分，每段选锐度、尺寸和检测分数综合较好的观察，不使用 Qwen 门槛，不宣称无遮挡或真实身份已确认。保存全部人物的样本用于困难负例；整条 MOTIP 轨迹可能混人，跨片段没有自动合并，不生成真实 PL 身份。

```python
import json
from pathlib import Path
import numpy as np

root = Path("test")
folder = root / "identity" / "3"
samples = [json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines()]
with np.load(folder / "features.npz", allow_pickle=False) as data:
    print(data["embeddings"].shape)         # [N, parts, dimension]
    print(data["visibility_scores"].shape)  # [N, parts]
    assert list(data["sample_ids"]) == [s["sample_id"] for s in samples]
    print(root / samples[0]["crop_path"])
```

其他重识别方案使用同一批 PNG 重新提特征；不能把不同模型特征直接混在一个空间。KPR 距离遵循现有 visibility-aware 合同，距离不等于身份概率。球员库机制测试可先使用人工确认的关联，再换成模型关联；避免把识别错误和更新逻辑错误混在一起。

## 本地验证范围

`python -m analysis.goal_test_export_selfcheck` 使用合成轨迹和合成特征，真实执行视频解码、PNG 裁剪、H.264 人物轨迹预览、NPZ 索引、校验和及 tar 读回，并测试中断复用和子进程环境调度。它不代表 MOTIP/KPR GPU 已在本轮数据上运行，也不评价动作或身份准确率。
