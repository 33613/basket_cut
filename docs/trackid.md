# TrackID3x3 固定机位跨数据集评估

数据根目录保留上传包的 `manifest.csv`、`checksums.sha256`、`videos/{Indoor,Outdoor}/` 与 `annotations/original/{Indoor,Outdoor}/`。当前包是 Indoor 42 条、Outdoor 12 条；这是有标注的固定机位序列范围，不将其称为已核实的官方 test 划分，不混入 SportsMOT/TeamTrack 汇总。

本入口使用此前验证过的 SportsMOT 权重和 MOTIP 适配器：FP32、关闭 matmul/cuDNN TF32、PIL 预处理，短边 800、长边上限 1440；det=0.3、newborn=0.6、id=0.2、object-max、miss_tolerance=60 帧。模型一次加载，各原始视频重置 tracker，不通过 KPR/Qwen、号码、人数上限或 GT 删除预测。Indoor 与 Outdoor 的帧率不同，60 帧对应的秒数也不同，本轮保留模型原协议，不为数据集单独调参。

上传工具的 `check_status` 名称不作为评估通过证明：原文、VFR 状态和备注全部保留，非标准成功/警告标签会打印并重新进行实际检查；明确失败状态和非零解码错误、非法 GT、重复身份计数会阻止所选序列。未选中的视角状态不影响单视角试验。不修改源 manifest 将其伪装成 `verified`。

## 原始标注与时间

源 MOT10 的前六列是 `frame,id,x,y,width,height`，后四列均为 `-1` 占位。它们不能被当作 MOT17 的无效置信度、忽略类别或遮挡信息。派生评估 GT 仅将尾部写为 `1,1,1`，供 TrackEval 的球员类别使用；前六列不偏移、不裁剪，源文件不改写。遇到非占位尾字段直接停止，先核查语义，不猜测忽略规则。越界框保留并计数；不补缺帧人物框。

GT 一基帧号对应原视频按顺序解码出的第 N 帧。Indoor 示例约 19.98 FPS，Outdoor 为 4K MOV、约 29.97 FPS，可能存在时间戳变化。没有统一 FPS、补帧、抽帧、转码或把 Outdoor 先降为 720p。模型内部按原协议缩放，输出映射回原图坐标。检查每帧解码尺寸、总帧数、重复 ID、GT 几何、manifest 汇总数、来源 SHA-256。校验清单中的 `.DS_Store`、`._*` 和 `__MACOSX/` 仅为 macOS 文件管理元数据，打印说明并排除，不要求复制到服务器；不改写校验清单。视频、GT 和其他正常清单文件缺失或损坏仍会失败。记录自动旋转元数据，并导出首/中/尾 GT 叠加图片；旋转是否与标注方向一致仍须目视核查，尺寸检查不能排除 180 度方向问题。

原视频逐帧流式读取，不解包成庞大的 PNG/JPEG 图像目录。准备阶段 CPU 校验和推理各读取视频，文件存储较慢时会增加耗时。模型推理完成后评估前再核查原始视频/GT，避免恢复时输入悄悄变化。

无需 Python 长单行即可打印原上传检查状态（只读 CPU，不要求输出目录）：

```bash
/root/autodl-tmp/env/control/bin/python -m cli.benchmark_trackid \
  --dataset-root /root/autodl-fs/data/trackid3x3-fixed-camera --stage inspect
```

## 单 GPU 顺序运行

长视频初始化先完成并退出所有模型进程，再启动本评估。同一输出目录不要并发运行或改变选择、权重及代码。GPU 工作者会检查当前计算进程，有其他 GPU 任务则停止。

控制环境沿用 `requirements-teamtrack.txt`，无需更改 MOTIP 的 PyTorch：

```bash
/root/autodl-tmp/env/control/bin/python -m pip install -r requirements-teamtrack.txt
/root/autodl-tmp/env/control/bin/python -m cli.check_trackid
```

首次取每个视角中帧数最少的一条**完整**序列检查转换与对齐，不截断视频，输出选择及其原始名称。`--limit-per-view 1` 的结果不能代表全部 54 条：

```bash
/root/autodl-tmp/env/control/bin/python -u -m cli.benchmark_trackid \
  --env-file /root/autodl-tmp/tool/manifests/basket-autodl.env \
  --dataset-root /root/autodl-fs/data/trackid3x3-fixed-camera \
  --output-dir /root/autodl-tmp/output/benchmarks/trackid-smoke-001 \
  --limit-per-view 1 --stage all
```

检查 `fixed_camera/alignment/`：`gt-*.jpg` 是首/中/尾的 GT，`comparison-*.jpg` 是前 100 帧范围内的 GT（绿）与预测（红）。图片仅为展示缩放到最多 1600 像素宽；评分始终采用原图坐标。先核对两种视角的框位置、旋转方向和帧顺序，再用新输出目录运行全部序列：

```bash
/root/autodl-tmp/env/control/bin/python -u -m cli.benchmark_trackid \
  --env-file /root/autodl-tmp/tool/manifests/basket-autodl.env \
  --dataset-root /root/autodl-fs/data/trackid3x3-fixed-camera \
  --output-dir /root/autodl-tmp/output/benchmarks/trackid-all-001 \
  --limit-per-view 0 --stage all
```

`--subset Indoor` 或 `--subset Outdoor` 可限制视角。`--stage prepare` 只用 CPU，可在初始化占用 GPU 时先做；准备后用相同选择执行 `all` 会校验复用。`--stage infer` 是 MOTIP 环境的内部工作入口；`evaluate` 和 `preview` 用控制环境，`status` 不调用 GPU、也不重新扫描所有原视频。

## 指标与结果

```bash
/root/autodl-tmp/env/control/bin/python -m cli.benchmark_trackid \
  --output-dir /root/autodl-tmp/output/benchmarks/trackid-smoke-001 --stage status
```

终端打印完整序列计数、Indoor/Outdoor 和总体指标。完成状态须通过清单、预测、完成标记与汇总校验。异常操作另保存 `last_error.json`，不会把缺失/中断的推理当完成。

```text
<output>/
  status.json
  last_error.json                        # 出错时
  fixed_camera/
    dataset.json                        # 选择、原始/派生哈希、帧率、转换规则
    inference_protocol.json             # 权重、模型参数、代码、精度与解码器
    inference_environment.json
    completed/<sequence>.json
    gt/<sequence>/{seqinfo.ini,gt/gt.txt}
    predictions/motip/data/<sequence>.txt
    alignment/<sequence>/{gt-*.jpg,comparison-*.jpg}
    metrics/summary.csv
    metrics/hota_thresholds.csv
    metrics/metric_protocol.json
    metrics/trackeval_raw.json
    metrics/sequences/<sequence>/{clear_frames.csv,clear_matches.csv}
```

`summary.csv` 包含逐序列、Indoor、Outdoor、总体的 HOTA、DetA、AssA、MOTA、IDF1、TP、FP、FN、recall、precision、IDSW、Frag、IDTP、IDFP、IDFN、IDR、IDP。HOTA 按 0.05..0.95 共 19 个 IoU 阈值计算；CLEAR/Identity 阈值 0.5。百分比单位为 0–100；总体调用 TrackEval 合并，不算各视频分数的简单平均。逐帧匹配属于 CLEAR 的 IoU 0.5 口径，不能把这些 TP/FP/FN 当作全部 HOTA 阈值计数。

跨序列身份保持独立，身份数不能相加当独立球员人数。54 条来源视频可能来自少数拍摄场次，不把序列当作 54 场独立比赛。此处验证的是固定 SportsMOT 权重的跨数据表现，不是 TrackID3x3 训练/微调后的论文复现。

CPU 自测用合成视频、模拟追踪输出和真实 TrackEval，检查原始标注不变、不同 FPS、对齐、计数、恢复与损坏检测；真实 GPU 效果须由服务器运行得到。
