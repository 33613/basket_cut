# TCEI 环境、适配与跟踪对照

论文：[Dual-level Adaptation for Multi-Object Tracking](https://arxiv.org/abs/2603.21629)。作者仓库：[1941Zpf/TCEI](https://github.com/1941Zpf/TCEI)，固定提交 `145d1b8431398156f8d9f854430e306fdee39eaa`。

## 环境

作者推荐 Python3.12、PyTorch2.4.0、torchvision0.19.0、CUDA12.1。优先复用已经可用的 MOTIP 环境和 SportsMOT checkpoint；球检测环境独立保留。TCEI 使用推理缓存，没有在线反向传播，也没有新增可训练权重。

```bash
# GitHub 下载的加速设置只在这个子 shell 生效。
(
set -e
source /etc/network_turbo
mkdir -p /root/autodl-tmp/third_party
if [ ! -d /root/autodl-tmp/third_party/TCEI ]; then
  git clone https://github.com/1941Zpf/TCEI.git /root/autodl-tmp/third_party/TCEI
fi
git -C /root/autodl-tmp/third_party/TCEI fetch origin 145d1b8431398156f8d9f854430e306fdee39eaa
git -C /root/autodl-tmp/third_party/TCEI checkout --detach 145d1b8431398156f8d9f854430e306fdee39eaa
)
```

已有且通过预检的源码不用重新克隆。下面从项目根目录运行，解释器、源码和模型路径按部署调整：

```bash
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
/root/autodl-tmp/env/motip/bin/python -u -m cli.tcei_preflight \
  --source-root /root/autodl-tmp/third_party/TCEI \
  --checkpoint /root/autodl-tmp/model/motip/r50_deformable_detr_motip_sportsmot.pth \
  --report /root/autodl-tmp/tool/manifests/tcei/preflight.json
```

成功状态为 `passed_environment_and_model_smoke`。检查 CUDA/NMS、Deformable Attention 与参考实现、作者入口导入、原权重严格加载和检测/ID解码前向；合成输入不代表真实跟踪准确率。完整 checkpoint 加载前跳过多余 ImageNet 下载。

缺少普通依赖时参考 `requirements-tcei.txt`，保持现有 torch/torchvision/numpy 版本。已有算子通过就复用；只有算子缺失或ABI不兼容时，才在匹配的编译器 CUDA 下重编译。RTX4090 示例：

```bash
cd /root/autodl-tmp/third_party/TCEI/models/ops
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9 \
/root/autodl-tmp/env/motip/bin/python -m pip install \
  --no-build-isolation --no-deps --force-reinstall .
```

## 外部适配器

- `adapters/tcei.py` 校验作者提交和已跟踪源码，按文件加载熵/融合函数；不把作者的 `models/utils/configs` 加入项目推理进程的搜索路径。
- `pipeline/tracking/tcei.py` 包装原 RuntimeTracker。临时钩子取得最终 ID 分类头输入并修正输出 logits，原版完成后续 ID 分配、过滤和轨迹更新；钩子在异常时也清理。
- 对象层使用原 DETR 特征，缓存键为当前轨迹实际 ID；ID层使用解码末端 ID 特征，缓存键为 ID 词元。它不是跨视频永久球员库。
- `MotipBackend` 默认不启用校准。实验显式传入配置，支持 `off/object/id/dual`。当前只验证 FP32、softmax、object-max、单GPU；不影响现有进球入口。

实现名为 `motip-hooks-corrected-v1`，不是逐行复现。保留作者 SportsMOT 参数：正/负容量3/2，对象层正参数(alpha,beta)=(1,3)、负=(0.117,1.5)、温度0.05；ID层正=(1,5)、负=(0.117,2)，熵范围(0.2,0.5)、负掩码(0.03,1)，lambda_id=1。ID层作者温度参数未用于计算。缓存先加入当前样本再校准，保留原熵归一化和融合函数。

修订点明确如下：

- 单类/相同相似度缓存避免 min-max 除零；零特征安全归一化。
- 负对象缓存按 ID 词元概率映射到实际轨迹，避免把检测序号当词元，并按各缓存行正确生成掩码。
- 无硬编码类别50或 `.cuda()`，张量沿输入设备和 FP32 计算。
- 新生轨迹使用过滤后对应的原对象特征与不确定性，避免过滤前后下标错位。

作者 ID层经验缓存跨视频保留。默认实验 `--cache-scope sequence` 每序列清空，以消除处理历史差异；`stream` 显式复用经验，固定清单顺序并分别报告。局部对象缓存始终每序列重置，不同实验不共享经验。stream 中断后重跑整个变体恢复经验，不从中间隐式续接。

## 自检与 SportsMOT 运行

```bash
/root/autodl-tmp/env/motip/bin/python -m analysis.tcei_selfcheck \
  --source-root /root/autodl-tmp/third_party/TCEI
```

CPU检查使用原 RuntimeTracker 和合成模型，检查关闭校准精确一致、最终共享头特征、异常钩子清理、单类/重复缓存、负ID映射、序列重置和作者宽缓存公式。实际图像的关闭等价性由下面的GPU对照门禁检查。

先找到历史完整 SportsMOT 篮球 val 运行目录，要求含 `dataset.json`、`experiment_protocol.json`、`summary.csv`，不能拿上传检查日志或前5条结果代替全量清单。

记不清目录时，将 `--baseline-run ...` 替换成 `--baseline-search-root /root/autodl-fs /root/autodl-tmp`。只读查找已完成且哈希匹配的完整旧实验，跳过环境、模型和图像目录；若找到不同数据范围会停止，要求明确指定。

```bash
python -u -m cli.benchmark_tcei --env-file .env \
  --baseline-run /outputs/sportsmot-native/official-all \
  --source-root /root/autodl-tmp/third_party/TCEI \
  --output-dir /outputs/tcei-basketball-dev \
  --limit 1 --variants baseline off object id dual
```

控制入口调度已有 MOTIP Python，逐变体运行。先完成原版 baseline 和关闭校准 off，逐帧比较原始ID、框、分数，任何差异都停止后面的校准实验。所有变体使用同一权重、FP32、PIL预处理、TF32关闭、阈值和原始图像。GT仅用于最后的 TrackEval，不参与缓存或伪标签。

通过开发序列后，使用**新目录**和 `--limit 0` 跑完整清单，通常15条、12557帧。参数先固定，不依据完整确认集成绩反复调整。开发序列与完整范围的重叠需说明，完整范围不宣称独立留出测试。

```bash
python -m cli.benchmark_tcei --output-dir /outputs/tcei-basketball-dev --stage status
python -m cli.benchmark_tcei --output-dir /outputs/tcei-basketball-dev --stage evaluate
```

`summary.csv` 为各序列和总指标，`deltas.csv` 相对本次 baseline，`performance.csv` 记录同范围含解码/推理/输出的耗时、FPS与峰值显存，`parity.json` 记录关闭等价性。模型初始化和源文件哈希时间不算入FPS。`protocol.json` 固定代码、作者版本、权重、原帧清单和缓存协议；改变任何一项需新目录。逐序列缓存模式校验后可断点恢复。

历史参考 HOTA76.94、AssA67.79、IDF1 84.19、IDSW305，只用于同范围核对。主要比较 AssA、IDF1、HOTA、IDSW、Frag，同时看 DetA、MOTA、FP/FN和速度，不能把ID数量减少等同于准确率提高。新的GPU准确率需在服务器运行后报告。

## TrackID Indoor 对照

同一入口支持 `--dataset trackid-indoor`，沿用 TrackID 的原始视频解码、MOT10到评估GT转换和TrackEval。Indoor与Outdoor分开，权重、FP32/PIL预处理、阈值和缓存参数保持不变。`--limit 0` 选择全部Indoor（当前上传包42条、7534帧），不抽帧或转码。

```bash
python -u -m cli.benchmark_tcei --dataset trackid-indoor --env-file .env \
  --dataset-root /data/trackid3x3-fixed-camera \
  --source-root /root/autodl-tmp/third_party/TCEI \
  --output-dir /outputs/tcei-trackid-indoor \
  --limit 0 --cache-scope sequence --variants baseline off object id dual
python -m cli.benchmark_tcei --dataset trackid-indoor \
  --output-dir /outputs/tcei-trackid-indoor --stage status
```

原版/关闭版先逐帧比较，失败停止校准实验。输出根目录同样提供 `summary.csv/deltas.csv/performance.csv/parity.json`；各变体保留原TrackID的 `fixed_camera/` 结构和逐帧匹配。原视频、原GT和旧实验均不覆盖。缓存不继承SportsMOT的经验；跨序列经验另用新目录和 `--cache-scope stream`。

## 再接回出手者功能

先看跟踪指标，再保持球证据和归因规则不变做功能回归。新旧T号不能直接比较；若改成5秒冷启动，原版也需要相同5秒输入。追踪变化后重算或严格重新映射动作、裁剪和身份特征；球检测和进球时刻可校验复用。随后再扩展 PL-NBA。
