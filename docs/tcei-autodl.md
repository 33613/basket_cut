# AutoDL 上配置 TCEI

本阶段检查环境与原 SportsMOT 权重兼容性；尚未接入项目追踪入口，
没有新的跟踪评测结果。作者来源与后续比较协议见
[TCEI 接入调研](tcei-evaluation-plan.md)。

## 采用哪个环境

优先使用已有 `/root/autodl-tmp/env/motip/bin/python`。作者推荐
Python 3.12、PyTorch 2.4.0、torchvision 0.19.0、CUDA 12.1；这是推荐组合，
不根据它强制重装已经工作的 MOTIP 环境。球检测的 `ball` 环境独立保留。
TCEI 这里没有在线梯度更新，不需要再安装训练框架或更换 SportsMOT 权重。

作者源码单独存放在 `/root/autodl-tmp/third_party/TCEI`，固定提交
`145d1b8431398156f8d9f854430e306fdee39eaa`，不覆盖项目的 `MOTIP/`。
两者都有顶层 `models/configs/utils` 包，因此分别在独立 Python 进程中导入。

## 获取源码和记录环境

```bash
(
set -e
cd /root/autodl-tmp/basket_cut
git switch link_player
(
  source /etc/network_turbo
  git pull --ff-only origin link_player
  mkdir -p /root/autodl-tmp/third_party
  if [ ! -d /root/autodl-tmp/third_party/TCEI ]; then
    git clone https://github.com/1941Zpf/TCEI.git /root/autodl-tmp/third_party/TCEI
  fi
  git -C /root/autodl-tmp/third_party/TCEI fetch origin 145d1b8431398156f8d9f854430e306fdee39eaa
  git -C /root/autodl-tmp/third_party/TCEI checkout --detach 145d1b8431398156f8d9f854430e306fdee39eaa
)
mkdir -p /root/autodl-tmp/tool/manifests/tcei
nvidia-smi
/root/autodl-tmp/env/motip/bin/python -V
/root/autodl-tmp/env/motip/bin/python -m pip freeze \
  > /root/autodl-tmp/tool/manifests/tcei/motip-before.txt
)
```

AutoDL 加速仅放在 GitHub 下载的子 shell 中；pip 不继承这次加速设置。
已有 TCEI 目录如果有修改，先保留修改再处理，不用 `reset --hard`。

## 运行实际 GPU 预检

```bash
(
set -e
cd /root/autodl-tmp/basket_cut
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
/root/autodl-tmp/env/motip/bin/python -u -m cli.tcei_preflight \
  --source-root /root/autodl-tmp/third_party/TCEI \
  --checkpoint /root/autodl-tmp/model/motip/r50_deformable_detr_motip_sportsmot.pth \
  --report /root/autodl-tmp/tool/manifests/tcei/preflight.json
)
```

这一命令会检查：固定源码版本及已跟踪文件、依赖和 `pip check`、
CUDA 矩阵运算、torchvision CUDA NMS、Deformable Attention 的 CUDA 结果
与 PyTorch 参考实现、作者入口导入、原权重 `strict=True` 加载，以及
FP32 DETR/轨迹建模/ID 解码前向。ID 解码检查额外特征输出。
检测模型使用小尺寸合成图像，ID 解码使用合成轨迹；这是兼容性检查，
不测校准逻辑和真实视频准确率。严格加载完整权重前，构造骨干时跳过
多余的 ImageNet 权重下载。上游文件不改动。

成功应输出三条 `PASS`，JSON 的 `status` 为
`passed_environment_and_model_smoke`。失败也保存报告，记录出错原因；
需要先看到这个成功状态，再启动后续视频实验。

## 仅在预检失败时补依赖

若提示缺少 `einops/accelerate/wandb/yaml/scipy/tqdm`，再执行：

```bash
(
set -e
cd /root/autodl-tmp/basket_cut
/root/autodl-tmp/env/motip/bin/python - <<'PY'
from importlib.metadata import version
from pathlib import Path
path = Path('/root/autodl-tmp/tool/manifests/tcei/keep-torch.txt')
path.write_text(''.join(f'{name}=={version(name)}\n' for name in ('torch', 'torchvision', 'numpy')))
print(path.read_text())
PY
/root/autodl-tmp/env/motip/bin/python -m pip install \
  -r requirements-tcei.txt \
  -c /root/autodl-tmp/tool/manifests/tcei/keep-torch.txt \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
)
```

约束保持现有 torch、torchvision 和 numpy 版本，发生依赖冲突时先排查，
不先删约束。补完后重跑同一条预检。

若仅缺少 `MultiScaleDeformableAttention` 或报告该算子 ABI/CUDA 错误，
先检查实际编译工具链：

```bash
nvcc --version
/root/autodl-tmp/env/motip/bin/python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print('torch:', torch.__version__, 'torch CUDA:', torch.version.cuda, 'CUDA_HOME:', CUDA_HOME)
PY
```

编译器 CUDA 与当前 PyTorch CUDA 兼容时，在 RTX 4090 上可使用：

```bash
(
set -e
cd /root/autodl-tmp/third_party/TCEI/models/ops
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9 \
/root/autodl-tmp/env/motip/bin/python -m pip install \
  --no-build-isolation --no-deps --force-reinstall .
)
```

已有 MOTIP 算子通过预检就直接复用，不执行这次重编译。
此命令只重装当前 MOTIP 环境里的算子；不安装另一套 torch。

## 接入后的 SportsMOT 对照

1. 读取之前服务器实验的 `dataset.json`、`experiment_protocol.json` 和
   `summary.csv`，确定同一批篮球 val。完整范围应为 15 序列、12557 帧。
   不通过搜索名字包含 basketball 随意选择帧。
2. 保留同一个 SportsMOT checkpoint，FP32、上游 PIL 预处理、TF32 关闭，
   阈值、图像尺寸、分配协议与原基准一致。
3. 外部适配器关闭校准后，逐帧检查框、分数、ID 与原版一致；再测试对象层、
   ID层、双层。新实验使用新输出目录，不覆盖原指标。
4. 局部缓存每序列清空。ID层缓存的逐序列重置和跨序列保留分开报告；
   固定序列顺序，单 GPU 起步。GT 只供 TrackEval 计算指标。
5. 对比 HOTA、AssA、IDF1、IDSW、Frag、DetA、MOTA 和时间/显存。
   历史参考为 HOTA 76.94、AssA 67.79、IDF1 84.19、IDSW 305，
   必须先核对服务器原始协议，不能把环境通过当作性能提升。

当前 `cli.benchmark_sportsmot` 的 mode 仍为 `official/adapter/parity`，
尚不存在可执行的 `--mode tcei`。作者入口不能保证自动限定我们这批篮球
数据，也有已记录的缓存数值/映射问题；接入时需明示修订和保持公平对照。
