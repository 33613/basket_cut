# 单张 RTX 4090 部署与验收

本说明面向 Linux、24 GB 显存的单张 RTX 4090。GPU 安装命令依据保留的上游项目安装说明整理；尚未在目标机器执行，不应把 CPU 自检当作 GPU 验收。

## 1. 系统与控制环境

准备 NVIDIA 驱动、与编译环境匹配的 CUDA toolkit（包含 `nvcc`）、C/C++ 编译器、Conda 和 ffmpeg。PyTorch 的 CUDA wheel 只带运行库，不能代替 MOTIP 自定义算子编译需要的 toolkit。

```bash
nvidia-smi
nvcc --version
ffmpeg -version
# 在仓库根目录执行；控制环境推荐 Python 3.10 或 3.11。
bash tools/setup_control.sh
```

每个模型使用独立 Conda 环境，避免 MMCV、KPR 和 Transformers 的依赖互相覆盖。阶段串行运行，一次只加载一个模型。不要启动另一个 GPU 批处理或常驻 Qwen 服务与本队列竞争显存。

## 2. MOTIP

依据 [仓库上游安装说明](../MOTIP/docs/INSTALL.md)：

```bash
conda create -n basket-motip python=3.12 -y
conda install -n basket-motip pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda run -n basket-motip python -m pip install 'numpy<2' pyyaml tqdm matplotlib scipy pandas wandb accelerate einops opencv-python-headless
# 确认 CUDA_HOME 指向与 PyTorch 12.1 匹配的 toolkit，nvcc 可用。
# 从仓库根目录进入算子目录；4090 的 compute capability 为 8.9。
cd MOTIP/models/ops
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 conda run -n basket-motip python setup.py build install
conda run -n basket-motip python test.py
cd ../../..
```

从 [MOTIP 模型说明](../MOTIP/docs/MODEL_ZOO.md) 获取适配当前源码的 SportsMOT 权重，保存为 `runtime/models/motip/r50_deformable_detr_motip_sportsmot.pth`。使用上游可信权重，不修改预期校验值或关闭 TLS。

## 3. KPR

从官方仓库安装，目录 `KPR/` 已被 Git 忽略：

```bash
git clone https://github.com/VlSomers/keypoint_promptable_reidentification.git KPR
```

遵循 KPR 检出版本 README 中的 Installation。上游基线是 Python 3.10、PyTorch 1.13、torchvision 0.14、CUDA 11.7；该环境与 MOTIP 和 Qwen 隔离。先检查所选 PyTorch 在机器上能实际执行 CUDA 操作；若需要换用较新 CUDA/PyTorch 组合，必须重新验证 KPR，不能仅凭 import 成功认定兼容。

```bash
conda create -n basket-kpr python=3.10 pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda run -n basket-kpr python -m pip install 'numpy<2' 'Cython<3'
# 安装时保持 numpy<2，避免旧 PyTorch 与 NumPy 2 ABI 不兼容。
mkdir -p runtime
printf 'numpy<2\nCython<3\ntorch==1.13.0\ntorchvision==0.14.0\nmonai<1.4\n' > runtime/kpr-constraints.txt
conda run -n basket-kpr python -m pip install -c runtime/kpr-constraints.txt -r KPR/requirements.txt
cd KPR
conda run -n basket-kpr python setup.py develop
cd ..
```

约束文件保存在忽略的 `runtime/`；不要提交本地安装产物。KPR 的上游依赖未完整锁定，若安装后版本发生冲突，按照其安装说明诊断，并保留可工作的版本清单。

下载项目原有的多数据集权重并验证已知 SHA-256：

```bash
.venv/bin/python -m pip install 'huggingface_hub>=0.25,<1'
.venv/bin/python -m cli.download_kpr --output-dir runtime/models/kpr
```

默认不使用关键点提示。没有外部关键点时，不需要额外部署姿态网络。

## 4. SlowFast / MMAction2

保留当前 MultiSports 配置，必须安装带 CUDA 算子的 **mmcv**，不是 `mmcv-lite`。当前 MMAction2 要求 `mmcv<2.2`、`mmengine<1`；动作检测还需要 MMDetection。

以下是隔离环境的安装起点，需在 GPU 机器验证算子和权重加载：

```bash
conda create -n basket-action python=3.10 -y
conda run -n basket-action python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
conda run -n basket-action python -m pip install 'numpy<2' 'mmengine>=0.7.1,<1' 'mmcv==2.1.0' --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
conda run -n basket-action python -m pip install 'mmdet>=3.1,<3.3' decord einops matplotlib opencv-python-headless Pillow scipy
conda run -n basket-action python -m pip install --no-deps -e MMAction2
```

如果没有匹配的 MMCV wheel，安装会明确失败；根据 [MMAction2 安装说明](../MMAction2/docs/en/get_started/installation.md)选择支持的组合或在匹配 toolkit 下编译，不替换为没有算子的包。

从 [SlowFast MultiSports 模型说明](../MMAction2/configs/detection/slowfast/README.md) 获取相应动作检测权重，保存为 `runtime/models/action/slowfast_multisports.pth`。不是仅有 Kinetics 预训练的动作分类权重。

## 5. 本地 Qwen 视觉模型

第一版适配 `Qwen2.5-VL-3B-Instruct`，以 FP16、SDPA、逐张躯干图方式运行。显存占用和号码准确率需要实机测量，不能从模型名称推断。

```bash
conda create -n basket-qwen python=3.10 -y
conda run -n basket-qwen python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n basket-qwen python -m pip install -r requirements-qwen.txt
conda run -n basket-qwen python -m pip install 'huggingface_hub>=0.25,<1'
```

从官方 [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 下载完整快照到 `runtime/models/qwen/Qwen2.5-VL-3B-Instruct`，包括 safetensors、配置、tokenizer、processor 文件。为复现记录所用 snapshot commit SHA；可以通过 `huggingface_hub.snapshot_download` 的 `revision` 指定该 SHA。

运行时 `local_files_only=True`，不联网下载权重；`trust_remote_code=False`，只加载 safetensors。适配器记录模型文件与提示词摘要并缓存单图读数。若要换成其他 Qwen 架构，需先调整适配器和验证，不能只替换路径。

## 6. 私人运行配置

复制 `.env.example` 到 `.env`，将解释器改为各环境 Python 的**绝对路径**。用 `conda run -n <环境名> python -c 'import sys; print(sys.executable)'` 查看路径，不把私有机器路径提交仓库。

必须配置：

- `BASKET_MOTIP_PYTHON`
- `BASKET_KPR_PYTHON`
- `BASKET_ACTION_PYTHON`
- `BASKET_QWEN_PYTHON`
- `BASKET_QWEN_MODEL_DIR`

权重可以使用默认 `runtime/models` 布局，也可配置 `.env.example` 中的对应变量。`BASKET_REVIEW_PYTHON` 可使用有 OpenCV 的 MOTIP Python。加载 `.env` 只接受 `BASKET_*` 赋值，不执行其中的 shell 命令。

## 7. 先验收一个片段，再追加

```bash
.venv/bin/python -m cli.check_runtime --env-file .env
.venv/bin/python -m cli.process_batch --env-file .env --match-id game-001 \
  --input-dir runtime/data/game-001 --output-dir runtime/outputs/game-001 \
  --limit 1 --dry-run
.venv/bin/python -m cli.process_batch --env-file .env --match-id game-001 \
  --input-dir runtime/data/game-001 --output-dir runtime/outputs/game-001 --limit 1
```

`check_runtime` 检查 CUDA、核心 imports、算子和指定权重路径；它不加载全部模型，最终仍需真实片段运行。确认 `batch_summary.json` 的失败列表为空、各阶段产物存在、原视频与框对齐，并人工核查号码、身份和动作。实际处理失败时 CLI 返回非零状态，细节在该片段的 `pipeline.log`。

第一段通过后，用同一命令增加 `--limit` 或明确给定 selection，检查原有球员 ID 不变、重复运行不产生新档案。未知/遮挡人物可能进入待定队列，不应该为了使数量接近 10 人而放宽阈值。

```bash
.venv/bin/python -m cli.serve_web --env-file .env --host 0.0.0.0 --port 6006
```

通过租赁平台的受控端口访问网页，导入 `runtime/outputs/game-001`。服务没有登录认证，不直接暴露公网。

之后再拿未用于初始化的片段检验：已有球员误匹配、新球员误归并、同人重复建档、号码拒识与误读、轨迹混人和事件归属。备份整个运行目录及 `library/players.sqlite3`，不只备份 JSON。
