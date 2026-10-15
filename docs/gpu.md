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

## 5. 本地 Qwen3-VL 视觉模型

统一使用 `Qwen3-VL-8B-Instruct` 处理号码图片和有帧数限制的视频窗口。CUDA上优先BF16，不支持时FP16，使用SDPA；视频输入不等同于已经实现篮球回合切分。

现有Qwen2.5环境可以先克隆到新前缀，再安装 `requirements-qwen.txt`，复用torch2.6.0及torchvision0.21.0。清理过Conda包缓存后，克隆可能需要联网补齐基础包，不应强制离线；用新前缀的 `bin/python` 是否可执行来判断创建成功，不能只判断目录存在。新的独立环境采用transformers4.57.1，不能只改模型路径继续用旧加载器。若从空环境安装，PyTorch CUDA轮子单独选用原有兼容源，再安装其余依赖。

```bash
conda create -n basket-qwen3 python=3.10 -y
conda run -n basket-qwen3 python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n basket-qwen3 python -m pip install -r requirements-qwen.txt
```

视频由官方processor读取原始本地窗口并保留解码器提供的FPS/采样帧索引。该固定版本优先torchcodec（若安装），否则使用torchvision/PyAV；推荐新环境不安装torchcodec，使用与torch2.6匹配的torchvision0.21及PyAV14.2.0，避免额外FFmpeg共享库/ABI配置。依赖文件将PyAV固定为14.2.0并只允许其二进制wheel；14.3/14.4缺少Python3.10 Linux wheel，宽泛的14.x版本范围会触发源码编译并要求FFmpeg开发库。若镜像缺少14.2.0的wheel，应从官方PyPI补齐，不退回源码编译。不要传已经被该版本忽略的 `video_load_backend` 参数来声称强制切换了解码器。

通过ModelScope下载官方完整快照，使用本地目标目录，避免再复制一份模型缓存。下载时冻结远端文件清单和官方SHA256到模型目录之外的manifest，检查全部选中模型文件、权重分片索引和本地图片/视频processor。相同命令可以利用ModelScope下载断点；若远端master已变更则停止，不静默混用文件。可显式指定官方tag。

```bash
conda run -n basket-qwen3 python -u -m cli.download_qwen \
  --output-dir runtime/models/qwen/Qwen3-VL-8B-Instruct \
  --manifest runtime/manifests/qwen3-modelscope.json
```

运行时仅从本地加载safetensors，`local_files_only=True`、`trust_remote_code=False`。图片视觉预算256–512 tokens；视频默认总视觉预算上限约2048 tokens，并限制显式抽样帧数为4–64之间的偶数。视频烟测用8帧，不代表长视频任务的时间定位或回合准确率。

GPU自检会先查询其他GPU进程，忙时停止，让已有MOTIP等任务先结束；下载、模型哈希和CPU processor检查不需要占用GPU。图片与视频共用同一个模型实例，逐次推理。

```bash
conda run -n basket-qwen3 python -m cli.smoke_qwen \
  --model-dir runtime/models/qwen/Qwen3-VL-8B-Instruct \
  --output-dir runtime/checks/qwen3-smoke-001 \
  --with-video
```

该检查验证质量抽样、号码JSON、缓存及视频输入/输出，不评估识别准确率。只有命令成功结束，且 `qwen-smoke.json` 的status为completed_not_accuracy_evaluated、架构为qwen3_vl、video_model_invoked为true，才完成本次功能验收。

成功后更新私有env的 `BASKET_QWEN_PYTHON` 和 `BASKET_QWEN_MODEL_DIR`，备份旧env；之后可以删除旧Qwen2.5模型目录及旧环境，保留历史输出、裁剪图和球员库。新模型的证据合同和缓存摘要会变化，后续真实处理使用新的运行目录，避免把旧模型号码证据混入新实验。

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
