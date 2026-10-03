# AutoDL 克隆/重新租用检查单

## 首选：克隆到另一台可扩容的 4090

克隆以旧实例系统盘为模板创建新实例，并不是给原主机添加 GPU，也不是免费
获得第二台服务器。它省去手动重建环境；新实例按所选计费方式计费。

1. 保存工作、停止任务，原实例关机；不要释放旧实例。
2. 点击 GPU 不足弹窗中的「克隆实例」，或旧实例更多操作中的同名入口。
3. 优先选择同地区的另一台空闲 4090 主机，确认数据盘可扩容。
4. **勾选同时拷贝数据盘**。本项目的代码、Conda 环境、权重、数据、结果都在
   `/root/autodl-tmp`；只克隆系统盘不会复制这些文件。
5. 确认目标盘足够容纳当前实际使用量和后续数据；检查费用后创建。
6. 等待系统盘和数据盘都拷贝完成；更新 Mac SSH 配置中的新地址/端口。
7. 核对目录、环境、GPU与文件数量，单视频成功后再跑固定批量清单。

忘记复制数据盘时，可通过旧实例「更多操作 -> 跨实例拷贝数据」补拷贝。核对
源/目标，避免覆盖有价值的目标数据。系统盘保存镜像不包含数据盘，不可当作
`/root/autodl-tmp/envs` 和权重的备份。

```bash
nvidia-smi
df -h /root/autodl-tmp
ls /root/autodl-tmp
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motip
cd /root/autodl-tmp/project/basket_cut
git pull --ff-only
python -m cli.check_evaluation --with-tracking
```

该自检不使用 GPU；还须用一个真实视频检查 MOTIP、KPR、MMAction2 的推理。
不能只用 `nvidia-smi` 或 `torch.cuda.is_available()` 代替模型兼容性测试。

## 重新租空白实例

1. 租用新实例 -> 选择地区、GPU型号、1卡、实际CPU/内存和磁盘规格。
2. 4090优先复用已验证的软件环境；5090需支持Blackwell的PyTorch/CUDA构建。
   PyTorch 2.7提供CUDA 12.8/Blackwell支持，但第三方算子仍需要重编译、验证。
3. 创建时确认可扩容盘，先按实际数据大小规划容量，而不是按视频条数推算。
4. 用Git获取代码，用平台跨实例拷贝/文件传输搬运数据盘；确认复制完成后
   再启动程序。Git不保存权重、数据、结果或Conda环境。
5. 若迁移到5090，不覆盖新系统盘去恢复旧PyTorch；保留模型和数据，重建兼容
   的模块环境。更快硬件不代表更低迁移成本。
6. 单视频推理 -> 正确性检查 -> 整条链路计时 -> 决定是否运行50条。

需要频繁开关机且保证GPU时，考虑包日/包周；按量实例关机后GPU不预留。
付费磁盘即使关机仍可能收费。旧实例连续关机后的释放期限以控制台提示为准，
保留期间也应另行备份重要数据，不能无限期当作备份盘。

## 官方依据

- [同地区迁移/克隆与数据盘复制](https://www.autodl.com/docs/migrate_instance_2/)
- [计费、GPU预留和磁盘费用](https://www.autodl.com/docs/price/)
- [系统盘/数据盘与存储](https://www.autodl.com/docs/instance_pro/)
- [PyTorch 2.7 Blackwell支持](https://pytorch.org/blog/pytorch-2-7/)
