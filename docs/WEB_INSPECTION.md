# 在 Mac 上查看服务器的已有实验结果

网页运行在服务器，浏览器通过 SSH 隧道访问。只看已有结果不需要 GPU，
也不要求同时导入三个模型环境。默认界面为亮色。

## 1. 配置轻量 Web 环境

```bash
cd /root/autodl-tmp/project/basket_cut
git pull --ff-only origin main
conda create -p /root/autodl-tmp/envs/basket-web python=3.10 -y
conda activate /root/autodl-tmp/envs/basket-web
python -m pip install -r requirements-web.txt
```

环境已经存在时，跳过 `conda create`。需要转码时安装 CPU 版 FFmpeg：

```bash
conda install -c conda-forge ffmpeg -y
```

## 2. 导入五条已完成结果

```bash
python -m cli.import_web_results \
  --result-dir /root/autodl-tmp/outputs/basket_cut/pl_nba_check_20261004_003331 \
  --name "PL_NBA · 五片段检查" \
  --prepare-media
```

- 支持包含多个片段的结果目录，也支持一个片段的目录。
- 原输入路径读取自各片段 `tracking/video_meta.json` 的 `input_path`。
- 原模型输出不会移动、复制或改写；只在 Web 输出目录创建一个项目索引。
- `--prepare-media` 用 CPU 将原视频、轨迹视频和最终结果转换成浏览器
  兼容的 H.264 / yuv420p / faststart 预览，**预览不保留音频**。
- 预览只保存到 `/root/autodl-tmp/outputs/basket_cut/web/import-*/media/`，
  不改动原文件，也不会再次调用任何模型。它会额外占用磁盘空间。
- 重复导入同一个目录更新同一个实验；视频源文件大小/修改时间没变时，
  已准备好的预览不重复转换。转码失败会报错，不冒充播放成功。
- 也能在网页填写服务器结果目录进行导入。网页导入只建立索引，不自动
  转码；遇到视频编码不支持时用上面的 CLI 准备预览，再刷新。
- 已导入实验默认只读：不能上传视频或启动模型重跑。需要对比新配置时，
  新建实验并上传原视频，不覆盖基线结果。

默认只允许导入 `/root/autodl-tmp/outputs/basket_cut` 内的目录，原视频必须位于
`/root/autodl-tmp/data/basket_cut` 内；导入拒绝逃逸的软链接。若实际数据布局
不同，可在 CLI 和 Web 启动前设置 `BASKET_WEB_IMPORT_ROOT` 和
`BASKET_WEB_SOURCE_ROOT`，不要将边界放宽到整个 `/`。

## 3. 启动服务并从 Mac 访问

服务器执行（前台保持运行；长时运行可放入 screen）：

```bash
python -m uvicorn web.backend.app:app --host 127.0.0.1 --port 6006
```

Mac 的新终端使用 AutoDL **新实例**的 SSH 指令，增加 `-N -L`。
下面三个占位值替换成控制台的实际端口、用户和主机：

```bash
ssh -N -L 16006:127.0.0.1:6006 -p SSH_PORT SSH_USER@SSH_HOST
```

保持两个终端运行，在 Mac 浏览器打开 <http://127.0.0.1:16006>。
VS Code 也可以在 Ports 面板转发远程 6006 端口，然后打开转发后的地址。
服务没有账号认证，**不要直接暴露到公网**；默认仅绑定服务器回环地址。

## 4. 页面上如何检查

- 顶部切换原视频、轨迹视频和最终叠加视频。
- 轨迹检查：原 ID、观测范围、覆盖率和平均检测分数。覆盖率不是准确率。
- 人物档案：KPR 封面、代表图片、轨迹内距离与可见部位；点击人物打开
  对应事件。档案 ID 只在本片段有效，不保证每个档案就是一个独立真人。
- 事件时间区间：读取 **`action/events.jsonl`**，按人物/类型筛选，展示
  `[id, event, start, end, raw_score]` 与支持点数。点击区间跳转到开始时间。
  秒数对应原视频，区间是 `[start, end)`，结束时间不包含。
- 动作预测点：单独检查采样预测和阈值以下候选；候选不会计入事件总数。
- 没有聚合文件与存在空聚合文件分别显示“尚未生成”与“0 个区间”，不会
  用预测点替代。无法解析或 video_id 不一致的事件会给出警告，不隐藏错误。
- 原始数据：事件、轨迹、档案、动作、距离、采样图元信息。
- 导入阶段的状态仅表示对应文件是否存在，不是正确率或原进程退出状态。
  PL_NBA 没有当前任务对应真值时，不显示虚构的 F1 或身份准确率。
- 后续网页上传并运行完整流程时，会在人物事件连接后调用独立的
  `cli.aggregate_events` 适配层；不会把聚合逻辑放入动作基线。

## 5. 怎么把结果交给助手分析

助手不会因为网页在服务器运行，就自动获得服务器文件访问权限。

每条视频可点击“下载诊断包”，把 ZIP 上传到对话。包包含明确清单内的
JSON/JSONL、评估 CSV、日志，以及最多 24 张人物封面；不包含视频、模型
权重和 embedding 张量，原始文件总量超过 64 MiB 时拒绝打包。

需要检查实际框与动作时，点击播放器旁的“下载当前视频”，另行上传
该片段的结果视频、原视频或关键帧。准备过缓存时下载的是 H.264 预览。
结构化结果适合分析轨迹断裂、档案采样与事件碎片；没有真值不能据此
算出准确率。不要把密码、SSH 私钥、Token 或整套 Conda 环境上传。

## 6. CPU-only 自检

测试代码位于 `web/backend/selfcheck.py`，随 Git 提交，不依赖忽略的 `tests/`。

```bash
python -m pip install -r requirements-web-test.txt
python -m cli.check_web
```

使用合成样例验证五条导入、单片段导入、重复导入、路径与软链接边界、
只读重跑保护、事件/预测点分离、非法事件警告、诊断 ZIP、预览缓存、
HTTP 图片/视频读取和 Range 跳转支持。这不等于真实 PL_NBA 效果测试。
