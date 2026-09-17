# 工业缺陷检测系统（YOLOv8 + ONNXRuntime）

基于 **NEU-DET** 钢材表面缺陷数据集的端到端缺陷检测系统：
**数据转换 → 弱类增强 → 训练 → 导出 ONNX → FastAPI 推理服务（含实时 MJPEG 推流）→ Gradio 演示**。

## 技术链路

```
NEU-DET(Pascal VOC XML)
   │  src/data/convert.py      转 YOLO 标签
   │  src/data/split.py        切 train/val/test
   │  src/data/augment.py      弱类定向离线增强
   ▼
YOLOv8n 训练 ──► runs/detect/runs/<exp>/weights/best.pt
   │  scripts/export.py        固定尺寸导出
   ▼
weights/best.onnx
   │  src/inference/onnx_predictor.py   ONNXRuntime（不依赖 torch）
   ▼
FastAPI 服务 (src/inference/api.py)  +  MJPEG 实时推流 (src/inference/stream.py)
   └─► Gradio 演示 (src/ui/app.py)
```

## 项目结构

```
defect-detection/
├── configs/            data.yaml（数据集）+ hyp.yaml（超参）
├── data/
│   ├── raw/            原始 NEU-DET（不入库）
│   └── processed/      YOLO 格式 images/labels/{train,val,test}（不入库）
├── src/
│   ├── constants.py    类别表唯一真源（顺序即 class_id）
│   ├── data/           convert / split / augment
│   ├── evaluation/     metrics（整体+逐类评估）/ error_analysis（难例分析）
│   ├── inference/      api（HTTP）/ stream（流式）/ predictor(Torch) / onnx_predictor / schemas
│   ├── models/         wrapper
│   ├── training/       train / callbacks
│   ├── ui/             Gradio 演示界面
│   └── utils/          paths（路径唯一真源）/ imageio（中文路径安全读写+画框）/ io / logger
├── scripts/            CLI 入口（薄壳，共用 scripts/_bootstrap.py 定位项目根）
├── tests/              pytest 单元测试（不加载模型、不联网）
├── weights/            best.onnx
├── runs/               训练日志 / 曲线 / 评估图（不入库）
├── docs/               审计报告等工作文档
├── requirements.txt        开发机全量依赖（含 torch）
├── requirements-api.txt    仅推理服务依赖（部署/Docker 用）
└── Dockerfile              推理服务镜像
```

## 快速开始

> **权重不入库**：`git clone` 后仓库里没有模型，`/health` 会返回
> `model_missing`、`/detect` 返回 503。按下面第 1~4 步生成即可；
> 详情见 `weights/README.md`。

```bash
conda create -n defect python=3.11 && conda activate defect
pip install -r requirements.txt

# 0. 预训练初始权重 yolov8n.pt（6.2MB，同样不入库）
#    scripts/train.py 的 --weights 默认值是 <项目根>/yolov8n.pt。
#    缺它时 ultralytics 会去 GitHub 下载，而国内网络常失败（SSL 吊销检查超时），
#    建议先手动放到项目根目录：
#      curl -L -o yolov8n.pt https://hf-mirror.com/ultralytics/yolov8n/resolve/main/yolov8n.pt

# 1. 数据准备：先把 NEU-DET 解压到 data/raw/NEU-DET/，再执行
python scripts/prepare_data.py            # 转换 + 切分 train/val/test
python scripts/augment.py                 # 可选：弱类定向离线增强

# 2. 训练（产物落在 runs/detect/runs/<name>/weights/best.pt）
python scripts/train.py --name exp

# 3. 评估（默认在独立 test 集上评估 paths.BEST_PT）
python scripts/evaluate.py
python scripts/evaluate.py --weights runs/detect/runs/exp/weights/best.pt --split test

# 4. 导出 ONNX 到 weights/best.onnx
python scripts/export.py --weights runs/detect/runs/exp/weights/best.pt

# 5. 启动推理服务（默认只绑 127.0.0.1）
python scripts/serve.py

# 6. 演示界面
python scripts/demo.py                    # http://127.0.0.1:7860
```

> **路径说明**：项目位于中文目录，路径统一由 `src/utils/paths.py` 解析
> （环境变量 `DEFECT_PROJECT_ROOT` > 由文件位置推导 > 本机已知路径，
> 每个候选都会校验是否真的含 `configs/data.yaml`）。
> 迁移到别的机器时只需设 `DEFECT_PROJECT_ROOT`。

## 推理服务接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 极简可视化页（`<img src="/stream">` 直接看实时画面） |
| GET | `/health` | 服务与模型状态（后端 provider、输入尺寸、类别表） |
| POST | `/detect` | 上传图片检测；query 可覆盖 `conf` / `iou` |
| GET | `/stream` | MJPEG 实时推流（连接即启动，断开自动释放） |
| GET | `/stream/state` | 轮询最新稳定状态 |
| POST | `/stream/stop` | 显式停止当前推流 |

```bash
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/detect?conf=0.25&iou=0.5" -F file=@data/processed/images/test/crazing_1.jpg

# 用 test 集图片做合成流（无摄像头也能验证推流）
# /stream?source=synthetic:<目录> ；本地路径限制在 ALLOWED_SOURCE_ROOTS 内
```

**安全边界**：`/stream?source=` 的本地路径被限制在 `utils.paths.ALLOWED_SOURCE_ROOTS`
（默认项目目录）内，防止任意文件读取；需要放行别的目录时设
`DEFECT_ALLOWED_SOURCE_ROOTS`（多个用 `os.pathsep` 分隔）。
RTSP/HTTP/RTMP 网络流默认放行（摄像头接入是既定用法），部署在不可信网络请自行收紧或加鉴权。

## 数据集

- 来源：**NEU-DET**（东北大学，公开学术数据集，6 类钢材表面缺陷，200×200 灰度图）
- 类别与 class_id：`crazing(0) / inclusion(1) / patches(2) / pitted_surface(3) / rolled-in_scale(4) / scratches(5)`
- 规模：train 1439 / val 181 / test 180（弱类增强后 train 2877）
- **本项目使用公开数据集训练，非自建产线数据**；面试请如实说明。

## 评估结果

在**独立 held-out test 集**（训练与调参全程未使用）上，ONNXRuntime 与
PyTorch 两条链路结果一致（`python scripts/verify_onnx.py` 校验）：

| 指标 | 值 |
| --- | --- |
| mAP@0.5 | 0.787 |
| mAP@0.5:0.95 | 0.410 |
| Precision / Recall（conf=0.25 工作点） | 0.858 / 0.592 |

数值以 `python scripts/evaluate.py` 的实际输出为准（会打印整体 + 逐类表）。

**已知短板（如实记录）**：

- `crazing`（裂纹）是最难类——低对比纹理、无明确边界，是 NEU-DET 公认难点，逐类 AP 明显低于其他类。
- 弱类定向增强 + 降分辨率到 320 的消融实验中，`scratches` 的 mAP@0.5 提升约 +0.10，
  但 `crazing` / `rolled-in_scale` 未改善，`pitted_surface` 的 precision 反而下降。
  该实验同时改动了「增强」与「分辨率」两个变量，**无法归因**，不能作为结论性证据。
- 模型存在误检（FP）多于漏检（FN）的倾向，见 `src/evaluation/error_analysis.py` 的输出。

## 工程约定

- **路径唯一真源**：`src/utils/paths.py`。代码中不写死绝对路径。当前工作目录只由
  `scripts/_bootstrap.py` 在 CLI 入口处设置一次，库代码不 `chdir`。
- **类别表唯一真源**：`src/constants.py`（顺序即 class_id，写错会静默错标）。
- **图像读写唯一实现**：`src/utils/imageio.py`。
  Windows 中文路径下 `cv2.imread` / `cv2.imwrite` / `np.tofile` 会**静默失败**，
  统一走 `np.fromfile + cv2.imdecode` 读、`cv2.imencode + Path.write_bytes` 写。
- **依赖方向**：`scripts → 各层模块 → utils`，下层不反向依赖上层。
- **测试**：`pytest` 只收集 `tests/`（见 `pytest.ini`）。`scripts/` 下的
  `test_*.py` 是需要先启动服务的端到端脚本，不参与单测收集。

## 测试与验证

```bash
pytest -q                                    # 单元测试（不加载模型/不联网）

# 端到端验证（需先起服务 / 需已导出 ONNX）
python scripts/test_api.py                   # /health 与 /detect
python scripts/test_stream_api.py            # /stream 系列（真实 uvicorn + socket）
python scripts/test_ui.py                    # Gradio 推理/绘图逻辑（无头）
python scripts/verify_onnx.py                # ONNX 与 PyTorch 结果一致性
python scripts/verify_stream.py              # 限频 / 去抖 / 生命周期断言
```

## Docker

```bash
python scripts/export.py                          # 先确保 weights/best.onnx 存在
docker build -t defect-detection .
docker run --rm -p 8000:8000 defect-detection
```

- 只装 `requirements-api.txt`（onnxruntime + fastapi + opencv-**headless**，**不含 torch**），
  镜像约 600MB，而不是把训练依赖一起打进去的 3GB+。
- 用 `opencv-python-headless` 避开 slim 镜像缺 `libGL` 导致 `import cv2` 崩溃的问题。
- 非 root 用户运行 + `HEALTHCHECK` 打 `/health`。
- 因此 `.pt` 回退分支在容器内不可用，**必须**提供 ONNX 权重。

## 说明

`docs/audit-report-2026-09-17.md` 记录了本项目的一次完整自检
（代码质量 / 潜在缺陷 / 依赖与配置 / 结构合理性），以及每条问题的修复状态。
