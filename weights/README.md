# weights/

本目录**不入库**（见 `.gitignore`），只存放本地产出的模型文件：

| 文件 | 大小 | 来源 |
| --- | --- | --- |
| `best.onnx` | ~9.3 MB | `python scripts/export.py` 导出，服务端主链路（当前为 YOLO26n） |
| `best.pt` | ~5.2 MB | 训练产出，默认不出现在这里，在 `runs/exp_yolo26/weights/` |

## 为什么权重不入库

模型二进制会随每次实验变化，进 git 会让仓库迅速膨胀且产生无法 review 的 diff。
正确做法是「代码 + 数据 + 超参可复现」，权重作为产物单独发布。

## 如何重新得到权重（M10：新克隆的仓库不是开箱可用的）

```bash
# 1. 准备数据：把 NEU-DET 解压到 data/raw/NEU-DET/
python scripts/prepare_data.py

# 2. 训练（默认 imgsz=640，产物落在 runs/<name>/weights/best.pt）
python scripts/train.py --name exp_yolo26

# 3. 导出 ONNX 到本目录
python scripts/export.py                    # 默认取 runs/exp_yolo26/.../best.pt
python scripts/export.py --weights runs/exp_yolo26/weights/best.pt
```

在此之前，`/health` 会返回 `status: model_missing`，`/detect` 返回 503 —— 这是
预期行为（fail loudly，而不是静默给出错误结果）。

## 换机器 / 交付

把 `best.onnx` 拷到本目录即可，无需改代码：路径由
`src/utils/paths.ONNX_PATH` 统一解析，也可用环境变量
`DEFECT_PROJECT_ROOT` 指向项目根。
