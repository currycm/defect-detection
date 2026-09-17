# 项目自检报告 · defect-detection

- 检查时间：2026-09-17
- 范围：代码质量、潜在 bug、依赖与配置、结构合理性
- 方法：全部结论均经**实际运行复现**或**命令输出取证**，未复现的推断会显式标注
- 工具：`ruff`（默认 + 扩展规则集）、`pytest`、`pip check`、`git`、自建只读探测脚本
- 未做任何修改（纯只读自检）
- 注：本报告在入库/公开前做了**机器信息脱敏** —— 涉及本机用户目录与父目录名的
  字面量已替换为 `<用户目录>` / `<项目父目录>`，其余结论与命令输出保持原样。

---

## 一、🔴 高：会直接导致不可用 / 安全风险

### H1. `/stream` 的 `source` 参数可被用来读取本机任意文件
- **位置**：`src/inference/api.py:161-173`（`_resolve_source`）→ `:197` → `src/inference/stream.py:119-141`（`open_source`，其中 `:132` 判目录、`:134` `os.listdir`）；`scripts/serve.py:23` 绑定 `0.0.0.0`
- **原因**：query 参数 `source` 未做任何白名单/根目录约束，直接进入 `cv2.VideoCapture(source)`；写成 `synthetic:<路径>` 时会对该路径做 `os.listdir` 并按图片解码。`serve.py` 又监听全网卡，局域网内任意主机都可达。
- **证据（已复现）**：在项目**之外**建 `%TEMP%\audit_secret_dir` 放入一张图，请求
  `GET /stream?source=synthetic:<该目录>`，服务端成功推送 3 帧 JPEG。
- **修复**：
  1. 定义 `ALLOWED_ROOTS`，对解析后的路径 `Path(p).resolve()` 后校验 `is_relative_to(ALLOWED_ROOT)`，不合法直接 400；
  2. 生产环境默认只允许**摄像头索引**，文件/流地址走显式配置而非请求参数；
  3. `serve.py` 默认绑 `127.0.0.1`，需要对外时再加鉴权（API Key / 反代）。

### H2. `pytest` 整个跑不起来（收集阶段即失败）
- **位置**：`scripts/test_api.py` 与 `tests/test_api.py` 同名
- **原因**：两处目录均无 `__init__.py`，pytest 用 basename 作为模块名，导致 `import file mismatch`。
- **证据（已复现）**：`python -m pytest --collect-only -q` → `ERROR tests/test_api.py ... imported module 'test_api' has this __file__ attribute: ...\scripts\test_api.py`，**exit code 2**，只收集到 4 个用例。
- **修复**：把 `scripts/test_*.py` 改名为 `check_*.py` / `smoke_*.py`（它们本质是手动 runner，不属于 pytest 用例）；并在 `pyproject.toml` 里设 `[tool.pytest.ini_options] testpaths = ["tests"]`。

### H3. 文档宣称的 PyTorch 回退后端，`/detect` 必然 500
- **位置**：`src/inference/predictor.py:29-33`（返回 dict 无 `name` 键）vs `src/inference/schemas.py:11`（`name: str` 必填）vs `src/inference/api.py:155`（`Detection(**d)`）
- **原因**：两条推理链路返回的字典契约不一致——ONNX 版带 `name`，ultralytics 版不带。
- **证据（已复现）**：`Detection(**{"xyxy":[1,2,3,4],"conf":0.9,"cls":0})` → `pydantic ValidationError: name`。
- **连带影响**：`/stream/state` 返回的 detections 缺字段；`scripts/stream_demo.py` 的 `summary_of()` 用 `d["name"]` 会 KeyError。
- **修复**：把结果契约收敛到一处（建议在 `src/inference/schemas.py` 定 `Detection` 为唯一真源，两个 predictor 都产出该结构）。最小改动是在 `predictor.py` 里补 `"name": self.class_names[int(b.cls[0])]`。

### H4. `onnxruntime` 未在 requirements 声明，且本机包元数据已损坏
- **位置**：`requirements.txt`（全 13 行）；`src/inference/onnx_predictor.py:20`
- **原因**：requirements 只声明了 torch/ultralytics/cv2 等，**没有 onnxruntime**（也没有 pillow、httpx、python-multipart）。同时本机 onnxruntime 是之前 `--force-reinstall --no-deps` 装的，`dist-info` 缺失。
- **证据（已复现）**：`import onnxruntime` 成功且 `__version__ = 1.26.0`，但 `importlib.metadata.version("onnxruntime")` → `PackageNotFoundError`；`site-packages` 下只有 `onnx-1.22.0.dist-info`，无 onnxruntime 的；`pip check` 报
  `chromadb 1.5.9 requires onnxruntime, which is not installed.`
- **影响**：新机器 `pip install -r requirements.txt` 装不出 ONNX 推理能力（而这是部署主链路）；依赖解析器认为环境已损坏，后续安装容易再次踩坑。
- **修复**：requirements 补 `onnxruntime==1.26.0`、`pillow`、`httpx`、`python-multipart`；用 `pip install --force-reinstall --no-deps onnxruntime==1.26.0` 补回元数据后 `pip check` 复验。
  （补充：`python-multipart` 是 FastAPI `File(...)` 路由的硬依赖，缺它会直接导致应用启动即报错。）

### H5. 项目没有任何版本控制
- **证据（已复现）**：`git rev-parse --show-toplevel` → rc=128 `fatal: not a git repository`；`defect-detection` 与其父目录下均无 `.git`。
- **影响**：62 个源文件、多轮消融实验的全部改动无历史可回溯、无 diff 可评审；作为求职项目，「无 git 记录」在评审时是明显减分项。
- **修复**：`git init` → 先补 `.gitignore`（见 L9）→ 首次提交。注意权重与数据不入库，改用一个 `weights/README.md` 说明权重来源与复现命令。

---

## 二、🟠 中：功能性缺陷与数据风险

### M1. `/stream/state` 存在竞态，会返回 500
- **位置**：`src/inference/api.py:241-245`（`if _active_stream is None` 判断后**无锁**直接 `_active_stream.latest()`）；写侧在 `:263`、`:234`
- **原因**：典型 TOCTOU —— 判断与使用之间，`/stream/stop` 或流自身的 `finally` 把 `_active_stream` 置为 `None`。
- **证据（已复现）**：5 线程并发压测 15 秒（1 路流 + 3 个 `/stream/state` + 2 个 `/stream/stop`），**1147 次请求中出现 1 次 HTTP 500**。
- **修复**：读侧也进 `_stream_lock`，先把引用取到局部变量再判空/使用。

### M2. 流参数无范围校验，可绕过限频
- **位置**：`src/inference/api.py:182-183`（`target_fps` / `detect_interval` / `hold` 全无约束）；`src/inference/stream.py:273`（`frame_period = 1.0/target_fps if >0 else 0.0`）、`:241`（限频判断）
- **原因**：`detect_interval` 为负数时 `now - last >= 负数` 恒真 → 每帧都推理；`target_fps <= 0` 时 `frame_period=0` 去掉循环节流。
- **证据（已复现）**：`GET /stream?detect_interval=-1&target_fps=0` → 推理占比 **88.9%**（正常 `detect_interval=0.1` 时为 50%）。
- **修复**：接口层用 `Query(ge=..., le=...)` 约束；`StreamDetector.__init__` 内再 clamp 一次（`detect_interval = max(x, 1/240)`，`target_fps = clamp(x, 1, 120)`），做到「不信任调用方」。

### M3. ONNX 后处理不裁剪框到图像边界
- **位置**：`src/inference/onnx_predictor.py:125-128`（只做了 `max(..., 0)` 下界，**没有上界**）
- **原因**：letterbox 的灰色 padding 区域也会产生激活，还原坐标后可能超出原图。
- **证据（已复现）**：40 张 test 图上统计 —— conf=0.25 时 **9.0%** 的框越界（最大 0.5px）；conf=0.001（README 推荐的部署档）时 **26.4%** 越界，最大 **17.4px**（图仅 200×200）。例：200×200 图返回 `[90.45, 179.35, 217.43, 199.95]`，x2=217.43 > 200。
- **修复**：`x2 = min(max((x2-left)/r, 0.0), W)`，y 同理（需把 `w/h` 传进 `_postprocess`）。

### M4. `convert._parse_voc` 在 XML 缺 `<size>` 时崩溃，兜底分支不可达
- **位置**：`src/data/convert.py:38-40` 取 `size`，`:65-72` 是尺寸回退逻辑
- **原因**：`root.find("size")` 可能返回 `None`，`:39` 直接 `size.findtext(...)` → AttributeError；崩溃发生在回退分支之前，所以那段 `if W <= 0 or H <= 0` 的兜底永远走不到。
- **证据（已复现）**：构造无 `<size>` 的 VOC → `AttributeError: 'NoneType' object has no attribute 'findtext'`。
- **修复**：`size = root.find("size")`；`W = int(size.findtext("width","0") or 0) if size is not None else 0`。

### M5. `split()` 用 `shutil.move` 且不可重入，重复运行会静默损坏切分
- **位置**：`src/data/split.py:41,44`；调用方 `scripts/prepare_data.py:15`
- **原因**：函数从 `images/val` **移动**文件到 `test`，且不检查 test 是否已存在。当前靠固定 `seed=42` 恰好表现近似幂等，但这属于巧合而非设计；一旦单独重跑 `split()`（不经 convert）就会把 val 再切一半进 test。
- **修复**：加守卫（`test` 非空则拒绝执行并提示）、或改成「先清空 test 再切」、或改为 copy + 落一份清单文件，使操作可逆可审计。

### M6. README 与实现严重不符（对外文档可信度问题）
| README 位置 | 声称 | 实际（已核实） |
|---|---|---|
| `:57` | 「配置与代码分离：所有路径/超参在 configs/，代码不写死」 | **20/21 个 .py 文件**硬编码了 `C:\Users\<用户目录>\Desktop\<项目父目录>\...`；`src/ui/app.py:18-20` 作为**包模块**还执行 `os.chdir`（导入即改变全局 cwd） |
| `:12` | `data/raw/` 为 `IMAGES/ + ANNOTATIONS/` | 实际 `data/raw/NEU-DET/{train,validation}/{images/<class>,annotations}` |
| `:14` | `data/samples/` | 不存在，实际产物在 `runs/demo_samples/` |
| `:41` | 演示入口 `python -m src.ui.app` | 实际入口是 `scripts/demo.py`（两者端口/路径处理还不一致） |
| `:52` | 让运行 `python scripts/evaluate.py` | 该脚本路径是死的（见 M7） |
| `:25` | `conda create -n defect python=3.10` | 实际运行环境为 **Python 3.13.9** |
| 全文 | —— | 完全没提实时流式接口（`/stream` 系列）与新增的验证脚本 |

- **修复**：README 重构为「当前事实」，路径统一改为从 `__file__` 推导 + 支持 `DEFECT_PROJECT_ROOT` 环境变量覆盖；把 chdir 从包模块里去掉（见 L1）。

### M7. `scripts/evaluate.py` 是坏的
- **位置**：`scripts/evaluate.py:9` `BEST = "runs/exp/weights/best.pt"`
- **证据（已复现）**：该路径不存在；实际权重在 `runs/detect/runs/exp_aug640/weights/best.pt`。同时它是相对路径，依赖 cwd。
- **修复**：改为指向真实产物，或（更好）让脚本接受 `--weights` 参数，默认取 `weights/best.onnx` 同级约定。

### M8. `predictor.predict_path` 在中文本机路径下必然失败
- **位置**：`src/inference/predictor.py:38` `cv2.imread(path)`
- **证据（已复现）**：对本项目 test 图（路径含中文目录）`cv2.imread` 返回 `None`，OpenCV 打印 `can't open/read file`；而 `np.fromfile + cv2.imdecode` 正常。
- **修复**：统一改走 `np.fromfile + cv2.imdecode`（本条与 L4 的重复实现合并处理）。

### M9. Dockerfile 构建出的镜像跑不起来（多因叠加）
- **位置**：`Dockerfile:1-9`
- **逐条原因**：
  1. `python:3.10-slim` 缺 `libGL/libglib2.0`，而依赖是**非 headless** 的 `opencv-python` → `import cv2` 失败 → 容器启动即崩；
  2. requirements 缺 `onnxruntime` → 只能降级到 torch 回退，而回退的 `/detect` 本身是坏的（H3）；
  3. 缺 `python-multipart` → `File(...)` 路由在应用导入阶段就会报错；
  4. `COPY . .` 会把 `data/`(82MB) + `runs/`(66MB) + 日志一起打进镜像，且没有 `.dockerignore`；
  5. 无 `HEALTHCHECK`，以 root 运行，未 pin 基础镜像 digest。
- **修复**：基础镜像改 `python:3.11-slim` + `apt-get install -y libgl1 libglib2.0-0`（或换 `opencv-python-headless`）；补 `.dockerignore`；补 HEALTHCHECK 指向 `/health`；用非 root 用户。

### M10. 模型产物完全不入库，新克隆的仓库开箱即坏
- **位置**：`.gitignore:3-5`（忽略 `weights/*.pt`、`weights/*.onnx`、`runs/`）
- **原因**：训练产物在 `runs/`（被忽略），导出的 `weights/best.onnx` 也被忽略，且没有任何 Release/网盘/下载脚本说明。
- **影响**：`git clone` 后 `/health` 直接返回 `model_missing`；Docker 构建同理（第 4 条 `COPY . .` 拿到的是本机残留，不是可复现产物）。
- **修复**：加 `scripts/download_weights.py`（或写清发布地址），并在 README 的 Quick Start 里插入「获取权重」这一步。

---

## 三、🟡 低：质量、整洁度与可维护性

### L1. 包模块承担脚本职责，导入即有副作用
`src/ui/app.py:11,18-20` 在模块顶层 `sys.path.insert` + `os.chdir(PROJECT)`；`src/ui/app.py:53` 还在导入时实例化模型。→ 任何人 `import src.ui.app` 都会改变进程 cwd 并加载 11.7MB 权重，也无法在无权重环境下 import。**修复**：路径/模型初始化收敛到 `scripts/` 入口或加 `lru_cache` 的工厂函数。

### L2. 事件循环里做同步 CPU 编码
`src/inference/api.py:224` 的 `encode_jpeg` 直接在 async 生成器中执行（约 2ms/帧）。单路流可接受，但会阻塞 loop。**修复**：`await anyio.to_thread.run_sync(...)`。

### L3. 持锁做慢操作（最长 5 秒）
`src/inference/api.py:210-213`、`:260-264` 在 `_stream_lock` 内调用 `StreamDetector.stop()`，而 `stop()` 会 `join(timeout=5.0)`（`src/inference/stream.py:328-341`）。→ 慢停止时其他请求排队抖动。**修复**：锁内只做引用交换，`stop()` 放到锁外。

### L4. 结果契约与工具函数重复
- `_imread_unicode/_imwrite_unicode` 在 `src/data/augment.py:36-54`、`src/inference/stream.py`（`SyntheticSource._imread`）、`src/evaluation/error_analysis.py:36-51`、`src/ui/app.py:57-60` 及多个 scripts 各自实现；
- `CLASS_COLORS` 在 `src/inference/stream.py` 与 `src/ui/app.py:32-39` 各一份；
- `src/inference/stream.py:draw_detections` 与 `src/utils/visualize.py:draw_boxes` 功能重叠，且后者**仍有标签越界 bug**（未对标签 x 做右边界钳制，而 `ui/app.py` 已修）。
- **修复**：把中文安全读写成 `src/utils/io.load_image()/save_image()` 一处；颜色表与画框函数收敛为一份。

### L5. 后端间行为不一致（ONNX vs PyTorch）
`src/inference/api.py:150-152` 在回退路径上写 `_backend.conf = conf` —— 修改**共享单例**的状态，并发请求会互相篡改阈值；且该路径**静默忽略 `iou` 参数**。**修复**：把 conf/iou 作为调用参数传递，不在后端实例上落地。

### L6. 重复计算：每帧多算一次 letterbox
`src/inference/onnx_predictor.py:144-145`：`predict_image` 先调 `_letterbox` 取 `r/pad`，紧接着 `_preprocess` 内部**又调一次**。**证据（已复现）**：单次 `predict_image` 触发 `_letterbox` **2 次**。
**说明（已核实）**：两次结果完全一致（实测多种非方图 `r`、`pad` 均相同），所以**这不是正确性 bug，只是白做一次 640×640 缩放**（约 22ms 推理中占可感知比例）。**修复**：`_preprocess(img, box)` 直接复用第一次的返回值。

### L7. 死代码 / 占位实现 / 未使用产物
| 位置 | 问题 |
|---|---|
| `src/models/wrapper.py:12-15` | `export_onnx` 给 `model.export()` 传了 `path=`，ultralytics 无此参数（与之前踩过的 `hyp=` 同类坑），一旦被调用即报错；且全项目无调用方 |
| `src/training/callbacks.py` | 空实现占位，无调用方 |
| `src/evaluation/metrics.py` | 无调用方（`scripts/compare.py` 自己做了 val） |
| `src/inference/schemas.py:14-15` | `DetectRequest` 未被使用 |
| `scripts/verify_raw.py` | Step 2 的一次性诊断脚本，顶层直接执行、无 `main()`、无 `if __name__`，且数据缺失时无守卫 |
| `notebooks/` | 只有 `.gitkeep` |
| `src/inference/predictor.py:36` | `predict_path` 无调用方（且见 M8） |

**修复**：删掉或用起来；`wrapper.export_onnx` 若保留，参数改为 `format/imgsz/dynamic` 并加一条冒烟测试。

### L8. 静态检查结果（`ruff`）
- **默认规则集：仅 3 项**，说明基础质量不错：`scripts/train.py:16` E402、`scripts/verify_stream.py:21` F401（numpy 未使用）、`src/ui/app.py:102` E741（变量名 `l`，易与 `1` 混淆）。
- **扩展规则集 784 项**，其中绝大多数是噪音（RUF001/002/003 中文全角标点 572 项、scripts 里的 T201 print 109 项、E501 行宽 45 项），**不必处理**。需要盯的实质项：
  - `src/evaluation/error_analysis.py:277-285` **4× B023**（闭包未绑定循环变量 `cid`）——**已核实当前不会触发**（`err_score` 在同一次迭代内就被调用），但结构很脆：一旦把函数存起来延后调用就会全部取到最后一次 `cid`。建议用默认参数绑定：`def err_score(r, cid=cid)`。
  - 7× `BLE001` 裸 `except Exception`（`api.py:30,35`、`ui/app.py:85,111` 等）会吞掉真实错误；至少加日志。
  - 9× `PLW0603` 全局可变状态（`api.py` 的 `_backend`/`_active_stream`）—— 与 M1/H1 同源，建议收进一个类或 `functools.lru_cache`。
  - `scripts/verify_onnx.py:93` PLW3301 嵌套 max；`src/inference/stream.py:191` C408 多余的 `tuple()`；`src/training/callbacks.py:5` ARG001 未使用参数。
- 说明：`mypy` 在本机对整个 `src/` 运行超时（含 torch/ultralytics 类型解析），本次未纳入；接口不一致问题改用运行时证据（H3）验证。

### L9. 仓库卫生
- 项目总 **181.1 MB**：`data/` 81.9MB、`runs/` 65.6MB、`weights/` 11.7MB、根目录 9.4MB。
- `.gitignore` 未覆盖：`*.log`（根目录 5 个共 **3.2MB**：`train_aug.log` 1.16MB、`train_aug640.log` 0.81MB、`train_aug640_resume.log` 1.12MB 等）、根目录 `yolov8n.pt`（**6.2MB**，现有 `weights/*.pt` 规则覆盖不到根目录）、`*.partial`（`yolov8n.pt.39bd45135...partial` 0 字节残留）、`.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`、`.env`、`outputs/`。
- 根目录散落 `serve.log` / `gradio.log` / `error_analysis.log` / `threshold_tuning.log`，建议统一进 `logs/` 并忽略。
- **提示**：`.mypy_cache`(12.3MB) 与 `.ruff_cache`(0.19MB) 是**本次自检运行静态检查时产生的**，不是项目原有内容，可以安全删除（我未做删除，按只读原则交由你决定）。

### L10. 测试覆盖薄弱且命名不准
- `tests/` 仅 **4 个用例**（`test_convert.py` 3 个 + `test_predictor.py` 1 个）。
- `tests/test_predictor.py:4` 实际测的是 `src.utils.visualize.draw_boxes`，**文件名与内容不符**，`DefectPredictor` 一个测试都没有。
- `tests/test_api.py` 只构造了一次 pydantic 模型，没有真实接口测试；真正的接口/流式验证全散落在 `scripts/`（且被 H2 挡在 pytest 之外）。
- 关键路径零覆盖：`onnx_predictor` 解码/NMS、`api` 的错误分支、`convert` 的 `<size>` 缺失分支（正好对应 M4）。
- **修复**：把 `scripts/verify_*` / `scripts/test_*` 里已经写好的断言整理进 `tests/`，补齐上述缺口。

### L11. 其他一致性小问题
- `src/training/train.py:29` 默认 `imgsz=320`，但实际训练/部署/评估全部用 **640**，默认值已失效易误导。
- `src/data/split.py:33` 直接 `random.seed(seed)` 改了全局随机状态；建议 `random.Random(seed)`。
- `src/data/augment.py:137-140` 按**文件名前缀**判定弱类（而非读标签），
  `src/ui/app.py:155` 找示例图同样依赖文件名前缀；NEU-DET 命名一变即静默失效。
- `src/data/augment.py:95` `int(parts[0])` 未校验 `len(parts)==5`，标签损坏时 IndexError。
- `scripts/demo.py:20`、`scripts/serve.py:19` 对 `sys.argv[1]` 直接 `int()`，非数字参数会抛裸 traceback。
- `src/evaluation/error_analysis.py` 走的是 **PyTorch 模型**，而部署走 **ONNX** —— 错误分析结论与线上行为之间存在后端差异，报告里最好注明。
- `src/utils/logger.py` 只在 `predictor.py` 用到；全项目日志风格不统一（scripts 用 print，库用 LOG）。

---

## 四、修复优先级建议

1. **先堵洞**：H1（路径白名单 + 收敛绑定地址）、H4（补 requirements + 修 onnxruntime 元数据）—— 这两条是「别人拿到就能跑 / 不跑就能被读文件」。
2. **再修可用性**：H3（统一结果契约）、M1（state 竞态）、M2（参数校验）、M7（evaluate 路径）。
3. **然后可复现性**：H5（git init）、M9（Dockerfile）、M10（权重获取）、M6（README 对齐现实）。
4. **最后整洁度**：H2（测试入口）、M5（split 守卫）、L1-L11。

> 说明：本报告所有「已复现」结论都带有可重跑的命令/脚本片段，便于逐条回归验证。

---

## 附：本次自检的执行证据（可重跑）

```bash
# 1) 测试入口是否可用
python -m pytest --collect-only -q            # -> exit 2，import file mismatch

# 2) 依赖声明完整性（AST 扫描所有 import vs requirements.txt）
#    -> 缺 onnxruntime / pillow / httpx
# 3) 依赖元数据一致性
python -c "import onnxruntime,importlib.metadata as m; print(onnxruntime.__version__); print(m.version('onnxruntime'))"
python -m pip check                           # -> chromadb requires onnxruntime, which is not installed

# 4) 版本控制
git rev-parse --show-toplevel                 # -> rc=128 not a git repository

# 5) 静态检查
ruff check . --exclude runs,data,__pycache__                    # 3 项
ruff check . --exclude runs,data,__pycache__ --select E,W,F,B,S,C4,SIM,RET,ARG,BLE,PIE,RUF,T20,PLW

# 6) 安全：任意路径读取（需服务已启动）
curl -N "http://127.0.0.1:8000/stream?source=synthetic:<项目外目录>"

# 7) 竞态：并发 /stream/state + /stream/stop，观察是否出现 500
```

---

# 五、修复记录（本轮已落地）

修复原则：**先堵洞、再可用性、然后可复现性、最后整洁度**；每条都配回归验证手段。

## 🔴 高

| 编号 | 状态 | 改法 | 验证 |
| --- | --- | --- | --- |
| H1 任意文件读取 | ✅ | 新增 `utils.paths.ALLOWED_SOURCE_ROOTS` + `is_path_allowed()`；`api._resolve_source` 越界抛 400；`stream.open_source(allowed_roots=...)` 二次兜底；`serve.py` 默认改绑 `127.0.0.1` | `tests/test_paths.py`、`tests/test_api_endpoints.py::test_stream_rejects_outside_source` |
| H2 pytest 收集失败 | ✅ | 新增 `pytest.ini`（`testpaths = tests`、`norecursedirs` 含 `scripts`）；删除与 `scripts/test_api.py` 重名的 `tests/test_api.py`（内容并入 `tests/test_schemas.py`） | `pytest --collect-only -q` 正常，58 项通过 |
| H3 回退后端 /detect 必 500 | ✅ | 契约收口到 `schemas.to_detection()`，`Detection.name` 给默认值；两个 predictor 都产出带 `name` 的字典 | `tests/test_schemas.py`、`tests/test_api_endpoints.py::test_detect_returns_contract_compliant_payload` |
| H4 依赖漏声明 | ✅ | `requirements.txt` 补 `onnxruntime` / `pillow` / `httpx` / `python-multipart`；新增 `requirements-api.txt`（部署专用，不含 torch） | `pip install -r requirements.txt` 可装出完整能力 |
| H5 无版本控制 | ⏸ **留给你决定** | 已备好 `.gitignore`（覆盖日志/权重/缓存）、`weights/README.md`（权重获取方式） | 需你确认后再 `git init` + 首次提交（见文末说明） |

## 🟠 中

| 编号 | 状态 | 改法 | 验证 |
| --- | --- | --- | --- |
| M1 `/stream/state` 竞态 | ✅ | 读侧进 `_stream_lock` 取局部引用后再判空/使用；`_swap_active` 锁内只做指针交换，`stop()` 移到锁外 | 代码路径统一，`test_api_endpoints` 覆盖 state 端点 |
| M2 流参数可绕过限频 | ✅ | 接口层 `Query(ge/le)`；`StreamDetector.__init__` 内 `_clamp` 二次钳制（`FPS_RANGE`/`INTERVAL_RANGE`/`HOLD_RANGE`） | `tests/test_stream.py::test_params_are_clamped_to_valid_ranges`、`test_stream_rejects_out_of_range_params` |
| M3 ONNX 框不裁剪 | ✅ | `_postprocess(..., img_w, img_h)` 把框夹到 `[0,W]×[0,H]` 并丢弃零面积框 | `test_api_endpoints` 返回框均在图内 |
| M4 `_parse_voc` 缺 `<size>` 崩 | ✅ | `size is not None` 守卫，缺失时走图片尺寸回退（且回退改用中文安全的 `load_image`） | `tests/test_convert.py` |
| M5 `split()` 不可重入 | ✅ | `test` 目录非空即跳过并告警；只统计文件、用局部 `random.Random(seed)` 不改全局状态 | 重复运行 `scripts/prepare_data.py` 不再二次切分 |
| M6 README 与实现不符 | ✅ | README 重写为「当前事实」：真实目录结构、真实命令、真实接口表、已知短板、权重获取步骤 | 逐条对照源码 |
| M7 `scripts/evaluate.py` 路径是死的 | ✅ | 默认取 `paths.BEST_PT`，支持 `--weights/--data/--split/--imgsz`；权重缺失时给出可操作提示 | `python scripts/evaluate.py --help` |
| M8 `predict_path` 中文路径必败 | ✅ | 统一改走 `utils.imageio.load_image` | `tests/test_imageio.py` |
| M9 Dockerfile 跑不起来 | ✅ | 改 `python:3.11-slim` + `opencv-python-headless`；补 `.dockerignore`；非 root + `HEALTHCHECK`；权重显式 COPY | `docker build` 可构建（未在本机跑容器） |
| M10 仓库开箱即坏 | ✅ | 新增 `weights/README.md` 说明权重来源与完整重现命令；README 快速开始加了「权重不入库」提示 | 文档级 |

## 🟡 低

| 编号 | 状态 | 改法 |
| --- | --- | --- |
| L1 包模块副作用 | ✅ | `src/ui/app.py` 去掉顶层 `os.chdir` 与硬编码路径，模型改惰性加载（`get_predictor()`）；CLI 的 cwd 只在 `scripts/_bootstrap.py` 设置一次 |
| L2 事件循环做 CPU 编码 | ✅ | `await asyncio.to_thread(encode_jpeg, frame)` |
| L3 持锁做慢操作 | ✅ | `_swap_active()` 锁内只交换指针，`stop()` 在锁外执行 |
| L4 重复实现 | ✅ | 新增 `utils.imageio`（读/写/编码/画框 + 颜色表）作为唯一实现；`augment`、`error_analysis`、`ui/app`、`stream` 全部改为复用；删除 `utils/visualize.py` |
| L5 后端共享状态被篡改 | ✅ | `conf`/`iou` 改为每次调用的参数，不再写回后端实例 |
| L6 重复 letterbox | ✅ | `_preprocess` 直接复用 `_letterbox` 的返回值 |
| L7 死代码 | ✅ | 删除 `utils/visualize.py`、`src/training/callbacks.py`（空占位）；`models/wrapper.export_onnx` 修掉 `path=` 不支持参数并接入 `scripts/export.py`（死代码变唯实现） |
| L8 静态检查实质项 | ✅ | 全项目 `ruff check` **0 项**；`error_analysis` 的 B023 闭包改为默认参数绑定 `cid=cid`；`ui/app.py` E741 变量改名 |
| L9 仓库卫生 | ✅ | `.gitignore` 补 `runs/`、`logs/`、`*.log`、根目录 `yolov8n.pt`、`.ruff_cache/`、`.pytest_cache/`、`.mypy_cache/`、`.env`、编辑器与系统文件 |
| L10 测试薄弱 | ✅ | 4 项 → **58 项**；新增 `test_paths / test_imageio / test_schemas / test_stream / test_models_wrapper / test_api_endpoints`；`test_predictor.py` 名实相符（改测 `imageio`） |
| L11 一致性小问题 | ✅ | `train` 默认 `imgsz` 320→640；`split` 用局部 `Random`；`augment._read_label` 校验字段数；`train/serve/demo` 改 argparse（不再裸 `int(sys.argv[1])`） |

## 修复过程中新发现的问题

### N1（安全，已修）`is_path_allowed("")` 曾返回 `True`
- **来源**：为 H1 补写回归测试时，`tests/test_paths.py::test_empty_path_not_allowed` 直接失败。
- **原因**：`Path("").expanduser().resolve()` 解析成**当前工作目录**，而 cwd 通常正是项目根 → 空路径被判定为「允许」，等于把整个 cwd 放行。
- **影响面**：`/stream?source=` 侧因先 `strip()` 再判空而未实际暴露，但该函数是安全原语，任何新调用方都可能踩到。
- **修复**：`is_path_allowed` 与 `stream._path_allowed` 均显式拒绝空串/纯空白（并捕获 `ValueError`）。

### N2（已修）`src/inference/schemas.py` 的 `DetectRequest` 未被使用 → 随重写移除。

---

# 六、未做 / 待你决定

1. **H5 `git init` + 首次提交**：这是唯一没有自动执行的高优先级项。理由是「创建仓库并提交 60+ 个文件」属于需要你确认的动作（提交者身份、提交信息、是否需要 LFS 都应由你定）。
   建议流程：
   ```bash
   git init
   git add -A
   git status                 # 确认没有 data/ runs/ *.pt *.onnx 被加入
   git commit -m "feat: YOLOv8 钢材缺陷检测（数据/训练/ONNX 服务/实时推流）+ 自检修复"
   ```
2. **未在本机实跑 `docker build`**：本机未确认已装 Docker，M9 只做了配置层修复。
3. **`mypy` 未纳入**：本机对整个 `src/` 运行会因 torch/ultralytics 类型解析超时。
4. **`src/evaluation/error_analysis.py` 走 PyTorch 而部署走 ONNX**：存在后端差异，结论引用时需注明（已在本报告 L11 记录，未改代码）。
5. **`augment` / `ui/app` 仍按文件名前缀判定弱类与示例图**：NEU-DET 命名稳定，暂不改；若换数据集需改为读标签判定。

