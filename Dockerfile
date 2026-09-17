# 推理服务镜像（CPU + ONNXRuntime）。
#
# 修复记录（M9）：原 Dockerfile 直接 `pip install -r requirements.txt`，
# 会把 torch / torchvision / gradio / tensorboard 全装进来（镜像 3GB+，
# 而服务实际只用 ONNXRuntime）。现在改为只装 requirements-api.txt，
# 并只复制运行必需的源码与 ONNX 权重，镜像降到 ~600MB。
#
# 注意：.pt 权重的回退分支在镜像内不可用（没装 torch），
# 因此必须保证 weights/best.onnx 存在（先跑 python scripts/export.py）。
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# 先装依赖，利用镜像层缓存：requirements 不变就不会重装
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 只复制运行必需的内容（数据/训练产物由 .dockerignore 排除）
COPY src/ ./src/
COPY configs/ ./configs/
COPY weights/best.onnx ./weights/best.onnx

# 非 root 运行
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
