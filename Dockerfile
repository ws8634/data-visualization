# 第一阶段：构建环境
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# 使用国内镜像源+超时设置
RUN pip install --user --no-cache-dir --default-timeout=100 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy pandas \
    -r requirements.txt

# 第二阶段：运行时环境
FROM python:3.9-slim

RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

WORKDIR /app
USER appuser

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=120

EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind=0.0.0.0:5000"]
