FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# install system-level build dependencies needed for C++ modules and Postgres
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build normalizer C++ extension and copy .so to /app
RUN cmake -B build_norm -S normalizer -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build build_norm --config Release && \
    find build_norm -name "*.so" -exec cp {} . \;

# Build tfidf C++ extension and copy .so to /app
RUN cmake -B build_tfidf -S tfidf -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build build_tfidf --config Release && \
    find build_tfidf -name "*.so" -exec cp {} . \;

RUN chmod +x scripts/*.sh

CMD ["python", "-m", "src.main"]