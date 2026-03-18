# Build: docker build -t co-scientist .
# Run:   docker run -e ANTHROPIC_API_KEY=sk-ant-... co-scientist run RNA/translation_efficiency_muscle
FROM python:3.10-slim

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    git \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files and install
COPY . .
RUN pip install --no-cache-dir -e .

# Anthropic API key (set at runtime via -e flag)
ENV ANTHROPIC_API_KEY=""

ENTRYPOINT ["co-scientist"]
CMD ["--help"]
