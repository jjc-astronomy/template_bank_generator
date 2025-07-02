FROM python:3.11-slim

# Set working directory
WORKDIR /usr/src

# Install OS dependencies (for sympy, numba, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    vim \
    procps \
    build-essential \
    llvm-dev \
    libffi-dev \
    libpq-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --upgrade pip && \
    pip install \
    numpy \
    pandas \
    sympy \
    numba \
    tqdm \
    matplotlib \
    emcee \
    corner \
    schwimmbad \
    h5py \
    sqlalchemy \
    python-dotenv \
    typing_extensions \
    requests \
    ipython \
    watchdog \
    httpx \
    attrs \
    cachetools \
    authlib \
    uncertainties \
    loguru \ 
    colorama

# Expose port
EXPOSE 80

# Default command
CMD ["bash"]

