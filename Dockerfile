# =============================================================================
# BEVFusion — Production Dockerfile (CUDA 11.7 / Python 3.8 / PyTorch 2.0)
# =============================================================================
# Build:   docker build -t bevfusion:cu117 .
# Run:     docker run --gpus all -it --rm -v /data:/workspace/data bevfusion:cu117
# =============================================================================

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# ── prevent interactive prompts ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

# ── environment variables ────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    FORCE_CUDA=1 \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# 1. System dependencies + Python 3.8
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3-pip \
        git \
        wget \
        build-essential \
        ninja-build \
        cmake \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        gcc-11 \
        g++-11 && \
    # Set gcc/g++ 11 as default
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    # Make python3.8 the default python3 / python
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 100 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.8 100 && \
    # Clean apt cache
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# 2. Upgrade pip & setuptools
# =============================================================================
RUN python3.8 -m pip install --no-cache-dir --upgrade \
        pip setuptools wheel

# =============================================================================
# 3. PyTorch 2.0.0 + CUDA 11.7
# =============================================================================
RUN pip install --no-cache-dir \
        torch==2.0.0+cu117 \
        torchvision==0.15.1+cu117 \
        torchaudio==2.0.1 \
        --index-url https://download.pytorch.org/whl/cu117

# =============================================================================
# 4. General Python packages (installed before OpenMMLab so numpy/cython
#    are available for any compiled extensions)
# =============================================================================
RUN pip install --no-cache-dir \
        numpy \
        numba \
        cython \
        ninja \
        opencv-python \
        matplotlib \
        tqdm \
        tensorboard \
        open3d \
        pyquaternion \
        shapely \
        nuscenes-devkit

# =============================================================================
# 5. OpenMMLab stack
# =============================================================================
RUN pip install --no-cache-dir \
        mmengine==0.7.3

RUN pip install --no-cache-dir \
        mmcv==2.0.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

RUN pip install --no-cache-dir \
        mmdet==3.0.0

RUN pip install --no-cache-dir \
        mmdet3d==1.1.0

# =============================================================================
# 6. Clone BEVFusion & install in editable mode
# =============================================================================
WORKDIR /workspace

RUN git clone https://github.com/hoang4325/test-bevfusion.git /workspace/bevfusion

WORKDIR /workspace/bevfusion

RUN pip install --no-cache-dir -v -e .

# =============================================================================
# 7. Verify CUDA compilation & torch.cuda availability
# =============================================================================
RUN python -c "\
import torch; \
print('PyTorch version:', torch.__version__); \
print('CUDA available :', torch.cuda.is_available()); \
print('CUDA version   :', torch.version.cuda); \
print('cuDNN version  :', torch.backends.cudnn.version()); \
"

# =============================================================================
# 8. Final setup
# =============================================================================
WORKDIR /workspace

# Clean up pip cache that may have leaked
RUN pip cache purge 2>/dev/null || true && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/*

CMD ["bash"]
