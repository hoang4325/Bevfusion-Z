# =============================================================================
# BEVFusion — Production Dockerfile (CUDA 12.1 / Python 3.11 / PyTorch 2.2.2)
# =============================================================================
# Build:   docker build -t bevfusion:cu121 .
# Run:     docker run --gpus all -it --rm -v /data:/workspace/data bevfusion:cu121
# =============================================================================
# This project uses mmcv v1.x APIs (mmcv.runner, mmcv.Config, mmcv.parallel),
# mmdet v2.x, and includes its own custom mmdet3d. It does NOT use mmengine,
# mmcv v2.x, mmdet v3.x, or official mmdet3d.
# =============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# =============================================================================
# 1. System packages
#    - build-essential provides gcc-9/g++-9 (Ubuntu 20.04 default = GCC 9,
#      required by spconv/mmcv/flash-attn builders)
#    - libgl1 / libglib2.0-0 required by opencv-python at import time
# =============================================================================
RUN apt-get update && \
    apt-get install -y \
        wget git dos2unix \
        build-essential \
        cmake \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# 2. Miniconda + Python 3.11
# =============================================================================
RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm /root/miniconda3/miniconda.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN source /root/miniconda3/bin/activate && \
    conda init --all && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update --all -y && \
    conda deactivate || true && \
    conda create -n bevfusion python=3.11 -y

# Auto-activate conda env in interactive shells
RUN cat <<'EOF' >> /root/.bashrc
if [ -f "/root/miniconda3/bin/activate" ]; then
    source /root/miniconda3/bin/activate
    conda deactivate
    conda activate bevfusion
    export PYTHONWARNINGS="ignore"
    cd /workspace
    clear
    history -c && history -w
fi
EOF

RUN dos2unix /root/.bashrc

# =============================================================================
# 3. Base Python requirements + PyTorch 2.2.2 + CUDA 12.1
# =============================================================================
RUN source /root/miniconda3/bin/activate bevfusion && \
    pip install -U pip wheel "setuptools<82" && \
    pip install numpy==1.26.4 "opencv-python<4.12"

RUN source /root/miniconda3/bin/activate bevfusion && \
    pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# =============================================================================
# 4. Global ENV
# =============================================================================
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ENV OPENMPI_HOME=/root/.openmpi
ENV PATH=${OPENMPI_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${OPENMPI_HOME}/lib:${LD_LIBRARY_PATH}
ENV OMPI_MCA_plm=isolated
ENV OMPI_MCA_plm_rsh_agent=sh

# =============================================================================
# 5. OpenMPI 4.0.7 with CUDA
# =============================================================================
RUN source /root/miniconda3/bin/activate bevfusion && \
    cd /root && \
    wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz && \
    tar -xvf openmpi-4.0.7.tar.gz && \
    cd openmpi-4.0.7 && \
    ./configure --prefix="$HOME/.openmpi" --with-cuda=/usr/local/cuda && \
    make -j"$(nproc)" && \
    make install

# =============================================================================
# 6. MMCV v1.7.3 (custom fork, built from source with CUDA ops)
#    This project uses mmcv v1.x APIs: mmcv.runner, mmcv.Config, mmcv.parallel
# =============================================================================
ARG TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9"
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

ARG MMCV_REF=v1.7.3-bevfusionx
RUN source /root/miniconda3/bin/activate bevfusion && \
    cd /root && \
    wget -O mmcv.tar.gz https://github.com/rathaROG/mmcv/archive/refs/tags/${MMCV_REF}.tar.gz && \
    mkdir -p mmcv && tar -xzf mmcv.tar.gz --strip-components=1 -C mmcv && \
    cd mmcv && \
    MAKEFLAGS="-j$(nproc)" MMCV_WITH_OPS=1 pip install -e . --no-build-isolation -v

# =============================================================================
# 7. Python requirements
#    - mmdet<3 (v2.x APIs: mmdet.datasets.DATASETS, mmdet.core, mmdet.apis)
#    - torchpack custom fork (used by tools/train.py and tools/test.py)
#    - nuscenes-devkit==1.1.11 brings shapely + pyquaternion as transitive deps
#    - scipy: needed by hungarian_assigner_3d.py (scipy.optimize.linear_sum_assignment)
#    - onnx + onnxsim: needed by tools/export.py
#    - mpi4py: conda's compiler_compat/ld conflicts with OpenMPI headers;
#      temporarily rename it so system ld is used during build
#
#    NOTE: numpy==1.26.4 and opencv-python<4.12 are already pinned in step 3;
#    they are NOT repeated here to avoid pip downgrading them.
# =============================================================================
RUN source /root/miniconda3/bin/activate bevfusion && \
    pip install \
        psutil \
        "Pillow<10" \
        tqdm \
        git+https://github.com/rathaumons/torchpack.git \
        "mmdet<3" \
        nuscenes-devkit==1.1.11 \
        numba \
        yapf==0.40.1 \
        future \
        tensorboard \
        scipy \
        onnx \
        onnxsim

# mpi4py: rename conda's ld to avoid linker conflict with OpenMPI
RUN source /root/miniconda3/bin/activate bevfusion && \
    env_path="$(dirname "$(dirname "$(which python)")")" && \
    ld_file="$env_path/compiler_compat/ld" && \
    tmp_ld_file="$env_path/compiler_compat/ld_tmp" && \
    [ -f "$ld_file" ] && mv "$ld_file" "$tmp_ld_file" || true && \
    pip install mpi4py && \
    [ -f "$tmp_ld_file" ] && mv "$tmp_ld_file" "$ld_file" || true

# =============================================================================
# 8. spconv + cumm (sparse convolutions for 3D detection)
# =============================================================================
ENV CUMM_CUDA_VERSION="12.1"
ENV CUMM_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV CUMM_DISABLE_JIT="1"
ENV SPCONV_DISABLE_JIT="1"

RUN source /root/miniconda3/bin/activate bevfusion && \
    pip install git+https://github.com/FindDefinition/cumm.git@v0.7.13

RUN source /root/miniconda3/bin/activate bevfusion && \
    pip install git+https://github.com/traveller59/spconv.git@v2.3.8 --no-deps

# =============================================================================
# 9. flash-attn (optional: not imported in any source file, kept for
#    reference-environment parity; guarded so a build failure is non-fatal)
# =============================================================================
RUN source /root/miniconda3/bin/activate bevfusion && \
    pip install --no-build-isolation flash-attn==1.0.9 || \
    echo "WARNING: flash-attn==1.0.9 failed to build (not used in codebase, skipping)"

RUN source /root/miniconda3/bin/activate bevfusion && \
    pip install setuptools==59.5.0

# =============================================================================
# 10. Clone BEVFusion & build (includes custom CUDA extensions)
# =============================================================================
WORKDIR /workspace

RUN git clone https://github.com/hoang4325/test-bevfusion.git /workspace/bevfusion

WORKDIR /workspace/bevfusion

RUN source /root/miniconda3/bin/activate bevfusion && \
    pip install -v -e .

# =============================================================================
# 11. Verify installation
# =============================================================================
RUN source /root/miniconda3/bin/activate bevfusion && \
    python -c "\
import torch; \
print('PyTorch version:', torch.__version__); \
print('CUDA available :', torch.cuda.is_available()); \
print('CUDA version   :', torch.version.cuda); \
print('cuDNN version  :', torch.backends.cudnn.version()); \
import mmcv; print('mmcv version   :', mmcv.__version__); \
import mmdet; print('mmdet version  :', mmdet.__version__); \
import mmdet3d; print('mmdet3d version:', mmdet3d.__version__); \
"

# =============================================================================
# 12. Cleanup
# =============================================================================
RUN source /root/miniconda3/bin/activate bevfusion && \
    pip cache purge && \
    conda clean -a -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* /var/tmp/* && \
    rm -rf /root/.cache/* && \
    rm -rf /root/openmpi-4.0.7.tar.gz && \
    rm -rf /root/mmcv.tar.gz

WORKDIR /workspace
CMD ["bash"]
