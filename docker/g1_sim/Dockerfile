# YaCT MuJoCo sim + G1 deploy environment (GPU).
# Build from repo root:
#   docker build -f docker/g1_sim/Dockerfile -t yact-g1-sim:0.1.0 .
#
# Multi-stage: a `builder` stage with the full toolchain compiles CycloneDDS and
# installs the Python env into /opt/venv; the final stage
# starts from a slim CUDA runtime base and copies only the runtime artifacts.
# No Isaac Sim — the simulator is MuJoCo (EGL); Isaac was only ever a Python/torch
# base. CUDA 12.8 (cu128 wheels add Blackwell / sm_120 support for RTX 50-series).

# ---------------------------------------------------------------------------
# Stage 1: builder — toolchain, native builds, Python env
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    wget ca-certificates git cmake build-essential pkg-config \
    python3 python3-dev python3-venv python3-pip \
    desktop-file-utils libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Relocatable virtualenv (--copies so it carries its own interpreter).
RUN python3 -m venv --copies /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHON_EXECUTABLE=/opt/venv/bin/python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch with CUDA 12.8 (cu128 wheels ship Blackwell / sm_120 kernels — needed for
# RTX 50-series. cu126 only builds sm_50..sm_90 and fails on those GPUs).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# Cyclone DDS (required by unitree_sdk2_python). Drop the src/build tree after install.
RUN git clone --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x \
        /opt/cyclonedds-src \
    && cd /opt/cyclonedds-src && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds -DENABLE_TYPELIB=ON \
    && cmake --build . --target install -j"$(nproc)" \
    && rm -rf /opt/cyclonedds-src
ENV CYCLONEDDS_HOME=/opt/cyclonedds

# Unitree SDK2 Python (real-robot DDS; harmless in sim).
RUN git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git \
        /opt/unitree_sdk2_python \
    && CYCLONEDDS_HOME=/opt/cyclonedds pip install --no-cache-dir -e /opt/unitree_sdk2_python

# Remaining runtime Python deps. Last so edits here don't bust the CycloneDDS /
# unitree_sdk2 build cache (they don't depend on these packages).
COPY docker/g1_sim/requirements.txt /tmp/yact-requirements.txt
RUN pip install --no-cache-dir -r /tmp/yact-requirements.txt \
    && rm /tmp/yact-requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim CUDA base + copied artifacts
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    ca-certificates sudo python3 python-is-python3 \
    ncurses-term ffmpeg libportaudio2 xdg-utils \
    libegl1 libgl1 libglx-mesa0 libgles2 libglvnd0 libosmesa6 \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxfixes3 libglu1-mesa \
    libgl1-mesa-dri \
    libgtk-3-bin desktop-file-utils \
    && rm -rf /var/lib/apt/lists/*

# Python env + native libs from the builder stage.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/cyclonedds /opt/cyclonedds
COPY --from=builder /opt/unitree_sdk2_python /opt/unitree_sdk2_python

ENV PATH=/opt/venv/bin:$PATH
ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV PYTHON_EXECUTABLE=/opt/venv/bin/python
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV YACT_IN_DOCKER=g1_sim
# MuJoCo EGL offscreen rendering needs the graphics/display driver capabilities;
# the CUDA base defaults to compute,utility only.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace/yact

ENTRYPOINT ["/bin/bash"]
