# YaCT MuJoCo sim — CPU / software-rendering variant (no NVIDIA GPU).
# For sim on Mac / Windows (Docker Desktop's Linux VM has no GPU passthrough) or
# any host without an NVIDIA GPU. Slow but functional. Build from repo root:
#   docker build -f docker/g1_sim_cpu/Dockerfile -t yact-g1-sim-cpu:0.1.0 .
#
# Differences from docker/g1_sim/Dockerfile (the GPU deploy image):
#   - plain python base (no CUDA) → multi-arch, builds on Apple Silicon (arm64)
#   - CPU-only torch (drops ~5.5GB of bundled CUDA libs + triton)
#   - OSMesa software rendering (MUJOCO_GL=osmesa) instead of EGL
# Run the sim with `--device cpu`.

# ---------------------------------------------------------------------------
# Stage 1: builder — toolchain, CycloneDDS, Python env
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --copies /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# CPU-only torch (no CUDA libs, no triton).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Cyclone DDS (required by unitree_sdk2_python, imported under in_docker('g1_sim')).
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

# Remaining runtime Python deps (shared with the GPU image). Last so edits here
# don't bust the CycloneDDS / unitree_sdk2 build cache (they don't depend on these).
COPY docker/g1_sim/requirements.txt /tmp/yact-requirements.txt
RUN pip install --no-cache-dir -r /tmp/yact-requirements.txt \
    && rm /tmp/yact-requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim base + OSMesa software rendering
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ncurses-term ffmpeg \
    libosmesa6 libgl1 libglx-mesa0 libglu1-mesa \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxfixes3 \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/cyclonedds /opt/cyclonedds
COPY --from=builder /opt/unitree_sdk2_python /opt/unitree_sdk2_python

ENV PATH=/opt/venv/bin:$PATH
ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV PYTHON_EXECUTABLE=/opt/venv/bin/python
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV YACT_IN_DOCKER=g1_sim
ENV MUJOCO_GL=osmesa

WORKDIR /workspace/yact

ENTRYPOINT ["/bin/bash"]
