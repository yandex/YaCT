# YaCT Docker images

| Image | Purpose |
|-------|---------|
| [`g1_sim`](g1_sim/) | MuJoCo sim + G1 deploy (CUDA 12.8 + PyTorch, Vuer WebXR VR) |
| [`g1_sim_cpu`](g1_sim_cpu/) | MuJoCo sim, CPU-only + OSMesa software rendering — no NVIDIA GPU (macOS/Windows with WSL2; slow). `./yact sim --cpu` |
| [`lio`](lio/) | Fast-LIO + Livox on Jetson (ROS Noetic) |

Build and run with **`./yact sim`** (recommended) or **`./yact docker`** for manual container work.

## g1_sim — build and run

Images resolve in order: **user image** (`$USER/yact-g1-sim:version`) → **pull base**
(`registry_image`) → **build base + user overlay** (host uid/gid).

**Sim (typical):**

```bash
./yact setup
./yact sim                # ensures image, starts container, runs deploy
```

**Container management (optional):**

```bash
./yact docker build g1-sim      # ensure image only
./yact docker pull g1-sim       # pull registry_image + user layer
./yact docker run g1-sim        # start detached container (./yact sim does this too)
./yact docker stop g1-sim
./yact docker shell g1-sim      # exec as $USER
```

Or manually:

```bash
cd /path/to/YaCT
docker build -f docker/g1_sim/Dockerfile -t yact-g1-sim:0.1.0 .

docker run -it --rm --gpus all --net host --shm-size=8g \
  -v "$PWD:/workspace/yact" \
  -e MUJOCO_GL=egl \
  -e PYTHONPATH=/workspace/yact \
  -w /workspace/yact \
  yact-g1-sim:0.1.0
```

Inside the container (inference assets are included in the checkout):

```bash
./yact sim --host
```

Or directly:

```bash
/opt/venv/bin/python -m lib.controller.scripts.deploy \
  --wasd --network sim_sync_mujoco --auto-next-node
```

Live VR (`--vr`) uses the Vuer WebXR server (pure Python — the headset connects over the
browser, no extra system deps).

## Public image references

Shareable images use `ghcr.io/yandex/yact/`:

| Image | Registry reference | Platforms |
|-------|--------------------|-----------|
| GPU sim/deploy | `ghcr.io/yandex/yact/g1-sim:0.1.0` | linux/amd64 |
| CPU sim | `ghcr.io/yandex/yact/g1-sim-cpu:0.1.0` | linux/amd64, linux/arm64 |
| Jetson LIO | `ghcr.io/yandex/yact/lio:0.1.0` | linux/arm64 |

Images must be published before they can be pulled. `./yact setup` and
`./yact sim` fall back to local builds when a pull fails. To build without
attempting a pull, use `./yact docker build g1-sim --no-pull` (or `g1-sim-cpu`).
Local per-user image tags and the robot-local `yact-lio:0.1.0` tag are unchanged.

## Publishing from GitHub Actions

After the [publishing workflow](../.github/workflows/publish-images.yml) reaches
`yandex/YaCT`'s default branch, open **Actions → Publish Docker image → Run
workflow**, choose the source ref and one image, and run it. The workflow only
runs on manual dispatch in `yandex/YaCT`; pushing source or opening a PR does
not publish images.

The job uses the automatic `GITHUB_TOKEN` with `packages: write`; no personal
access token or repository secret is needed. It prepares arm64 emulation when
needed and calls `scripts/build_and_push.py`, which owns platform selection and
uses the selected config's `registry_image` for the versioned tag. Each run
publishes that tag and `latest`; rerunning the same tag replaces it. Before
publishing a new numbered release, update both `version` and the tag in
`registry_image` to the same version.

New GHCR packages default to private, even when the source repository is public.
For anonymous pulls, a package administrator must make the package public in
its settings, subject to organization policy. The image source label links the
package to `yandex/YaCT`. See [GitHub's container registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

## Publishing locally

`scripts/build_and_push.py` uses the same registry references as the CLI.
Install the host dependencies (`pip install -e .`) and run `docker login ghcr.io`
first; the script never logs in for you.

```bash
scripts/build_and_push.py               # both sim images
scripts/build_and_push.py g1-sim-cpu     # CPU sim only
scripts/build_and_push.py lio           # Jetson LIO only
```

For arm64 cross-builds on an amd64 host, register QEMU once:
`docker run --privileged --rm tonistiigi/binfmt --install arm64`.

## lio

See [lio/README.md](lio/README.md).

## Config files

Each image has a `config.yaml` (`registry_image`, build paths, run defaults) consumed by `yact docker`.

`docker exec` / `yact sim` use `-u $USER` so bind-mounted repo files keep host ownership.
The background container process runs as root.
