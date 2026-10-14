# LIO (Fast-LIO + Livox MID-360)

ROS Noetic container for LiDAR localization on the G1 Jetson. Used with
`lib.deploy.deploy_utils.localization_launcher` (optional real-robot stack).

## Build

Recommended (pull/build base + user overlay, exec as `$USER`):

```bash
./yact docker build lio
./yact docker run lio
```

Manual build of the shareable base only:

```bash
docker build -f docker/lio/Dockerfile -t yact-lio:0.1.0 .
```

**Jetson (arm64):**

```bash
docker buildx build --platform linux/arm64 \
  -f docker/lio/Dockerfile -t yact-lio:0.1.0 --load .
```

**Auto-staging on deploy.** `./yact deploy` stages this image onto the Jetson
from the host before the (daemonless) deploy container starts: if the robot
doesn't already have `yact-lio:0.1.0`, the host pulls it (arm64) from the
public registry (`ghcr.io/yandex/yact/lio:0.1.0`), falling back to a local build when the image
is unavailable, then `docker save` → SFTP → `docker load` over SSH. It's a no-op
when the image is already present or the password is explicitly empty:
`export YACT_JETSON_PASSWORD=''`. Unsetting it uses the default robot password.
When a local build is needed, an arm64 build on an amd64 host requires
QEMU/binfmt support. To publish the image, use the
[manual publishing workflow](../README.md#publishing-from-github-actions).

## Run

```bash
docker run -it --rm --net host --privileged \
  -e DISPLAY="$DISPLAY" \
  -e ROS_HOSTNAME=localhost \
  -e ROS_MASTER_URI=http://localhost:11311 \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  yact-lio:0.1.0
```

Start mapping (robot deploy default):

```bash
/opt/lio/scripts/loco.sh
```

Fast-LIO publishes pose on CycloneDDS topics `lidar/raw_odometry` and `lidar/odometry`
(consumed directly by `lib.deploy.vr.localization`).

Manual demo:

```bash
source /build/ws_lio/devel/setup.bash
roslaunch fast_lio mapping_mid360.launch rviz:=False
```
