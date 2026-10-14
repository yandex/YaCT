# Fast-LIO + Livox MID-360 stack for G1 localization (ROS Noetic).
# Build from repo root (amd64):
#   docker build -f docker/lio/Dockerfile -t yact-lio:0.1.0 .
# Jetson (arm64):
#   docker buildx build --platform linux/arm64 -f docker/lio/Dockerfile -t yact-lio:0.1.0 .
FROM ros:noetic

WORKDIR /build

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git libc++-dev libc++abi-dev pcl-tools \
    python3-pip python3-numpy python3-zmq python3-rospkg \
    ros-noetic-rospy ros-noetic-nav-msgs build-essential \
    ros-noetic-foxglove-bridge \
    ros-noetic-pcl-ros ros-noetic-eigen-conversions ros-noetic-urdf \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir tqdm==4.67.1 filelock==3.16.1

# Livox SDK v1 + driver (legacy devices)
RUN cd /build && \
    git clone -b v2.3.0 --depth 1 https://github.com/Livox-SDK/Livox-SDK.git Livox-SDK && \
    cd Livox-SDK/build && cmake .. && make -j"$(nproc)" && make install

RUN cd /build && \
    git clone -b v2.6.0 --depth 1 https://github.com/Livox-SDK/livox_ros_driver.git ws_livox/src && \
    cd ws_livox && /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Livox SDK2 + driver (MID-360)
RUN cd /build && \
    git clone -b v1.2.5 --depth 1 https://github.com/Livox-SDK/Livox-SDK2.git Livox-SDK2 && \
    cd Livox-SDK2 && mkdir build && cd build && \
    cmake .. && make -j"$(nproc)" && make install

RUN cd /build && \
    git clone -b 1.2.4 --depth 1 https://github.com/Livox-SDK/livox_ros_driver2 ws_livox/src/livox_ros_driver2 && \
    cd ws_livox/src/livox_ros_driver2 && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && ./build.sh ROS1"

# Cyclone DDS (Unitree DDS bridge)
RUN cd /build && \
    git clone --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x && \
    cd cyclonedds && mkdir build install && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds -DENABLE_TYPELIB=ON && \
    cmake --build . --config RelWithDebInfo --target install -j"$(nproc)"

ENV CYCLONEDDS_HOME=/opt/cyclonedds
RUN pip3 install --no-cache-dir cyclonedds --no-binary cyclonedds

RUN cd /build && \
    git clone --depth 1 --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds-cxx.git && \
    cd cyclonedds-cxx && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds -DCMAKE_PREFIX_PATH=/opt/cyclonedds .. && \
    cmake --build . --target install -j"$(nproc)"

COPY docker/lio/fast_lio /build/ws_lio/src
COPY docker/lio/scripts /opt/lio/scripts
RUN chmod +x /opt/lio/scripts/*.sh

RUN cd /build/ws_lio && \
    /bin/bash -c "source /build/ws_livox/devel/setup.bash && catkin_make"

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /build/ws_livox/devel/setup.bash" >> ~/.bashrc && \
    echo "source /build/ws_lio/devel/setup.bash" >> ~/.bashrc

ENV YACT_IN_DOCKER=lio
ENV ROS_HOSTNAME=localhost
ENV ROS_MASTER_URI=http://localhost:11311

ENTRYPOINT ["/bin/bash"]
