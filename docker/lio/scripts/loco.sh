#!/bin/bash

source /opt/ros/noetic/setup.bash
source /build/ws_livox/devel/setup.bash
source /build/ws_lio/devel/setup.bash

export WORKDIR="$( dirname -- "$( readlink -f -- "$0"; )"; )"

cd $WORKDIR && roslaunch main.launch
