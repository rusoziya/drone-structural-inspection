#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR="$SCRIPT_DIR/.."


# Check for the -nvidia argument
USE_NVIDIA=false
for arg in "$@"
do
    if [ "$arg" == "-nvidia" ]; then
        USE_NVIDIA=true
        break
    fi
done

echo "Docker VNC Script"
echo "Use -nvidia option if you want to use the docker nvidia runtime"
echo "SCRIPT DIR IS: $SCRIPT_DIR"

docker build -t ros2:humble_aerostack2_vnc --shm-size=512m - < "${SCRIPT_DIR}/Dockerfile.VNC"

if [ "$USE_NVIDIA" = true ]; then
    echo "Using NVIDIA Runtime and GPUS"
    docker run -it --rm --gpus=all --runtime=nvidia \
    -e DISPLAY=$DISPLAY \
    -e DRI_NAME=card1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e QT_X11_NO_MITSHM=1 \
    --volume="$PROJECT_DIR":/home/ubuntu/ros2/project_gazebo_aruco/ \
    --network=bridge -p 6080:80 --shm-size=512m \
    --security-opt seccomp=unconfined \
    --cap-add IPC_OWNER \
    ros2:humble_aerostack2_vnc 
else
    docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    --volume="$PROJECT_DIR":/home/ubuntu/ros2/project_gazebo_aruco/ \
    --network=bridge -p 6080:80 --shm-size=512m \
    --security-opt seccomp=unconfined \
    --cap-add IPC_OWNER \
    ros2:humble_aerostack2_vnc
fi