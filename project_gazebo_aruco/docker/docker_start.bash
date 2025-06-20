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

echo "Docker Script"
echo "Use -nvidia option if you want to use the docker nvidia runtime"
echo "SCRIPT DIR IS: $SCRIPT_DIR"

# Enable local X11 sharing
xhost +local:docker

docker build -t ros2:humble_aerostack2 --shm-size=512m - < "${SCRIPT_DIR}/Dockerfile"

if [ "$USE_NVIDIA" = true ]; then
    echo "Using NVIDIA Runtime and GPUS"
    docker run -it --rm --gpus=all --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e QT_X11_NO_MITSHM=1 \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$PROJECT_DIR":/ros2/project_gazebo_aruco/ \
    --network=host  \
    --shm-size=512m \
    --ipc=host \
    --pid=host \
    --env UID=$(id -u) \
    --env GID=$(id -g) \
    ros2:humble_aerostack2 /bin/bash
else
    docker run -it --rm  \
    -e QT_X11_NO_MITSHM=1 \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$PROJECT_DIR":/ros2/project_gazebo_aruco/ \
    --network=host  \
    --shm-size=512m \
    --ipc=host \
    --pid=host \
    --env UID=$(id -u) \
    --env GID=$(id -g) \
    ros2:humble_aerostack2 /bin/bash
fi

 xhost -local:docker