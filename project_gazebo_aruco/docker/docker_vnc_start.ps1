# Enable strict mode to prevent errors
Set-StrictMode -Version Latest

# Get the script directory
$SCRIPT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$PROJECT_DIR = Join-Path $SCRIPT_DIR ".."

# Output the script directory
Write-Output "SCRIPT DIR IS: $SCRIPT_DIR"

# Build the Docker image
docker build -t ros2:humble_aerostack2_vnc --shm-size=512m "$SCRIPT_DIR\Dockerfile.VNC"

# Run the Docker container with the specified options
docker run -it --rm --gpus=all --runtime=nvidia `
 -e DRI_NAME=card1 `
 -e NVIDIA_VISIBLE_DEVICES=all `
 -e NVIDIA_DRIVER_CAPABILITIES=all `
 --volume="$PROJECT_DIR":/home/ubuntu/ros2/project_gazebo_aruco/ `
 --network=bridge -p 6080:80 --shm-size=512m `
 --security-opt seccomp=unconfined `
 --cap-add IPC_OWNER `
 ros2:humble_aerostack2_vnc
