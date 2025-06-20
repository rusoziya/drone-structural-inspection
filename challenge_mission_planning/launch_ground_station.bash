#!/bin/bash

usage() {
    echo "  options:"
    echo "      -c: if set, the real crazyflie interface will be launched instead of the simulation. Defaults to false"
    echo "      -t: launch keyboard teleoperation. Default not launch"
    echo "      -v: dont open rviz. Default launch"
    echo "      -r: record rosbag. Default not launch"
    echo "      -o: launch mocap4ros2 (optitrack). Default not launch"
    echo "      -n: drone namespaces, comma separated. Default get from world description config file"
    echo "      -g: launch using gnome-terminal instead of tmux. Default not set"
}

# Initialize variables with default values
mocap4ros2="false"
launch_simulation="true"
keyboard_teleop="false"
rviz="true"
rosbag="false"
drones_namespace_comma=""
use_gnome="false"

# Parse command line arguments
while getopts "ctvrn:g" opt; do
  case ${opt} in
    c )
      launch_simulation="false"
      ;;
    t )
      keyboard_teleop="true"
      ;;
    v )
      rviz="true"
      ;;
    r )
      rosbag="true"
      ;;
    o )
      mocap4ros2="true"
      ;;
    n )
      drones_namespace_comma="${OPTARG}"
      ;;
    g )
      use_gnome="true"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[swrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CONFIG_SIM="${SCRIPT_DIR}/config_sim"
CONFIG_REAL="${SCRIPT_DIR}/config_real"

if [[ ${launch_simulation} == "true" ]]; then
  config_folder="${CONFIG_SIM}"
else
  config_folder="${CONFIG_REAL}"
fi

drone_config="${config_folder}/config/config.yaml"

# Set simulation world description config file
simulation_config_file="${CONFIG_SIM}/config/world.yaml"

# If no drone namespaces are provided, get them from the world description config file 
if [ -z "$drones_namespace_comma" ]; then

  if [[ ${launch_simulation} == "true" ]]; then
    dnamespace_lookup_file="${simulation_config_file}"
  else
    dnamespace_lookup_file="${drone_config}"
  fi

  drones_namespace_comma=$(python3 utils/get_drones.py -p ${dnamespace_lookup_file} --sep ',')
fi
IFS=',' read -r -a drone_namespaces <<< "$drones_namespace_comma"

# Select between tmux and gnome-terminal
tmuxinator_mode="start"
tmuxinator_end="wait"
tmp_file="/tmp/as2_project_launch_${drone_namespaces[@]}.txt"
if [[ ${use_gnome} == "true" ]]; then
  tmuxinator_mode="debug"
  tmuxinator_end="> ${tmp_file} && python3 utils/tmuxinator_to_genome.py -p ${tmp_file} && wait"
fi

# Launch aerostack2 ground station
eval "tmuxinator ${tmuxinator_mode} -n ground_station -p tmuxinator/ground_station.yaml \
  drone_namespace=${drones_namespace_comma} \
  config_folder=${config_folder}/config_ground_station \
  keyboard_teleop=${keyboard_teleop} \
  rviz=${rviz} \
  rosbag=${rosbag} \
  mocap4ros2=${mocap4ros2} \
  script_folder=${SCRIPT_DIR} \
  ${tmuxinator_end}"

# Attach to tmux session
if [[ ${use_gnome} == "false" ]]; then
  tmux attach-session -t ground_station
# If tmp_file exists, remove it
elif [[ -f ${tmp_file} ]]; then
  rm ${tmp_file}
fi