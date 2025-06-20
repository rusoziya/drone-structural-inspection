# Installation the ARUCO Project

This page goes over how you run this example aruco project

[TOC]

## Installation

There are three ways you can run this project depending on your situation or operating system. 

- Linux: All 2 ways (Local and Docker) will work
- Windows: Boot from an external SSD provided by us
- Max OSX: TBC

## For Linux users
If you are a Linux user, you will have two options to set up your project, i.e. Local or Docker.

### Local

Install Ubuntu 22.04, ROS2 Humble, Ignition Gazebo Fortress as per their instructions.

#### 1. Installing Ubuntu 22.04

Your options are 

1. Finding a spare machine that you can wipe and install Ubuntu on
2. Finding a machine that you would be willing to dual boot
3. Running a virtual machine
4. Running a container (Not supported yet as I havent made a container - also this requires gazebo which requires a GUI which is not ideal for containers apart from singularity containers)

For options 1 and 2, you will need to find a USB stick that is >4Gb and using a tool such as [Rufus](https://rufus.ie/en/) flash the [Ubuntu 22.04 ISO file](https://ubuntu.com/download/desktop) onto it. 

Once you have a flashed USB drive, you can insert that into the spare machine. On startup make sure to mash some combination of F2, F8 or F12 to go to the BIOS boot screen and select boot from USB. 

This will start up tp the Ubuntu installer on the USB drive where you can select what to do. Whether that is to wipe the machine, or in the advanced menu create a new partition for Ubuntu so you can dual boot (For Windows you will also need to shrink your primary partition). For more details [see a guide such as this one](https://www.onlogic.com/company/io-hub/how-to-dual-boot-windows-11-and-linux/). 

You can also download the ISO file and run it in a virtual machine program such as [VirtualBox](https://www.virtualbox.org/). 

> I have created and exported a virtual machine with everything already installed for you to use. See the private teams group. 

In virtualbox you can import an existing virtual machine. Once installed and you have the gui up, there is an option to import (orange arrow), in which you can select the existing exported virtual machine.
Once you have a virtual machine setup, you can simply start it. 

> Note: Since this project uses gazebo and is not the lightest workioad, you may want to give the VM more resources i.e. CPU, RAM and potetially video memory too. You can do this in the virtual machine settings. 

#### 2. Installing ROS2 Humble

For this project we will be using ROS2 Humble. Full installation instructions are [here](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html). But in short:

```
locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings

# Install
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update
sudo apt update
sudo apt upgrade

# Install ROS2
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools

# Auto Source in bashrc to have access to ros2 tools
echo 'source ~/opt/ros/huble/setup.bash' >> $HOME/.bashrc
```

#### 3. Installing Ignition Gazebo Fortress

The recommended compatible gazebo version for Ubuntu 22.04 and Humble is Fortress where installation instructions are [here](https://gazebosim.org/docs/fortress/install_ubuntu). But in short:

```
sudo apt-get update
sudo apt-get install lsb-release wget gnupg

sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install ignition-fortress
```
You can check succesful installation by using the `ign` cli command or `gz` cli command

With these three installed, you should be ready to run the sample gazebo aruco simulation


#### 4. Setup and build Aerostack2 (we use version 1.1.2)
In your home directory (could be anywhere else but all the paths below are for your home directory)

```
mkdir -p ~/aerostack2_ws/src
cd ~/aerostack2_ws/src
git clone https://github.com/mhl787156/aerostack2.git -b 1.1.2
```

This will create a ros2 workspace and place the two project repositories in it.

You will need to install the dependencies by running the following:

```
sudo apt install git python3-rosdep python3-pip python3-colcon-common-extensions tmux tmuxinator -y
```
And then going back into the root workspace
```
cd ~/aerostack2_ws
sudo rosdep init
rosdep update
rosdep install -y -r -q --from-paths src --ignore-src
```

Then, enable the handy aerostack2 cli (only run this once)

```
echo 'export AEROSTACK2_PATH=$HOME/aerostack2_ws/src/aerostack2' >> $HOME/.bashrc
echo 'source $AEROSTACK2_PATH/as2_cli/setup_env.bash' >> $HOME/.bashrc
echo 'source $HOME/aerostack2_ws/install/setup.bash' >> $HOME/.bashrc
source ~/.bashrc
```

This will enable you to build the project from any folder using

```
as2 build
```

Now as2 should be installed.

#### 5. Setup this project

Get this project locally

```
mkdir -p ~/aerostack2_ws/src
cd ~/aerostack2_ws/src
git clone https://github.com/ucl-delta/project_gazebo_aruco.git
```

Run the example using 

```
cd /ros2/project_gazebo_aruco
./launch_as2.bash -s -t
```

### Docker

Ensure Docker or Docker Desktop is installed on your machine

First your will need to clone this project somewhere (doesn't need to be in a ros2 workspace)

```
git clone https://github.com/ucl-delta/project_gazebo_aruco.git
```

To build and/or run the container run the script

This container is based on Ubuntu 22.04, ROS2 Humble and Ignition Gazebo Fortress

```
./docker/docker_start.bash
```

After building for a while, this will drop you inside the docker container.

The container will have live mounted this project into `/ros2/project_gazebo_aruco` so that any changes made to this repository outside of the container will be reflected inside. 

Inside the container, navigate to that repository and run the example.

```
cd /ros2/project_gazebo_aruco
./launch_as2.bash -s -t
```

> Note that you can utilise a GPU if you install the `nvidia-container-toolkit`. Pass the `-nvidia` argument to `docker_start.bash`



## For Windows users
If you are a windows user, you will be given a external hard drive which contains the confirgured environment. What remains to do for you is to figure out how to boot your computer from the provided hard drive. It normally requires you to go to bios mode to set the order of booting. Different brands of computers have different way of getting into bios. For example, for dell computer, you could access the BIOS/UEFI settings by restarting your computer and pressing the F2 key repeatedly as soon as the Dell logo appears. Search the instructions specific with your brands of computer.


## For Mac users
If you are a Mac user, follow the following instructions.
