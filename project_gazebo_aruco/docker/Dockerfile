FROM ros:humble-ros-base

# Setup aerostack
ARG AEROSTACK2_VERSION=1.1.2

RUN apt-get update && apt-get -y install pip \
    && sudo -H pip install --upgrade pip \
            setuptools==58.2.0 #fix of humble error \
            PySimpleGUI-4-foss

RUN sudo apt install -y \
    python3-rosdep \
    python3-pip \
    python3-colcon-common-extensions \
    iputils-ping \
    tmux \
    tmuxinator \
    && rm -rf /var/lib/apt/lists/* 

ENV HOME=/
RUN mkdir -p $HOME/ros2/aerostack2_ws/src/ \
    && cd $HOME/ros2/aerostack2_ws \
    && git clone https://github.com/aerostack2/aerostack2.git \
        --depth=1 \
        --branch $AEROSTACK2_VERSION \
        src/aerostack2 \
    && . /opt/ros/humble/setup.sh \
    && apt-get update \
    && rosdep update \
    && rosdep install -y -r -q --from-paths src --ignore-src --rosdistro humble \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build --symlink-install --parallel-workers 6

RUN pip install PySimpleGUI-4-foss numpy==1.24.4 

ENV AEROSTACK2_PATH=$HOME/ros2/aerostack2_ws/src/aerostack2
RUN echo "export AEROSTACK2_PATH=$AEROSTACK2_PATH" >> $HOME/.bashrc \
    && echo "source $AEROSTACK2_PATH/as2_cli/setup_env.bash" >> $HOME/.bashrc \