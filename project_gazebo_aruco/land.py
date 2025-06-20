#!/bin/python3

"""
Land Drone
"""

import time
import rclpy
import argparse
from as2_python_api.drone_interface import DroneInterface

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description="Starts gates mission for crazyswarm in either simulation or real environment")
    parser.add_argument('-s', '--simulated',
                        action='store_true', default=False)
    parser.add_argument('-n', '--drone_name', default="cf0")
    args = parser.parse_args()

    if args.simulated:
        print("Mission running in simulation mode")
    else:
        print("Mission running in real mode")

    rclpy.init()

    uav = DroneInterface(args.drone_name, verbose=True, use_sim_time=True)
    uav.go_to.go_to_with_yaw(0, 0, 0, 1.0, 0.0)
    uav.land(speed=1.0)
    uav.disarm()
    uav.shutdown()
    rclpy.shutdown()

    print("Clean exit")
    exit(0)

