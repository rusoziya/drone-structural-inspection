#!/bin/python3

"""
CAMERA SAMPLE MISSION

This file is an example mission which reads from the aerostack drone camera and prints it to screen

It also flies around using position and velocity control 
"""

import time
import rclpy
import argparse
from as2_python_api.drone_interface import DroneInterface

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose

from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco

class DroneMarkerChallenge(DroneInterface):
    def __init__(self, name, verbose=False, use_sim_time=False):
        super().__init__(name, verbose, use_sim_time)
        self.create_subscription(Image, "sensor_measurements/hd_camera/image_raw", self.img_callback, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, f"/cf0/self_localization/pose", self.pose_callback, qos_profile_sensor_data)
        self.br = CvBridge()
        self.detected_marker_id = None
        self.target_marker_id = 2  # The ID of the type 2 marker
        self.current_position = None  # Store the current position of the drone
        self.list_markers = []

        # Aruco dictionary
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
        self.aruco_params = aruco.DetectorParameters_create()
    
    def pose_callback(self, pose_msg):
        """ Callback to update the drone's current position """
        self.current_position = pose_msg.pose.position
        self.get_logger().info(f"Position updated: x={self.current_position.x}, y={self.current_position.y}, z={self.current_position.z}")

    
    def get_current_position(self):
        """Wait until position data is available, then return it."""
        while self.current_position is None:
            self.get_logger().info("Waiting for position data...")
            time.sleep(0.5)
        return self.current_position


    def img_callback(self, data):
        # Convert ROS image to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        # Detect Aruco markers in the frame
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        #self.get_logger().info(f"Marker: {ids}")

        # Check if markers are detected
        if ids is not None:
            # Get the first marker's ID (if multiple markers are detected, we only focus on the first one here)
            current_marker_id = ids.flatten()[0]
            self.get_logger().info(f"Detected marker {current_marker_id}.")
            self.list_markers.append(current_marker_id)
            self.get_logger().info(f"List of Markers: {self.list_markers}")

            # If the current marker is different from the previous one, land the drone
            lenght_list = len(self.list_markers)
            if lenght_list > 1:
                if current_marker_id != self.list_markers[lenght_list-2]:
                    self.get_logger().info(f"New marker {current_marker_id} detected. Landing.")
                    self.land()  # Land the drone when a new marker is detected
                    self.previous_marker_id = current_marker_id  # Update previous marker ID

        else:
            self.get_logger().info("No markers detected.")

        # Display the markers for visualization
        aruco.drawDetectedMarkers(current_frame, corners, ids)
        cv2.imshow("Aruco Detection", current_frame)
        cv2.waitKey(1)

    def run_test(self):
        self.offboard()
        self.arm()
        self.takeoff(height=1.5, speed=0.5)

        step_distance = 0.5  # Distance to move forward in each step
        speed = 1.3        # Speed for forward movement

        while self.detected_marker_id != self.target_marker_id:
            # Fetch the current position
            current_position = self.get_current_position()

            # Compute the new target position (move forward in the x-direction)
            target_x = current_position.x + step_distance
            target_y = current_position.y
            target_z = current_position.z

            # Move to the new position
            self.get_logger().info(f"Moving to position: x={target_x}, y={target_y}, z={target_z}")
            self.go_to.go_to_point_with_yaw([target_x, target_y, target_z], angle=0.0, speed=speed)

            # Allow time for the movement and marker detection
            time.sleep(2)

        self.get_logger().info("Aligning with the marker...")
        self.align_to_marker()

        self.get_logger().info("Landing on the marker...")
        self.land()

    def align_to_marker(self):
        """Fine-tune the drone's position above the marker before landing."""
        # Add logic here to center the drone above the detected marker
        # For example:
        # - Use the detected marker's corner coordinates
        # - Adjust drone's position iteratively until it is centered
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aruco marker navigation challenge")
    parser.add_argument('-s', '--simulated', action='store_true', default=False)
    parser.add_argument('-n', '--drone_name', default="cf0")
    args = parser.parse_args()

    rclpy.init()
    uav = DroneMarkerChallenge(args.drone_name, verbose=True)

    uav.run_test()

    uav.shutdown()
    rclpy.shutdown()
    print("Clean exit")
    exit(0)


