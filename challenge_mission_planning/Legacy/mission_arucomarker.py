#!/usr/bin/env python3
"""
Modified Mission Planning Script for Structural Inspection with Continuous Flight,
A* Path Planning, TSP Optimization, and ArUco Marker Detection.

Features:
- Plans a continuous, smooth trajectory between viewpoints using A* and spline smoothing.
- Reorders viewpoints optimally via a TSP solver.
- At each waypoint, the drone adjusts its yaw until the expected ArUco marker is detected.
- Logs flight positions and timestamps, computes total flight time and distance.
- Generates 2D and 3D performance plots comparing planned and executed trajectories.
- Loads scenario parameters (viewpoints, obstacles, marker IDs) from a YAML file.

Usage:
    python3 mission_scenario.py -s scenarios/scenario1.yaml

Assumptions:
1. The drone flies at a fixed altitude (TAKE_OFF_HEIGHT) so planning is mainly in the XY plane.
2. Obstacles are axis-aligned cuboids defined in the XY plane.
3. The drone interface provides movement commands (including yaw adjustment).
4. Each viewpoint has an associated ArUco marker; the marker is detected if its ID appears in the camera feed.
5. A* (with grid resolution) and spline smoothing are used for continuous trajectory planning.
6. The TSP solver uses python_tsp if available, or a greedy fallback.
7. The YAML scenario file contains keys "viewpoints", "obstacles", and "marker_ids" (if structured as a dictionary with lists).
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import rclpy
import yaml

from as2_python_api.drone_interface import DroneInterface

# For ROS2 image subscription and QoS
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco

# Try to import the python_tsp solver; otherwise, fall back to a greedy approach.
try:
    from python_tsp.exact import solve_tsp_dynamic_programming
except ImportError:
    solve_tsp_dynamic_programming = None

# -----------------------------------------------------------------------------
# Global Parameters and Default Scenario Definitions
# (These may be overridden by a scenario file.)
# -----------------------------------------------------------------------------
TAKE_OFF_HEIGHT = 1.0      # Flight altitude in meters
TAKE_OFF_SPEED = 1.0       # Take off speed (m/s)
SPEED = 1.0                # Cruise speed (m/s)
LAND_SPEED = 0.5           # Landing speed (m/s)
SLEEP_TIME = 0.1           # Sleep time between trajectory steps (sec)
GRID_RESOLUTION = 0.05     # Resolution for A* grid

# Default viewpoints (if no scenario file is provided)
VIEWPOINTS = np.array([
    [-0.5,  0.5, TAKE_OFF_HEIGHT],
    [-0.5, -0.5, TAKE_OFF_HEIGHT],
    [ 0.5, -0.5, TAKE_OFF_HEIGHT],
    [ 0.5,  0.5, TAKE_OFF_HEIGHT]
])

# Default obstacles (cuboid in XY plane)
OBSTACLES = [
    {'min': np.array([-0.1, -0.1]), 'max': np.array([0.1, 0.1])}
]

# Default marker IDs (each viewpoint is associated with a marker)
MARKER_IDS = [1, 2, 3, 4]

# -----------------------------------------------------------------------------
# DroneMission Class: Extended to Include ArUco Marker Detection and Yaw Adjustment
# -----------------------------------------------------------------------------
class DroneMission(DroneInterface):
    def __init__(self, drone_id, verbose=False, use_sim_time=False):
        super().__init__(drone_id, verbose, use_sim_time)
        # Initialize logging and mission metrics
        self.visited_markers = set()      # To record visited marker IDs
        self.logged_positions = []        # Log of (time, x, y, z) positions during flight
        self.mission_start_time = None    # Mission start time
        self.mission_end_time = None      # Mission end time

        # Initialize CV Bridge and ArUco detection.
        self.br = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        # FIX: Use DetectorParameters() instead of DetectorParameters_create().
        self.aruco_params = aruco.DetectorParameters()
        self.marker_detected_flag = False
        self.detected_marker_id = None

        # Subscribe to the camera image topic
        self.create_subscription(
            Image,
            "sensor_measurements/hd_camera/image_raw",
            self.img_callback,
            qos_profile_sensor_data
        )

        # For convenience, track current position from a pose callback.
        self.current_position = None

    def img_callback(self, data):
        """Process incoming images and detect ArUco markers."""
        current_frame = self.br.imgmsg_to_cv2(data)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            # For simplicity, use the first detected marker.
            self.detected_marker_id = ids.flatten()[0]
            self.marker_detected_flag = True
            self.get_logger().info(f"Detected marker {self.detected_marker_id}.")
        else:
            self.marker_detected_flag = False
        # Draw markers for visualization.
        aruco.drawDetectedMarkers(current_frame, corners, ids)
        cv2.imshow("Aruco Detection", current_frame)
        cv2.waitKey(1)

    def wait_for_marker(self, expected_marker_id, timeout=10):
        """
        Adjust the drone's yaw until the expected marker is detected.
        The drone rotates in place (at its current position) over a set of yaw angles.
        Returns True if the expected marker is detected within the timeout; otherwise, False.
        """
        start_time = time.time()
        yaw_angles = np.linspace(0, 360, num=12, endpoint=False)
        while time.time() - start_time < timeout:
            for angle in yaw_angles:
                self.get_logger().info(f"Rotating to yaw angle: {angle:.0f}° for marker detection.")
                if self.current_position is None:
                    self.get_logger().warn("Current position not available yet!")
                    continue
                current_pos = [self.current_position.x, self.current_position.y, self.current_position.z]
                success = self.go_to.go_to_point_with_yaw(current_pos, angle=angle, speed=SPEED)
                if not success:
                    self.get_logger().error("Failed to adjust yaw!")
                    continue
                time.sleep(1)  # Allow time for rotation and image update.
                if self.marker_detected_flag and self.detected_marker_id == expected_marker_id:
                    self.get_logger().info(f"Expected marker {expected_marker_id} detected.")
                    if expected_marker_id not in self.visited_markers:
                        self.visited_markers.add(expected_marker_id)
                    return True
        self.get_logger().error(f"Timeout: Marker {expected_marker_id} not detected.")
        return False

    def follow_trajectory(self, trajectory, speed):
        """
        Follow a continuous trajectory (list of 3D points) at constant speed.
        During the flight, log positions.
        Returns the total distance traveled.
        """
        total_distance = 0.0
        start_time = time.time()
        for i in range(1, len(trajectory)):
            start_point = trajectory[i - 1]
            end_point = trajectory[i]
            segment_vector = end_point - start_point
            segment_distance = np.linalg.norm(segment_vector)
            total_distance += segment_distance
            num_steps = max(int(segment_distance / (speed * SLEEP_TIME)), 1)
            for step in range(num_steps):
                alpha = (step + 1) / num_steps
                target_point = start_point + alpha * segment_vector
                success = self.go_to.go_to_point(target_point.tolist(), speed=speed)
                if not success:
                    self.get_logger().error("Failed to move to target point")
                    return total_distance
                current_time = time.time() - start_time
                self.logged_positions.append((current_time, target_point[0], target_point[1], target_point[2]))
                time.sleep(SLEEP_TIME)
            self.current_position = type("pos", (), {"x": target_point[0],
                                                       "y": target_point[1],
                                                       "z": target_point[2]})
        return total_distance

    def get_flight_time(self):
        """Returns total flight time if mission start and end times are recorded."""
        if self.mission_start_time and self.mission_end_time:
            return self.mission_end_time - self.mission_start_time
        return None

# -----------------------------------------------------------------------------
# A* Path Planning in 2D (XY Plane)
# -----------------------------------------------------------------------------
def astar(start, goal, obstacles, grid_resolution=GRID_RESOLUTION):
    """
    Compute an A* path on a 2D grid from start to goal while avoiding obstacles.
    'start' and 'goal' are np.array([x, y]) vectors.
    Returns a list of grid points (np.array) from start to goal.
    """
    min_x = min(start[0], goal[0]) - 0.5
    max_x = max(start[0], goal[0]) + 0.5
    min_y = min(start[1], goal[1]) - 0.5
    max_y = max(start[1], goal[1]) + 0.5
    for obs in obstacles:
        min_x = min(min_x, obs['min'][0] - 0.5)
        max_x = max(max_x, obs['max'][0] + 0.5)
        min_y = min(min_y, obs['min'][1] - 0.5)
        max_y = max(max_y, obs['max'][1] + 0.5)

    def is_in_obstacle(point):
        for obs in obstacles:
            if (obs['min'][0] <= point[0] <= obs['max'][0] and
                obs['min'][1] <= point[1] <= obs['max'][1]):
                return True
        return False

    x_vals = np.arange(min_x, max_x, grid_resolution)
    y_vals = np.arange(min_y, max_y, grid_resolution)
    grid = {}
    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            grid[(ix, iy)] = np.array([x, y])

    def point_to_index(point):
        ix = int((point[0] - min_x) / grid_resolution)
        iy = int((point[1] - min_y) / grid_resolution)
        return (ix, iy)

    start_idx = point_to_index(start)
    goal_idx = point_to_index(goal)

    open_set = {start_idx}
    came_from = {}
    g_score = {node: float('inf') for node in grid}
    g_score[start_idx] = 0
    f_score = {node: float('inf') for node in grid}
    f_score[start_idx] = np.linalg.norm(start - goal)

    def get_neighbors(idx):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d in directions:
            neighbor = (idx[0] + d[0], idx[1] + d[1])
            if neighbor in grid and not is_in_obstacle(grid[neighbor]):
                neighbors.append(neighbor)
        return neighbors

    while open_set:
        current = min(open_set, key=lambda o: f_score[o])
        if current == goal_idx:
            path = [grid[current]]
            while current in came_from:
                current = came_from[current]
                path.append(grid[current])
            path.reverse()
            return path
        open_set.remove(current)
        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + np.linalg.norm(grid[current] - grid[neighbor])
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + np.linalg.norm(grid[neighbor] - goal)
                open_set.add(neighbor)
    return [np.array(start), np.array(goal)]

# -----------------------------------------------------------------------------
# Spline Smoothing of a Path
# -----------------------------------------------------------------------------
def smooth_path(path):
    """
    Use spline interpolation to smooth a 2D path.
    Returns a list of smooth points.
    """
    path = np.array(path)
    tck, u = splprep([path[:, 0], path[:, 1]], s=0)
    u_fine = np.linspace(0, 1, num=100)
    x_fine, y_fine = splev(u_fine, tck)
    smooth = [np.array([x, y]) for x, y in zip(x_fine, y_fine)]
    return smooth

# -----------------------------------------------------------------------------
# TSP Solver for Optimal Viewpoint Ordering
# -----------------------------------------------------------------------------
def solve_tsp(viewpoints):
    """
    Solve the TSP to obtain the optimal order for visiting the viewpoints.
    Returns ordered_viewpoints, ordered_marker_ids, and the total TSP distance.
    """
    num_points = len(viewpoints)
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(viewpoints[i][:2] - viewpoints[j][:2])
    if solve_tsp_dynamic_programming is not None:
        permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
    else:
        unvisited = list(range(num_points))
        current = unvisited.pop(0)
        permutation = [current]
        while unvisited:
            next_point = min(unvisited, key=lambda j: dist_matrix[current, j])
            unvisited.remove(next_point)
            permutation.append(next_point)
            current = next_point
        distance = sum(dist_matrix[permutation[i], permutation[i + 1]] for i in range(num_points - 1))
    ordered_viewpoints = viewpoints[permutation]
    ordered_marker_ids = [MARKER_IDS[i] for i in permutation]
    return ordered_viewpoints, ordered_marker_ids, distance

# -----------------------------------------------------------------------------
# Mission Start, Run, and End Functions
# -----------------------------------------------------------------------------
def drone_start(drone_interface: DroneMission) -> bool:
    """
    Start the mission: arm the drone, set offboard mode, and take off.
    """
    print('Start mission')
    print('Arm')
    success = drone_interface.arm()
    print(f'Arm success: {success}')
    print('Offboard')
    success = drone_interface.offboard()
    print(f'Offboard success: {success}')
    print('Take Off')
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f'Take Off success: {success}')
    return success

def drone_end(drone_interface: DroneMission) -> bool:
    """
    End the mission: land the drone and switch to manual control.
    """
    print('End mission')
    print('Land')
    success = drone_interface.land(speed=LAND_SPEED)
    print(f'Land success: {success}')
    print('Manual')
    success = drone_interface.manual()
    print(f'Manual success: {success}')
    return success

def drone_run(drone_interface: DroneMission):
    """
    Run the mission:
    - Solve the TSP to determine the optimal order for the viewpoints.
    - For each viewpoint, plan a leg using A* and smooth the path.
    - Follow the continuous trajectory to the viewpoint.
    - At the waypoint, adjust yaw until the expected ArUco marker is detected.
    - Log flight positions and compute metrics.
    - Finally, plot performance graphs.
    """
    drone_interface.mission_start_time = time.time()

    ordered_viewpoints, ordered_marker_ids, tsp_distance = solve_tsp(VIEWPOINTS)
    print("Optimal viewpoint order (TSP):")
    print(ordered_viewpoints)
    print("Expected marker IDs order:")
    print(ordered_marker_ids)

    total_distance = 0.0
    current_xy = np.array([0.0, 0.0])  # Assume start at origin.
    
    # Process each leg separately.
    for vp, marker_id in zip(ordered_viewpoints, ordered_marker_ids):
        target_xy = vp[:2]
        raw_path = astar(current_xy, target_xy, OBSTACLES)
        smooth_pts = smooth_path(raw_path)
        leg_trajectory = [np.array([pt[0], pt[1], TAKE_OFF_HEIGHT]) for pt in smooth_pts]
        print(f"Following leg to viewpoint at {vp} (expected marker {marker_id})")
        flight_distance = drone_interface.follow_trajectory(np.array(leg_trajectory), speed=SPEED)
        total_distance += flight_distance

        print(f"Waiting for marker {marker_id} detection...")
        if not drone_interface.wait_for_marker(marker_id, timeout=10):
            print(f"Marker {marker_id} not detected. Aborting mission.")
            return
        current_xy = target_xy

    # Optionally, return to starting position.
    raw_path = astar(current_xy, np.array([0.0, 0.0]), OBSTACLES)
    smooth_pts = smooth_path(raw_path)
    return_traj = [np.array([pt[0], pt[1], TAKE_OFF_HEIGHT]) for pt in smooth_pts]
    total_distance += drone_interface.follow_trajectory(np.array(return_traj), speed=SPEED)

    drone_interface.mission_end_time = time.time()

    total_time = drone_interface.get_flight_time()
    print(f"Total flight time: {total_time:.2f} seconds")
    print(f"Total flight distance: {total_distance:.2f} meters")
    print(f"Markers visited: {drone_interface.visited_markers}")

    # Plot performance graphs (here we plot the final leg only).
    plot_performance(drone_interface.logged_positions, np.vstack(return_traj))

# -----------------------------------------------------------------------------
# Performance Plotting Functions
# -----------------------------------------------------------------------------
def plot_performance(logged_positions, planned_trajectory):
    """
    Plot the executed trajectory vs. the planned trajectory.
    Provides both a 2D (XY) and a 3D trajectory comparison.
    """
    logged_positions = np.array(logged_positions)  # Columns: time, x, y, z
    planned_trajectory = np.array(planned_trajectory)  # Columns: x, y, z

    plt.figure()
    plt.plot(logged_positions[:, 1], logged_positions[:, 2], label="Executed Path")
    plt.plot(planned_trajectory[:, 0], planned_trajectory[:, 1], label="Planned Path", linestyle='--')
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("2D Trajectory Comparison")
    plt.legend()
    plt.grid(True)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(logged_positions[:, 1], logged_positions[:, 2], logged_positions[:, 3], label="Executed Path")
    ax.plot(planned_trajectory[:, 0], planned_trajectory[:, 1], planned_trajectory[:, 2],
            label="Planned Path", linestyle='--')
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("3D Trajectory Comparison")
    ax.legend()

    plt.show()

# -----------------------------------------------------------------------------
# Main Function with Updated Argument Parser
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Drone Mission Planning with A*, TSP, and ArUco Marker Detection')
    parser.add_argument('-n', '--namespace', type=str, default='drone0', help='Drone namespace')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('-s', '--scenario', type=str, required=True, help='Path to scenario file (e.g. scenarios/scenario1.yaml)')
    parser.add_argument('--use_sim_time', action='store_true', default=False, help='Use simulation time')
    args = parser.parse_args()

    # Load scenario file and override global parameters if available.
    scenario_file = args.scenario
    try:
        with open(scenario_file, 'r') as f:
            scenario_data = yaml.safe_load(f)
        if not isinstance(scenario_data, dict):
            print("Scenario file structure is not a dictionary. Using default scenario values.")
        else:
            if 'viewpoints' in scenario_data:
                if isinstance(scenario_data['viewpoints'], list):
                    VIEWPOINTS = np.array(scenario_data['viewpoints'])
                else:
                    print("Key 'viewpoints' is not a list. Using default viewpoints.")
            if 'obstacles' in scenario_data:
                if isinstance(scenario_data['obstacles'], list):
                    OBSTACLES = scenario_data['obstacles']
                    for obs in OBSTACLES:
                        if isinstance(obs, dict) and 'min' in obs and 'max' in obs:
                            obs['min'] = np.array(obs['min'])
                            obs['max'] = np.array(obs['max'])
                        else:
                            print("An obstacle entry is invalid. Skipping it.")
                else:
                    print("Key 'obstacles' is not a list. Using default obstacles.")
            if 'marker_ids' in scenario_data:
                if isinstance(scenario_data['marker_ids'], list):
                    MARKER_IDS = scenario_data['marker_ids']
                else:
                    print("Key 'marker_ids' is not a list. Using default marker IDs.")
    except Exception as e:
        print(f"Error loading scenario file: {e}")
        exit(1)

    print(f'Running mission for drone {args.namespace}')
    rclpy.init()
    drone = DroneMission(drone_id=args.namespace, verbose=args.verbose, use_sim_time=args.use_sim_time)

    if drone_start(drone):
        drone_run(drone)
    drone_end(drone)

    drone.shutdown()
    rclpy.shutdown()
    print("Clean exit")
