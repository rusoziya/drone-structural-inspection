#!/usr/bin/env python3
"""
Modified Mission Planning Script for Structural Inspection with Continuous Flight,
A* Path Planning, TSP Optimization, and ArUco Marker Detection.

Features:
- Plans a continuous, smooth trajectory between viewpoints using an optimized A* that
  first checks for a direct path and uses a coarser grid resolution.
- Simplifies (prunes) the computed path so that waypoints are further apart, allowing
  the drone to move more quickly.
- Reorders viewpoints optimally via a TSP solver.
- Uses scenario file information to set:
    • The drone's starting pose,
    • The positions and expected yaw orientations of the viewpoints (and expected marker IDs),
    • The cuboid obstacles to avoid.
- The drone always orients toward the next waypoint; when near a marker, its yaw is blended
  with the expected marker orientation.
- Logs flight positions and timestamps, computes total flight time and distance.
- Generates 2D and 3D performance plots comparing planned and executed trajectories.
- Designed so that different scenario files can be used without hardcoding.

Usage:
    python3 mission_scenario.py -s scenarios/scenario1.yaml

Assumptions:
1. The drone flies at a fixed altitude for planning in the XY plane.
2. Obstacles are specified as cuboids (given as center and dimensions) in the scenario file.
3. Each viewpoint is specified with position (x,y,z) and expected yaw (from key "w") and an associated marker id.
4. The TSP solver optimizes the route based on the XY positions.
5. The drone interface provides movement commands, including yaw adjustment.
"""

import argparse
import time
import math
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

# Try to import python_tsp; if not available, use a greedy fallback.
try:
    from python_tsp.exact import solve_tsp_dynamic_programming
except ImportError:
    solve_tsp_dynamic_programming = None

# -----------------------------------------------------------------------------
# Global Parameters and Default Values
# -----------------------------------------------------------------------------
TAKE_OFF_HEIGHT = 1.0      # Flight altitude (m)
TAKE_OFF_SPEED = 1.0       # Take off speed (m/s)
SPEED = 1.0                # Cruise speed (m/s)
LAND_SPEED = 0.5           # Landing speed (m/s)
# SLEEP_TIME is used to provide time for the drone to reach each setpoint.
# Reducing or removing sleep can speed up the mission but may cause control issues.
SLEEP_TIME = 0.05         
# Increase grid resolution to 0.1 m for faster planning.
GRID_RESOLUTION = 1     

# Default start pose (if not provided in scenario)
DRONE_START_POSE = np.array([0.0, 0.0, TAKE_OFF_HEIGHT])

# Default obstacles and viewpoints (to be loaded from scenario file)
OBSTACLES = []
VIEWPOINTS = []

# -----------------------------------------------------------------------------
# Helper Function: Check if Direct Path is Clear
# -----------------------------------------------------------------------------
def direct_path_clear(start, goal, obstacles, samples=10):
    """Returns True if the straight-line path from start to goal is free of obstacles."""
    for i in range(samples + 1):
        alpha = i / samples
        pt = start + alpha * (goal - start)
        for obs in obstacles:
            if (obs['min'][0] <= pt[0] <= obs['max'][0] and 
                obs['min'][1] <= pt[1] <= obs['max'][1]):
                return False
    return True

# -----------------------------------------------------------------------------
# Helper Function: Simplify (Prune) a Path
# -----------------------------------------------------------------------------
def simplify_path(path, min_distance=0.5):
    """
    Given a list of points, return a new list where consecutive points are at least
    min_distance apart. This prevents the drone from having too many very close waypoints.
    """
    if not path:
        return []
    simplified = [path[0]]
    for pt in path[1:]:
        if np.linalg.norm(pt - simplified[-1]) >= min_distance:
            simplified.append(pt)
    # Always include the last point.
    if np.linalg.norm(path[-1] - simplified[-1]) > 0:
        simplified.append(path[-1])
    return simplified

# -----------------------------------------------------------------------------
# DroneMission Class: Includes ArUco Detection and Yaw Adjustment
# -----------------------------------------------------------------------------
class DroneMission(DroneInterface):
    def __init__(self, drone_id, verbose=False, use_sim_time=False):
        super().__init__(drone_id, verbose, use_sim_time)
        self.visited_markers = set()
        self.logged_positions = []
        self.mission_start_time = None
        self.mission_end_time = None

        # Initialize CV Bridge and ArUco detection.
        self.br = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        # Change: Use DetectorParameters() per OpenCV version.
        self.aruco_params = aruco.DetectorParameters()
        self.marker_detected_flag = False
        self.detected_marker_id = None

        self.create_subscription(
            Image,
            "sensor_measurements/hd_camera/image_raw",
            self.img_callback,
            qos_profile_sensor_data
        )
        self.current_position = None

    def img_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            self.detected_marker_id = ids.flatten()[0]
            self.marker_detected_flag = True
            self.get_logger().info(f"Detected marker {self.detected_marker_id}.")
        else:
            self.marker_detected_flag = False
        aruco.drawDetectedMarkers(current_frame, corners, ids)
        cv2.imshow("Aruco Detection", current_frame)
        cv2.waitKey(1)

    def wait_for_marker(self, expected_marker_id, expected_yaw, timeout=10):
        """
        First, command the drone to set yaw to the expected value.
        Then, if the marker is not detected, perform a yaw sweep.
        """
        if self.current_position is not None:
            current_pos = [self.current_position.x, self.current_position.y, self.current_position.z]
            self.get_logger().info(f"Setting yaw to expected value: {expected_yaw:.0f}°")
            success = self.go_to.go_to_point_with_yaw(current_pos, angle=expected_yaw, speed=SPEED)
            time.sleep(1)
            if self.marker_detected_flag and self.detected_marker_id == expected_marker_id:
                self.get_logger().info(f"Expected marker {expected_marker_id} detected at expected yaw.")
                self.visited_markers.add(expected_marker_id)
                return True
        yaw_angles = np.linspace(0, 360, num=12, endpoint=False)
        start_time = time.time()
        while time.time() - start_time < timeout:
            for angle in yaw_angles:
                self.get_logger().info(f"Rotating to yaw angle: {angle:.0f}° for marker detection.")
                if self.current_position is None:
                    self.get_logger().warn("Current position not available!")
                    continue
                current_pos = [self.current_position.x, self.current_position.y, self.current_position.z]
                success = self.go_to.go_to_point_with_yaw(current_pos, angle=angle, speed=SPEED)
                if not success:
                    self.get_logger().error("Failed to adjust yaw!")
                    continue
                time.sleep(1)
                if self.marker_detected_flag and self.detected_marker_id == expected_marker_id:
                    self.get_logger().info(f"Expected marker {expected_marker_id} detected after yaw sweep.")
                    self.visited_markers.add(expected_marker_id)
                    return True
        self.get_logger().error(f"Timeout: Marker {expected_marker_id} not detected.")
        return False

    def follow_trajectory_with_yaw(self, trajectory, speed, expected_yaw, blend_threshold=1.0):
        """
        Follow a continuous trajectory while adjusting yaw.
        For each step, the desired yaw is computed as the heading toward the next waypoint.
        If the drone is within blend_threshold of the final waypoint, the desired yaw is blended
        with the expected_yaw so that the drone’s orientation gradually aligns with the marker.
        """
        total_distance = 0.0
        start_time = time.time()
        n_points = len(trajectory)
        for i in range(1, n_points):
            start_point = trajectory[i - 1]
            end_point = trajectory[i]
            seg_vec = end_point - start_point
            seg_dist = np.linalg.norm(seg_vec)
            total_distance += seg_dist
            num_steps = max(int(seg_dist / (speed * SLEEP_TIME)), 1)
            for step in range(num_steps):
                alpha = (step + 1) / num_steps
                target_point = start_point + alpha * seg_vec
                # Compute heading from current to target.
                dx = target_point[0] - start_point[0]
                dy = target_point[1] - start_point[1]
                heading = math.degrees(math.atan2(dy, dx))
                # Blend near the final waypoint.
                remaining = np.linalg.norm(trajectory[-1][:2] - target_point[:2])
                if remaining < blend_threshold:
                    blend = (blend_threshold - remaining) / blend_threshold
                    desired_yaw = (1 - blend) * heading + blend * expected_yaw
                else:
                    desired_yaw = heading
                success = self.go_to.go_to_point_with_yaw(target_point.tolist(), angle=desired_yaw, speed=speed)
                if not success:
                    self.get_logger().error("Failed to move to target point.")
                    return total_distance
                current_time = time.time() - start_time
                self.logged_positions.append((current_time, target_point[0], target_point[1], target_point[2]))
                time.sleep(SLEEP_TIME)
            self.current_position = type("pos", (), {"x": target_point[0],
                                                       "y": target_point[1],
                                                       "z": target_point[2]})
        return total_distance

    def get_flight_time(self):
        if self.mission_start_time and self.mission_end_time:
            return self.mission_end_time - self.mission_start_time
        return None

# -----------------------------------------------------------------------------
# A* Path Planning Function (2D)
# -----------------------------------------------------------------------------
def astar(start, goal, obstacles, grid_resolution=GRID_RESOLUTION):
    """
    Compute an A* path on a 2D grid from start to goal while avoiding obstacles.
    If the direct path is clear, returns [start, goal].
    Otherwise, uses grid-based A*.
    """
    if direct_path_clear(start, goal, obstacles):
        return [start, goal]

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
            # Simplify the path so that waypoints are not too close.
            return simplify_path(path, min_distance=0.5)
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
# Path Simplification Function (New)
# -----------------------------------------------------------------------------
def simplify_path(path, min_distance=0.5):
    """
    Given a list of 2D points (numpy arrays), returns a new list where consecutive points
    are at least min_distance apart. This reduces the number of waypoints.
    """
    if not path:
        return []
    simplified = [path[0]]
    for pt in path[1:]:
        if np.linalg.norm(pt - simplified[-1]) >= min_distance:
            simplified.append(pt)
    if np.linalg.norm(path[-1] - simplified[-1]) > 0:
        simplified.append(path[-1])
    return simplified

# -----------------------------------------------------------------------------
# Spline Smoothing Function
# -----------------------------------------------------------------------------
def smooth_path(path):
    path = np.array(path)
    m = len(path)
    if m < 2:
        return path
    k = min(3, m - 1)
    tck, u = splprep([path[:, 0], path[:, 1]], s=0, k=k)
    u_fine = np.linspace(0, 1, num=100)
    x_fine, y_fine = splev(u_fine, tck)
    smooth = [np.array([x, y]) for x, y in zip(x_fine, y_fine)]
    return smooth

# -----------------------------------------------------------------------------
# TSP Solver for Viewpoints
# -----------------------------------------------------------------------------
def solve_tsp(viewpoints):
    positions = np.array([vp["position"] for vp in viewpoints])
    num_points = len(positions)
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(positions[i][:2] - positions[j][:2])
    if solve_tsp_dynamic_programming is not None:
        permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
    else:
        unvisited = list(range(num_points))
        current = unvisited.pop(0)
        permutation = [current]
        while unvisited:
            next_pt = min(unvisited, key=lambda j: dist_matrix[current, j])
            unvisited.remove(next_pt)
            permutation.append(next_pt)
            current = next_pt
        distance = sum(dist_matrix[permutation[i], permutation[i+1]] for i in range(num_points-1))
    ordered_viewpoints = [viewpoints[i] for i in permutation]
    return ordered_viewpoints, distance

# -----------------------------------------------------------------------------
# Mission Start/End Functions
# -----------------------------------------------------------------------------
def drone_start(drone_interface: DroneMission) -> bool:
    print("Start mission")
    print("Arm")
    success = drone_interface.arm()
    print(f"Arm success: {success}")
    print("Offboard")
    success = drone_interface.offboard()
    print(f"Offboard success: {success}")
    print("Take Off")
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f"Take Off success: {success}")
    return success

def drone_end(drone_interface: DroneMission) -> bool:
    print("End mission")
    print("Land")
    success = drone_interface.land(speed=LAND_SPEED)
    print(f"Land success: {success}")
    print("Manual")
    success = drone_interface.manual()
    print(f"Manual success: {success}")
    return success

# -----------------------------------------------------------------------------
# Main Mission Run Function
# -----------------------------------------------------------------------------
def drone_run(drone_interface: DroneMission):
    drone_interface.mission_start_time = time.time()
    ordered_viewpoints, tsp_distance = solve_tsp(VIEWPOINTS)
    print("Optimal viewpoint order (TSP):")
    for vp in ordered_viewpoints:
        print(f"Position: {vp['position']}, Expected yaw: {vp['yaw']}, Marker ID: {vp['marker_id']}")
    total_distance = 0.0
    current_xy = DRONE_START_POSE[:2]
    for vp in ordered_viewpoints:
        target_xy = vp["position"][:2]
        raw_path = astar(current_xy, target_xy, OBSTACLES)
        smooth_pts = smooth_path(raw_path)
        leg_traj = [np.array([pt[0], pt[1], TAKE_OFF_HEIGHT]) for pt in smooth_pts]
        print(f"Following leg to viewpoint at {vp['position']} (expected marker {vp['marker_id']}, yaw {vp['yaw']})")
        flight_distance = drone_interface.follow_trajectory_with_yaw(np.array(leg_traj), speed=SPEED, expected_yaw=vp["yaw"])
        total_distance += flight_distance
        if not drone_interface.wait_for_marker(vp["marker_id"], vp["yaw"], timeout=10):
            print(f"Marker {vp['marker_id']} not detected. Aborting mission.")
            return
        current_xy = target_xy
    raw_path = astar(current_xy, DRONE_START_POSE[:2], OBSTACLES)
    smooth_pts = smooth_path(raw_path)
    return_traj = [np.array([pt[0], pt[1], TAKE_OFF_HEIGHT]) for pt in smooth_pts]
    total_distance += drone_interface.follow_trajectory_with_yaw(np.array(return_traj), speed=SPEED, expected_yaw=0.0)
    drone_interface.mission_end_time = time.time()
    total_time = drone_interface.get_flight_time()
    print(f"Total flight time: {total_time:.2f} seconds")
    print(f"Total flight distance: {total_distance:.2f} meters")
    print(f"Markers visited: {drone_interface.visited_markers}")
    plot_performance(np.array(drone_interface.logged_positions), np.vstack(return_traj))

# -----------------------------------------------------------------------------
# Performance Plotting Function
# -----------------------------------------------------------------------------
def plot_performance(logged_positions, planned_trajectory):
    logged_positions = np.array(logged_positions)  # time, x, y, z
    planned_trajectory = np.array(planned_trajectory)  # x, y, z
    plt.figure()
    plt.plot(logged_positions[:,1], logged_positions[:,2], label="Executed Path")
    plt.plot(planned_trajectory[:,0], planned_trajectory[:,1], label="Planned Path", linestyle="--")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Trajectory Comparison")
    plt.legend()
    plt.grid(True)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(logged_positions[:,1], logged_positions[:,2], logged_positions[:,3], label="Executed Path")
    ax.plot(planned_trajectory[:,0], planned_trajectory[:,1], planned_trajectory[:,2], label="Planned Path", linestyle="--")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory Comparison")
    ax.legend()
    plt.show()

# -----------------------------------------------------------------------------
# Main Function with Scenario File Parsing
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drone Mission Planning with A*, TSP, and ArUco Marker Detection")
    parser.add_argument("-n", "--namespace", type=str, default="drone0", help="Drone namespace")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose output")
    parser.add_argument("-s", "--scenario", type=str, required=True, help="Path to scenario file (e.g. scenarios/scenario1.yaml)")
    parser.add_argument("--use_sim_time", action="store_true", default=False, help="Use simulation time")
    args = parser.parse_args()

    try:
        with open(args.scenario, "r") as f:
            scenario_data = yaml.safe_load(f)
        if not isinstance(scenario_data, dict):
            print("Scenario file structure is not a dictionary. Using default values.")
        else:
            if "drone_start_pose" in scenario_data:
                dsp = scenario_data["drone_start_pose"]
                DRONE_START_POSE = np.array([float(dsp.get("x", 0.0)), float(dsp.get("y", 0.0)), TAKE_OFF_HEIGHT])
            if "obstacles" in scenario_data:
                obs_data = scenario_data["obstacles"]
                if isinstance(obs_data, dict):
                    OBSTACLES = []
                    for key, val in obs_data.items():
                        cx = float(val.get("x", 0))
                        cy = float(val.get("y", 0))
                        w_val = float(val.get("w", 0))
                        d_val = float(val.get("d", 0))
                        min_pt = np.array([cx - w_val/2, cy - d_val/2])
                        max_pt = np.array([cx + w_val/2, cy + d_val/2])
                        OBSTACLES.append({"min": min_pt, "max": max_pt})
                elif isinstance(obs_data, list):
                    OBSTACLES = []
                    for obs in obs_data:
                        if isinstance(obs, dict) and "min" in obs and "max" in obs:
                            obs["min"] = np.array(obs["min"])
                            obs["max"] = np.array(obs["max"])
                            OBSTACLES.append(obs)
                else:
                    print("Key 'obstacles' is not in an expected format. Using default obstacles.")
            if "viewpoint_poses" in scenario_data:
                vp_data = scenario_data["viewpoint_poses"]
                VIEWPOINTS = []
                if isinstance(vp_data, dict):
                    for key in sorted(vp_data.keys(), key=lambda k: int(k)):
                        vp = vp_data[key]
                        pos = np.array([float(vp.get("x", 0.0)), float(vp.get("y", 0.0)), float(vp.get("z", TAKE_OFF_HEIGHT))])
                        yaw = float(vp.get("w", 0.0))
                        marker_id = int(key)
                        VIEWPOINTS.append({"position": pos, "yaw": yaw, "marker_id": marker_id})
                elif isinstance(vp_data, list):
                    for i, vp in enumerate(vp_data):
                        pos = np.array([float(vp.get("x", 0.0)), float(vp.get("y", 0.0)), float(vp.get("z", TAKE_OFF_HEIGHT))])
                        yaw = float(vp.get("w", 0.0))
                        marker_id = i+1
                        VIEWPOINTS.append({"position": pos, "yaw": yaw, "marker_id": marker_id})
                else:
                    print("Key 'viewpoint_poses' is not in an expected format. Using default viewpoints.")
            else:
                print("No viewpoint poses found in scenario file. Using default viewpoints.")
    except Exception as e:
        print(f"Error loading scenario file: {e}")
        exit(1)

    print(f"Running mission for drone {args.namespace}")
    rclpy.init()
    drone = DroneMission(drone_id=args.namespace, verbose=args.verbose, use_sim_time=args.use_sim_time)
    drone.current_position = type("pos", (), {"x": DRONE_START_POSE[0],
                                               "y": DRONE_START_POSE[1],
                                               "z": DRONE_START_POSE[2]})
    
    if drone_start(drone):
        drone_run(drone)
    drone_end(drone)
    
    drone.shutdown()
    rclpy.shutdown()
    print("Clean exit")
