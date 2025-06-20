#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Modified Mission Script for Structural Inspection Path Planning with ArUco
#
# This script integrates a TSP local search 3‑opt solver to determine the optimal visitation order
# of viewpoints and uses an OMPL‑based RRT‑Connect algorithm to plan collision‑free
# paths between waypoints while avoiding cuboid obstacles. In addition, the planner
# verifies that the continuous path (using a fine interpolation) is free of collisions.
# If any segment is in collision, it automatically subdivides the segment by inserting
# intermediate waypoints and replanning.
#
# A new B‑Spline smoothing function is integrated here that smooths the continuous path
# returned by RRT‑Connect. Instead of sending individual waypoint commands, the dense,
# smooth path is converted to a ROS Path message and sent as a single command to the
# TrajectoryGenerationModule. This module handles true continuous trajectory following.
#
# *CHANGE*: After following the path in "path facing" mode, we make a short final
# command that orients the drone to the final yaw at the last point.
#
# Assumptions:
# 1. Obstacles are cuboids, axis‑aligned, defined by center (x,y,z) and dimensions (d, w, h).
# 2. Viewpoint poses are specified in the scenario YAML with fields x, y, z and w.
# 3. The drone starting pose is provided and its z value is raised to TAKE_OFF_HEIGHT if needed.
# 4. TSP ordering is computed using Euclidean distances.
# 5. RRT‑Connect computes a continuous 3D path which is then smoothed.
# 6. The drone’s yaw is handled by the trajectory generation module.
# 7. A small safety margin is added around obstacles.
# ------------------------------------------------------------------------------

# ------------------------
# Configuration (Modifiable Parameters)
# ------------------------
TAKE_OFF_HEIGHT = 1.0      # Height (m) at takeoff 
TAKE_OFF_SPEED = 1.0       # Takeoff speed (m/s)
SLEEP_TIME = 0.05          # Delay between commands (s)
SPEED = 1.0                # Nominal flight speed (m/s)
LAND_SPEED = 0.5           # Landing speed (m/s)

SAFETY_MARGIN = 0.8        # Additional safety margin (m) around obstacles

COLLISION_CHECK_RESOLUTION = 0.5  # Step size (m) for collision checking
MAX_RECURSION_DEPTH = 100000      # Maximum subdivisions if a segment is in collision

PLANNING_TIME_LIMIT = 25.0        # Planning time limit per segment (s)
PLANNER_RANGE_SMALL = 1.0         # Fixed planner range used for all segments
PLANNER_RANGE_LARGE = 1.5         # (Currently unused)

# ------------------------
# Global Metrics and Data Logging Variables
# ------------------------
fallback_count = 0
segment_planning_times = []  # list of planning times per segment
segment_lengths = []         # list of segment lengths
global_rrt_tree_data = []    # list to store RRT tree data for each segment

# ------------------------
# Imports and Setup
# ------------------------
import argparse
import time
import math
import yaml
import logging
import numpy as np
import rclpy
import random
import os
import threading
import json

from as2_python_api.drone_interface import DroneInterface

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ArUco detection
import cv2
import cv2.aruco as aruco
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# OMPL
from ompl import base as ob
from ompl import geometric as og

# TSP, B‑spline
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm
from scipy.interpolate import splprep, splev

# ROS Path message
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# TrajectoryGenerationModule import
from as2_python_api.modules.trajectory_generation_module import TrajectoryGenerationModule


def create_path_msg(points, frame_id="earth"):
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    for point in points:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = point[2]
        path_msg.poses.append(pose)
    return path_msg

# Helper functions for geometry/collision, etc.
def load_scenario(scenario_file):
    with open(scenario_file, 'r') as f:
        scenario = yaml.safe_load(f)
    return scenario

def compute_euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def build_distance_matrix(points):
    n = len(points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = compute_euclidean_distance(points[i], points[j])
    return matrix

def interpolate_angle(a, b, t):
    diff = (b - a + math.pi) % (2*math.pi) - math.pi
    return a + diff * t

def is_state_valid_cuboids(state, obstacles):
    x, y, z = state[0], state[1], state[2]
    for obs in obstacles:
        ox, oy, oz = obs["x"], obs["y"], obs["z"]
        dx, dy, dz = obs["d"], obs["w"], obs["h"]
        hx, hy, hz = dx/2.0, dy/2.0, dz/2.0
        hx += SAFETY_MARGIN
        hy += SAFETY_MARGIN
        hz += SAFETY_MARGIN
        if (ox-hx <= x <= ox+hx and
            oy-hy <= y <= oy+hy and
            oz-hz <= z <= oz+hz):
            return False
    return True

def is_path_collision_free(path, obstacles, resolution=COLLISION_CHECK_RESOLUTION):
    if len(path) < 2:
        return True
    for i in range(len(path)-1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        seg_length = compute_euclidean_distance(p1, p2)
        steps = max(int(seg_length/resolution), 1)
        for j in range(steps+1):
            t_val = j/steps
            interp = p1 + t_val*(p2-p1)
            if not is_state_valid_cuboids(interp, obstacles):
                return False
    return True

def load_world_yaml(world_file=None):
    if world_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        world_file = os.path.join(script_dir, "config_sim", "world", "world.yaml")
    with open(world_file, 'r') as f:
        world_data = yaml.safe_load(f)
    return world_data

def get_marker_info(world_data):
    markers = []
    for obj in world_data.get("objects", []):
        mt = obj.get("model_type", "")
        if mt.startswith("aruco_id") and "_marker" in mt:
            start = len("aruco_id")
            end = mt.index("_marker")
            try:
                marker_id = int(mt[start:end])
            except Exception:
                continue
            marker = {
                "marker_id": marker_id,
                "model_name": obj.get("model_name"),
                "xyz": obj.get("xyz"),
                "rpy": obj.get("rpy")
            }
            markers.append(marker)
    return markers

def get_expected_marker_id_for_viewpoint(viewpoint, markers_list):
    vp_point = np.array([viewpoint["x"], viewpoint["y"], viewpoint["z"]])
    best_id = None
    best_dist = float("inf")
    for marker in markers_list:
        marker_point = np.array(marker["xyz"])
        dist = np.linalg.norm(vp_point - marker_point)
        if dist < best_dist:
            best_dist = dist
            best_id = marker["marker_id"]
    return best_id

def assign_viewpoints_to_markers(viewpoints, markers_list):
    n_view = len(viewpoints)
    n_mark = len(markers_list)
    if n_mark < n_view:
        print("[WARNING] Fewer markers than viewpoints!")
    cost_matrix = np.zeros((n_view, n_mark))
    for i in range(n_view):
        vx, vy, vz = viewpoints[i]
        for j in range(n_mark):
            mx, my, mz = markers_list[j]["xyz"]
            cost_matrix[i, j] = math.sqrt((vx-mx)**2 + (vy-my)**2 + (vz-mz)**2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned_marker_ids = [None]*n_view
    for i in range(len(row_ind)):
        assigned_marker_ids[row_ind[i]] = markers_list[col_ind[i]]["marker_id"]
    return assigned_marker_ids

def apply_3opt_move(tour, i, j, k, distance_matrix):
    A = tour[:i]
    B = tour[i:j]
    C = tour[j:k]
    D = tour[k:]
    candidates = [
        A+B+C+D,
        A+B[::-1]+C+D,
        A+B+C[::-1]+D,
        A+B[::-1]+C[::-1]+D,
        A+C+B+D,
        A+C+B[::-1]+D,
        A+C[::-1]+B+D,
        A+C[::-1]+B[::-1]+D
    ]
    best_candidate = candidates[0]
    best_cost = sum(distance_matrix[best_candidate[i]][best_candidate[i+1]] for i in range(len(best_candidate)-1))
    for candidate in candidates[1:]:
        cost = sum(distance_matrix[candidate[i]][candidate[i+1]] for i in range(len(candidate)-1))
        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate
    return best_candidate

def solve_tsp_3opt(distance_matrix, max_iterations=1000):
    n = len(distance_matrix)
    # initial solution: nearest neighbor from 0
    tour = [0]
    remaining = list(range(1, n))
    current = 0
    while remaining:
        next_city = min(remaining, key=lambda x: distance_matrix[current][x])
        tour.append(next_city)
        remaining.remove(next_city)
        current = next_city

    def tour_length(t):
        return sum(distance_matrix[t[i]][t[i+1]] for i in range(len(t)-1))

    best_tour = tour[:]
    best_length = tour_length(best_tour)

    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    new_tour = apply_3opt_move(best_tour, i, j, k, distance_matrix)
                    new_length = tour_length(new_tour)
                    if new_length < best_length:
                        best_tour = new_tour[:]
                        best_length = new_length
                        improved = True
        iteration += 1

    return best_tour, best_length

def smooth_path_bspline(path, degree=3, smoothing=0):
    points = np.array(path)
    if len(points) < degree + 1:
        return path

    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(dists)))
    t = cumulative / cumulative[-1]

    angles = []
    for i in range(1, len(points)-1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_angle = np.dot(v1, v2) / (norm1*norm2)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)

    avg_curvature = np.mean(angles) if angles else 0
    num_points = int(10 + 10*avg_curvature)
    num_points = max(num_points, len(points))

    tck, u = splprep(points.T, u=t, k=min(degree, len(points)-1), s=smoothing)
    u_fine = np.linspace(0, 1, num_points)
    x_new, y_new, z_new = splev(u_fine, tck)
    smoothed_path = np.vstack((x_new, y_new, z_new)).T.tolist()

    smoothed_path[0] = path[0]
    smoothed_path[-1] = path[-1]
    return smoothed_path

def compute_path_smoothness(path):
    if len(path) < 3:
        return 0.0
    smoothness = 0.0
    for i in range(1, len(path)-1):
        p_prev, p_curr, p_next = np.array(path[i-1]), np.array(path[i]), np.array(path[i+1])
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        smoothness += abs(interpolate_angle(angle1, angle2, 1.0))
    return smoothness

def plan_rrtconnect(start, goal, obstacles, bounds,
                    planner_range=PLANNER_RANGE_SMALL,
                    planning_time_limit=PLANNING_TIME_LIMIT):
    start_time = time.time()

    space = ob.RealVectorStateSpace(3)
    real_bounds = ob.RealVectorBounds(3)
    real_bounds.setLow(0, bounds['low'][0])
    real_bounds.setHigh(0, bounds['high'][0])
    real_bounds.setLow(1, bounds['low'][1])
    real_bounds.setHigh(1, bounds['high'][1])
    real_bounds.setLow(2, bounds['low'][2])
    real_bounds.setHigh(2, bounds['high'][2])
    space.setBounds(real_bounds)

    si = ob.SpaceInformation(space)
    def is_state_valid_fn(state):
        return is_state_valid_cuboids(state, obstacles)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid_fn))
    si.setup()

    start_state = ob.State(space)
    start_state[0], start_state[1], start_state[2] = start[0], start[1], start[2]
    goal_state = ob.State(space)
    goal_state[0], goal_state[1], goal_state[2] = goal[0], goal[1], goal[2]

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)

    planner = og.RRTConnect(si, True)
    planner.setRange(planner_range)
    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(planning_time_limit)
    if solved:
        planner_data = ob.PlannerData(si)
        planner.getPlannerData(planner_data)
        tree_nodes = []
        tree_edges = []
        num_vertices = planner_data.numVertices()
        for i in range(num_vertices):
            vertex = planner_data.getVertex(i)
            st = vertex.getState()
            point = [st[0], st[1], st[2]]
            tree_nodes.append(point)
            if i > 0:
                tree_edges.append((tree_nodes[0], point))
        global_rrt_tree_data.append({"nodes": tree_nodes, "edges": tree_edges})

        path = pdef.getSolutionPath()
        path_simplifier = og.PathSimplifier(si)
        path_simplifier.simplifyMax(path)

        path_states = []
        for i in range(path.getStateCount()):
            st = path.getState(i)
            point = [st[0], st[1], st[2]]
            path_states.append(point)

        planning_time = time.time() - start_time
        return path_states, planning_time
    else:
        planning_time = time.time() - start_time
        return None, planning_time

def plan_segment(start, goal, obstacles, bounds,
                 planner_range, planning_time_limit,
                 recursion_depth=MAX_RECURSION_DEPTH):
    path, ptime = plan_rrtconnect(start, goal, obstacles, bounds,
                                  planner_range, planning_time_limit)
    if path is None:
        return None, ptime

    if is_path_collision_free(path, obstacles):
        return path, ptime
    else:
        if recursion_depth <= 0:
            print("Max recursion reached; using current path.")
            return path, ptime

        mid_point = [(s+g)/2.0 for s, g in zip(start, goal)]
        if not is_state_valid_cuboids(mid_point, obstacles):
            print("Midpoint is invalid; aborting segment.")
            return None, ptime

        print("Subdividing segment using midpoint: {}".format(mid_point))
        path1, time1 = plan_segment(start, mid_point, obstacles, bounds,
                                    planner_range, planning_time_limit,
                                    recursion_depth-1)
        path2, time2 = plan_segment(mid_point, goal, obstacles, bounds,
                                    planner_range, planning_time_limit,
                                    recursion_depth-1)
        if path1 is None or path2 is None:
            return None, ptime + time1 + time2

        combined = path1[:-1] + path2
        return combined, ptime + time1 + time2

def plot_paths(planned_paths, waypoints, ground_truth, obstacles=None, markers_list=None):
    pass

def plot_costs(planning_times, path_lengths):
    pass

def plot_rrt_tree(rrt_tree_data, waypoints=None, obstacles=None, planned_paths=None):
    pass

def plot_rrt_tree_3d(rrt_tree_data, waypoints=None, obstacles=None, planned_paths=None):
    pass

def log_metrics_to_file(metrics, filename="metrics.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics logged to {file_path}")

class SingleNodeDroneMission(DroneInterface):
    def __init__(self, drone_id, use_sim_time=True, verbose=False):
        super().__init__(drone_id=drone_id, use_sim_time=use_sim_time, verbose=verbose)

        self.br = CvBridge()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.aruco_params = aruco.DetectorParameters()
        self.detected_marker_id = None
        self.fallback_count = 0
        self.last_marker_print_time = 0

        self.subscription = self.create_subscription(
            Image,
            "sensor_measurements/hd_camera/image_raw",
            self.image_callback,
            10
        )

        # Attach the trajectory generation module
        self.trajectory_generation = TrajectoryGenerationModule(self)

    def image_callback(self, msg):
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None and len(ids) > 0:
            self.detected_marker_id = ids.flatten()[0]
            current_time = time.time()
            if current_time - self.last_marker_print_time >= 2.0:
                print(f"[DEBUG] image_callback: Detected marker ids: {ids.flatten()}")
                self.last_marker_print_time = current_time
        else:
            self.detected_marker_id = None

        aruco.drawDetectedMarkers(cv_image, corners, ids)
        cv2.imshow("ArUco Detection", cv_image)
        cv2.waitKey(100)

    def wait_for_expected_marker(self, expected_marker_id, timeout=5.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            if self.detected_marker_id is not None:
                print(f"[DEBUG] Detected marker id: {self.detected_marker_id} (expected: {expected_marker_id})")
                if self.detected_marker_id == expected_marker_id:
                    detected = self.detected_marker_id
                    self.detected_marker_id = None
                    return detected
                else:
                    print(f"[DEBUG] Marker id {self.detected_marker_id} != expected {expected_marker_id}")
                    self.detected_marker_id = None
        print("Timeout: ignoring marker detection failure and continuing mission.")
        self.fallback_count += 1
        return None

    def wait_for_marker(self, timeout=1000.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            if self.detected_marker_id is not None:
                return self.detected_marker_id
        return None

def verify_marker_and_adjust(drone_interface: SingleNodeDroneMission,
                             current_point, target_point, expected_yaw, expected_marker_id,
                             marker_ids_set, timeout=5.0):
    if expected_marker_id not in marker_ids_set:
        print(f"Expected marker id {expected_marker_id} not found.")
        return False
    print(f"Waiting for marker id {expected_marker_id} at waypoint...")
    detected_id = drone_interface.wait_for_expected_marker(expected_marker_id, timeout=timeout)
    if detected_id is None:
        print("Marker verification fallback triggered; continuing anyway.")
        return True
    print("Marker verification successful.")
    return True

def drone_start(drone_interface: SingleNodeDroneMission) -> bool:
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

def drone_end(drone_interface: SingleNodeDroneMission) -> bool:
    print("End mission")
    print("Manual")
    success = drone_interface.manual()
    print(f"Manual success: {success}")
    return success

def drone_run(drone_interface: SingleNodeDroneMission, scenario: dict):
    """
    Runs the mission:
      - Plans TSP order
      - Plans RRT segments
      - Sends path in PATH_FACING mode
      - Then at final point, turns to the viewpoint orientation
    """
    print("Run mission with RRT‑Connect, TSP (3‑opt), and trajectory generation.")
    start_pose = scenario.get("drone_start_pose", {"x": 0.0, "y": 0.0, "z": 0.0})
    if start_pose["z"] < TAKE_OFF_HEIGHT:
        start_pose["z"] = TAKE_OFF_HEIGHT

    obstacles = [obs for key, obs in scenario.get("obstacles", {}).items()]
    world_data = load_world_yaml("config_sim/world/world.yaml")
    markers_list = get_marker_info(world_data)
    marker_ids_set = {m["marker_id"] for m in markers_list}

    viewpoint_dict = scenario.get("viewpoint_poses", {})
    view_keys = sorted(viewpoint_dict.keys(), key=lambda x: int(x))
    viewpoints = []
    marker_yaws = []
    for key in view_keys:
        vp = viewpoint_dict[key]
        viewpoints.append([vp["x"], vp["y"], vp["z"]])
        marker_yaws.append(vp["w"])

    assigned_marker_ids = assign_viewpoints_to_markers(viewpoints, markers_list)

    points = [[start_pose["x"], start_pose["y"], start_pose["z"]]] + viewpoints
    yaw_list = [None] + marker_yaws
    expected_marker_ids = [None] + assigned_marker_ids

    distance_matrix = build_distance_matrix(points)
    permutation, tsp_distance = solve_tsp_3opt(distance_matrix)
    print(f"TSP order (3‑opt): {permutation}, total straight-line distance: {tsp_distance:.2f}")

    ordered_points = [points[i] for i in permutation]
    ordered_yaws = [yaw_list[i] for i in permutation]
    ordered_expected_marker_ids = [expected_marker_ids[i] for i in permutation]

    ground_truth = ordered_points
    bounds = {"low": [-10, -10, 0], "high": [10, 10, 5]}

    planned_paths = []
    total_planning_time = 0.0
    total_path_length = 0.0
    fixed_planner_range = PLANNER_RANGE_SMALL

    for i in range(len(ordered_points) - 1):
        seg_start = ordered_points[i]
        seg_goal = ordered_points[i+1]

        # The final yaw we want at the final point
        dest_yaw = (ordered_yaws[i+1]
                    if ordered_yaws[i+1] is not None
                    else math.atan2(seg_goal[1]-seg_start[1], seg_goal[0]-seg_start[0]))
        print(f"Planning segment from {seg_start} to {seg_goal}, final yaw: {dest_yaw:.2f} rad")

        path, planning_time = plan_segment(seg_start, seg_goal, obstacles, bounds,
                                           fixed_planner_range, PLANNING_TIME_LIMIT)
        if path is None:
            print("No solution found for segment, aborting mission.")
            return False, None, None

        print(f"Segment planned in {planning_time:.2f} s with {len(path)} states.")
        total_planning_time += planning_time

        # B-Spline smoothing
        smoothed_path = smooth_path_bspline(path)
        planned_paths.append(smoothed_path)
        seg_length = sum(compute_euclidean_distance(smoothed_path[j], smoothed_path[j+1])
                         for j in range(len(smoothed_path)-1))
        total_path_length += seg_length
        segment_planning_times.append(planning_time)
        segment_lengths.append(seg_length)

        # 1) Follow the path in PATH_FACING mode
        path_msg = create_path_msg(smoothed_path, frame_id="earth")
        print("Following path in PATH_FACING mode (ignoring final yaw for now).")
        success_traj = drone_interface.trajectory_generation.traj_generation_with_path_facing(
            path=path_msg,
            speed=SPEED,
            frame_id="earth"
        )
        if not success_traj:
            print("Trajectory generation (path_facing) failed.")
            return False, None, None

        time.sleep(1.0)

        # 2) Once the path is done, we do a short final orientation fix at the last point
        final_point = smoothed_path[-1]
        # Build a "single-point path" for rotating in place to final yaw
        final_path_msg = create_path_msg([final_point], frame_id="earth")
        print(f"Now turning to final orientation (yaw={dest_yaw:.2f} rad) at last point.")
        success_traj_yaw = drone_interface.trajectory_generation.traj_generation_with_yaw(
            path=final_path_msg,
            speed=SPEED,
            angle=dest_yaw,
            frame_id="earth"
        )
        if not success_traj_yaw:
            print("Final yaw orientation command failed.")
            return False, None, None

        time.sleep(1.0)

        # Marker verification
        current_point = smoothed_path[-1]
        expected_marker_id = ordered_expected_marker_ids[i+1]
        if expected_marker_id is not None:
            print(f"Verifying marker {expected_marker_id} at final waypoint {current_point}")
            if not verify_marker_and_adjust(drone_interface,
                                            current_point=current_point,
                                            target_point=seg_goal,
                                            expected_yaw=dest_yaw,
                                            expected_marker_id=expected_marker_id,
                                            marker_ids_set=marker_ids_set,
                                            timeout=5.0):
                print("Marker verification failed; aborting mission.")
                return False, None, None
        else:
            print("No expected marker for this segment; continuing.")

    # --- Return path planning ---
    return_start = ordered_points[0]
    last_waypoint = ordered_points[-1]
    print("Planning return path from last waypoint to starting position")
    path, planning_time = plan_segment(last_waypoint, return_start, obstacles, bounds,
                                       fixed_planner_range, PLANNING_TIME_LIMIT)
    if path is None:
        print("No solution found for return segment, aborting mission.")
        return False, None, None
    print(f"Return segment planned in {planning_time:.2f} s with {len(path)} states.")
    total_planning_time += planning_time

    smoothed_path = smooth_path_bspline(path)
    planned_paths.append(smoothed_path)
    seg_length = sum(compute_euclidean_distance(smoothed_path[j], smoothed_path[j+1])
                     for j in range(len(smoothed_path)-1))
    total_path_length += seg_length
    segment_planning_times.append(planning_time)
    segment_lengths.append(seg_length)

    # Return path in PATH_FACING mode
    path_msg = create_path_msg(smoothed_path, frame_id="earth")
    print("Following return path in PATH_FACING mode.")
    success_traj = drone_interface.trajectory_generation.traj_generation_with_path_facing(
        path=path_msg,
        speed=SPEED,
        frame_id="earth"
    )
    if not success_traj:
        print("Trajectory generation for return path failed.")
        return False, None, None

    time.sleep(1.0)

    # After finishing the path, do not need a special yaw for the home point (or we could).
    # For consistency, you could do the same final orientation trick if you had a final yaw.

    print("Return to starting position complete. Landing now.")
    land_success = drone_interface.land(speed=LAND_SPEED)
    print(f"Landing success: {land_success}")

    total_ground_truth_path_length = sum(
        compute_euclidean_distance(ordered_points[i], ordered_points[i+1])
        for i in range(len(ordered_points)-1)
    )
    smoothness_list = [compute_path_smoothness(path) for path in planned_paths]
    energy_estimate_list = [segment_lengths[i] + 0.5*smoothness_list[i]
                            for i in range(len(segment_lengths))]

    metrics = {
        "total_planning_time": total_planning_time,
        "total_path_length": total_path_length,
        "segment_planning_times": segment_planning_times,
        "segment_lengths": segment_lengths,
        "fallback_count": drone_interface.fallback_count,
        "energy_estimate": energy_estimate_list,
        "path_smoothness": smoothness_list,
        "shortest_possible_path_length": tsp_distance,
        "total_ground_truth_path_length": total_ground_truth_path_length
    }
    log_metrics_to_file(metrics)

    plot_data = {
        "planned_paths": planned_paths,
        "ordered_points": ordered_points,
        "ground_truth": ordered_points,
        "segment_planning_times": segment_planning_times,
        "segment_lengths": segment_lengths,
        "obstacles": obstacles,
        "markers_list": markers_list,
        "fallback_count": drone_interface.fallback_count
    }

    return True, plot_data, metrics

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single drone mission with RRT‑Connect, TSP (3‑opt), obstacle avoidance, and trajectory generation enabled'
    )
    parser.add_argument('-n', '--namespace', type=str, default='drone0',
                        help='ID of the drone to be used in the mission')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--scenario', type=str, required=True,
                        help='Path to scenario YAML file')
    parser.add_argument('-t', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')

    args = parser.parse_args()
    drone_namespace = args.namespace
    verbosity = args.verbose
    scenario_file = args.scenario
    use_sim_time = args.use_sim_time

    logging.basicConfig(level=logging.INFO)
    print(f'Running mission for drone {drone_namespace} using scenario {scenario_file}')

    scenario = load_scenario(scenario_file)

    rclpy.init()
    uav = SingleNodeDroneMission(drone_id=drone_namespace,
                                 use_sim_time=use_sim_time,
                                 verbose=verbosity)

    spinner_thread = threading.Thread(target=rclpy.spin, args=(uav,), daemon=True)
    spinner_thread.start()

    success = drone_start(uav)
    try:
        start_time = time.time()
        if success:
            success, plot_data, metrics = drone_run(uav, scenario)
        duration = time.time() - start_time
        print("---------------------------------")
        print(f"Tour of {scenario_file} took {duration:.2f} seconds")
        print("---------------------------------")
    except KeyboardInterrupt:
        success = False

    drone_end(uav)
    uav.shutdown()
    rclpy.shutdown()

    if plot_data is not None:
        plot_paths(plot_data["planned_paths"],
                   plot_data["ordered_points"],
                   plot_data["ground_truth"],
                   obstacles=plot_data["obstacles"],
                   markers_list=plot_data["markers_list"])
        plot_costs(plot_data["segment_planning_times"], plot_data["segment_lengths"])
        plot_rrt_tree(global_rrt_tree_data,
                      waypoints=plot_data["ordered_points"],
                      obstacles=plot_data["obstacles"],
                      planned_paths=plot_data["planned_paths"])
        plot_rrt_tree_3d(global_rrt_tree_data,
                         waypoints=plot_data["ordered_points"],
                         obstacles=plot_data["obstacles"],
                         planned_paths=plot_data["planned_paths"])
    print("Clean exit")
    exit(0)
