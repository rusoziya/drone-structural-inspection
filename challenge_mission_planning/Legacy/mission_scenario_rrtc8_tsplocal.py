#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Modified Mission Script for Structural Inspection Path Planning with ArUco
#
# This script integrates a TSP local search 3‑opt solver to determine the optimal visitation order
# of viewpoints and uses an OMPL-based RRT‑Connect algorithm to plan collision‑free
# paths between waypoints while avoiding cuboid obstacles. In addition, the planner
# verifies that the continuous path (using a fine interpolation) is free of collisions.
# If any segment is in collision, it automatically subdivides the segment by inserting
# intermediate waypoints and replanning.
#
# Assumptions:
# 1. Obstacles are cuboids, axis-aligned, defined by center (x,y,z) and dimensions
#    (d: x-extent, w: y-extent, h: z-extent).
#
# 2. Viewpoint poses are specified under "viewpoint_poses" in the scenario YAML,
#    with fields x, y, z and w (the desired yaw when approaching the marker).
#
# 3. The drone starting pose is under "drone_start_pose". Its z value is overridden
#    to TAKE_OFF_HEIGHT if too low.
#
# 4. TSP ordering uses Euclidean distance.
#
# 5. The OMPL-based RRT‑Connect algorithm is used to compute a continuous 3D path,
#    which is then used directly (no post‐smoothing is performed).
#
# 6. The drone’s yaw is interpolated along the path so that it continuously faces
#    the direction of travel and converges to the marker’s desired orientation.
#
# 7. For collision checking, each obstacle is an axis‑aligned cuboid. A small safety
#    margin is added around each obstacle.
# ------------------------------------------------------------------------------

# ------------------------
# Configuration (Modifiable Parameters)
# ------------------------

# Drone motion parameters
TAKE_OFF_HEIGHT = 1.0      # Height in meters at takeoff 
TAKE_OFF_SPEED = 1.0       # m/s for takeoff 
SLEEP_TIME = 0.05          # Minimal delay between commands (seconds) 
SPEED = 1.0                # m/s during flight 
LAND_SPEED = 0.5           # m/s for landing 

# Obstacle avoidance parameters 
SAFETY_MARGIN = 0.8        # Additional margin (in meters) added around each obstacle 

# Collision checking parameters (used when interpolating along the planned path)
COLLISION_CHECK_RESOLUTION = 0.5  # Step size for interpolation (in meters)

# Recursive planning parameters for subdividing segments in collision 
MAX_RECURSION_DEPTH = 100000     # Maximum times a segment will be subdivided if in collision 

# RRT‑Connect planner parameters 
PLANNING_TIME_LIMIT = 25.0         # Time limit for planning each segment (in seconds) 
PLANNER_RANGE_SMALL = 1.0          # Fixed planner range used for all segments 
PLANNER_RANGE_LARGE = 1.5          # (Unused now)

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

from as2_python_api.drone_interface import DroneInterface

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# For ArUco detection (now enabled in mission logic)
import cv2
import cv2.aruco as aruco
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# OMPL imports for RRTConnect
from ompl import base as ob
from ompl import geometric as og

# For the assignment
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

# ------------------------
# Helper Functions
# ------------------------
def load_scenario(scenario_file):
    """Load scenario from a YAML file."""
    with open(scenario_file, 'r') as f:
        scenario = yaml.safe_load(f)
    return scenario

def compute_euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def build_distance_matrix(points):
    """Build a complete distance matrix for a list of 3D points."""
    n = len(points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = compute_euclidean_distance(points[i], points[j])
    return matrix

def interpolate_angle(a, b, t):
    """
    Interpolate between angles a and b (in radians) by fraction t.
    Handles angle wrap-around.
    """
    diff = (b - a + math.pi) % (2*math.pi) - math.pi
    return a + diff * t

def is_state_valid_cuboids(state, obstacles):
    """
    Checks whether a given (x, y, z) state is free of collisions with axis-aligned
    cuboid obstacles. A safety margin is added around each obstacle.
    """
    x, y, z = state[0], state[1], state[2]
    for obs in obstacles:
        ox, oy, oz = obs["x"], obs["y"], obs["z"]
        dx, dy, dz = obs["d"], obs["w"], obs["h"]
        hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
        hx += SAFETY_MARGIN
        hy += SAFETY_MARGIN
        hz += SAFETY_MARGIN
        if (ox - hx <= x <= ox + hx and
            oy - hy <= y <= oy + hy and
            oz - hz <= z <= oz + hz):
            return False
    return True

def is_path_collision_free(path, obstacles, resolution=COLLISION_CHECK_RESOLUTION):
    """
    Checks the continuous path for collision by interpolating between successive points.
    Returns True if the entire path is collision-free.
    """
    if len(path) < 2:
        return True
    for i in range(len(path)-1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        seg_length = compute_euclidean_distance(p1, p2)
        steps = max(int(seg_length / resolution), 1)
        for j in range(steps + 1):
            t = j / steps
            interp = p1 + t * (p2 - p1)
            if not is_state_valid_cuboids(interp, obstacles):
                return False
    return True

def load_world_yaml(world_file=None):
    """
    Load the world.yaml file containing marker and other world information.
    By default, attempts to load from 'config_sim/world/world.yaml' 
    relative to the script or current directory.
    """
    if world_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        world_file = os.path.join(script_dir, "config_sim", "world", "world.yaml")
    with open(world_file, 'r') as f:
        world_data = yaml.safe_load(f)
    return world_data

def get_marker_info(world_data):
    """
    Process the world data and extract marker information.
    Returns a list of dictionaries for each marker with a field 'marker_id'.
    """
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
    """
    (LEGACY) For a given viewpoint, find the single closest marker ID.
    Not used in final assignment-based approach, but kept for reference.
    """
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

# ------------------------
# Hungarian assignment for viewpoint -> marker
# ------------------------
def assign_viewpoints_to_markers(viewpoints, markers_list):
    """
    Compute a one-to-one assignment that pairs each viewpoint with exactly one marker,
    minimizing total distance. 
    Returns a list of marker IDs in the same order as 'viewpoints'.
    """
    n_view = len(viewpoints)
    n_mark = len(markers_list)
    if n_mark < n_view:
        print("[WARNING] Fewer markers than viewpoints! Some viewpoints may remain unassigned.")

    # Build cost matrix: rows = viewpoints, cols = markers
    cost_matrix = np.zeros((n_view, n_mark))
    for i in range(n_view):
        vx, vy, vz = viewpoints[i]
        for j in range(n_mark):
            mx, my, mz = markers_list[j]["xyz"]
            dist = math.sqrt((vx - mx)**2 + (vy - my)**2 + (vz - mz)**2)
            cost_matrix[i, j] = dist

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # row_ind[i], col_ind[i] => viewpoint row_ind[i] assigned to marker col_ind[i]
    assigned_marker_ids = [None] * n_view
    for i in range(len(row_ind)):
        v_idx = row_ind[i]
        m_idx = col_ind[i]
        assigned_marker_ids[v_idx] = markers_list[m_idx]["marker_id"]

    # If any viewpoint is unassigned because fewer markers than viewpoints:
    # assigned_marker_ids might remain None. That is an edge case.
    return assigned_marker_ids

# ------------------------
# TSP Local Search 3‑opt Solver
# (unchanged)
# ------------------------
def apply_3opt_move(tour, i, j, k, distance_matrix):
    A = tour[:i]
    B = tour[i:j]
    C = tour[j:k]
    D = tour[k:]
    candidates = []
    candidates.append(A + B + C + D)
    candidates.append(A + B[::-1] + C + D)
    candidates.append(A + B + C[::-1] + D)
    candidates.append(A + B[::-1] + C[::-1] + D)
    candidates.append(A + C + B + D)
    candidates.append(A + C + B[::-1] + D)
    candidates.append(A + C[::-1] + B + D)
    candidates.append(A + C[::-1] + B[::-1] + D)

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

# --------------------------
# RRT‑Connect Path Planning
# (unchanged)
# --------------------------
def plan_rrtconnect(start, goal, obstacles, bounds, planner_range=PLANNER_RANGE_SMALL, planning_time_limit=PLANNING_TIME_LIMIT):
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
        path = pdef.getSolutionPath()
        path_simplifier = og.PathSimplifier(si)
        path_simplifier.simplifyMax(path)
        path_states = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            point = [state[0], state[1], state[2]]
            path_states.append(point)
        planning_time = time.time() - start_time
        return path_states, planning_time
    else:
        planning_time = time.time() - start_time
        return None, planning_time

def plan_segment(start, goal, obstacles, bounds, planner_range, planning_time_limit, recursion_depth=MAX_RECURSION_DEPTH):
    path, ptime = plan_rrtconnect(start, goal, obstacles, bounds, planner_range, planning_time_limit)
    if path is None:
        return None, ptime

    if is_path_collision_free(path, obstacles):
        return path, ptime
    else:
        if recursion_depth <= 0:
            print("Max recursion reached; using the current path even if it is near obstacles.")
            return path, ptime
        mid_point = [(s + g) / 2.0 for s, g in zip(start, goal)]
        if not is_state_valid_cuboids(mid_point, obstacles):
            print("Arithmetic midpoint is invalid; aborting segment.")
            return None, ptime

        print("Subdividing segment using valid arithmetic midpoint: {}".format(mid_point))
        path1, time1 = plan_segment(start, mid_point, obstacles, bounds, planner_range, planning_time_limit, recursion_depth - 1)
        path2, time2 = plan_segment(mid_point, goal, obstacles, bounds, planner_range, planning_time_limit, recursion_depth - 1)
        if path1 is None or path2 is None:
            return None, ptime + time1 + time2

        combined = path1[:-1] + path2
        return combined, ptime + time1 + time2

def plot_paths(planned_paths, waypoints, ground_truth):
    fig2d, ax2d = plt.subplots()
    for path in planned_paths:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax2d.plot(xs, ys, label="Planned Path")

    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    ax2d.plot(wp_x, wp_y, 'ro-', label="Waypoints")

    gt_x = [p[0] for p in ground_truth]
    gt_y = [p[1] for p in ground_truth]
    ax2d.plot(gt_x, gt_y, 'g--', label="Ground Truth")

    ax2d.set_xlabel("X")
    ax2d.set_ylabel("Y")
    ax2d.legend()
    ax2d.set_title("2D Trajectory")
    
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    for path in planned_paths:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax3d.plot(xs, ys, zs, label="Planned Path")

    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    wp_z = [p[2] for p in waypoints]
    ax3d.plot(wp_x, wp_y, wp_z, 'ro-', label="Waypoints")

    gt_x = [p[0] for p in ground_truth]
    gt_y = [p[1] for p in ground_truth]
    gt_z = [p[2] for p in ground_truth]
    ax3d.plot(gt_x, gt_y, gt_z, 'g--', label="Ground Truth")

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend()
    ax3d.set_title("3D Trajectory")
    
    plt.show()

# --------------------------
# Single-Node ArUco Detection + Drone Interface
# (unchanged)
# --------------------------
class SingleNodeDroneMission(DroneInterface):
    def __init__(self, drone_id, use_sim_time=True, verbose=False):
        super().__init__(drone_id=drone_id, use_sim_time=use_sim_time, verbose=verbose)
        self.br = CvBridge()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.aruco_params = aruco.DetectorParameters()
        self.detected_marker_id = None
        self.subscription = self.create_subscription(
            Image,
            "sensor_measurements/hd_camera/image_raw",
            self.image_callback,
            10
        )

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
            print(f"[DEBUG] image_callback: Detected marker ids: {ids.flatten()}")
        else:
            self.detected_marker_id = None

        aruco.drawDetectedMarkers(cv_image, corners, ids)
        # cv2.imshow("Aruco Detection", cv_image)
        cv2.waitKey(1000)

    def wait_for_expected_marker(self, expected_marker_id, timeout=1000.0):
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
                    print(f"[DEBUG] Marker id {self.detected_marker_id} does not match expected marker {expected_marker_id}. Ignoring and waiting...")
                    self.detected_marker_id = None
        print(f"[DEBUG] Timeout reached: no detection of expected marker {expected_marker_id}.")
        return None

    def wait_for_marker(self, timeout=1000.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            if self.detected_marker_id is not None:
                return self.detected_marker_id
        return None

# --------------------------
# Marker Verification
# (unchanged)
# --------------------------
def verify_marker_and_adjust(drone_interface: SingleNodeDroneMission,
                             current_point, target_point, expected_yaw, expected_marker_id,
                             marker_ids_set, timeout=1000.0):
    if expected_marker_id not in marker_ids_set:
        print(f"Expected marker id {expected_marker_id} not found in world.yaml.")
        return False

    print(f"Waiting for marker id {expected_marker_id} at waypoint...")
    detected_id = drone_interface.wait_for_expected_marker(expected_marker_id, timeout=timeout)
    if detected_id is None:
        print("No marker detected within timeout.")
        return False

    print("Marker verification successful.")
    return True

# --------------------------
# Drone Mission Functions
# (unchanged)
# --------------------------
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
    print("Land")
    success = drone_interface.land(speed=LAND_SPEED)
    print(f"Land success: {success}")
    if not success:
        return success
    print("Manual")
    success = drone_interface.manual()
    print(f"Manual success: {success}")
    return success

def drone_run(drone_interface: SingleNodeDroneMission, scenario: dict) -> bool:
    """
    Run the mission:
      - Parse the scenario for starting pose, obstacles, and viewpoint poses.
      - Assign each viewpoint to a unique marker (Hungarian assignment).
      - Solve the TSP for optimal visitation using a local search 3‑opt solver.
      - For each segment between ordered points, plan a collision-free path using 
        OMPL RRT‑Connect. If a path is not collision-free, intermediate waypoints are 
        inserted automatically via recursive subdivision using an arithmetic midpoint.
      - The drone follows the computed path with continuous yaw interpolation.
      - After fully reaching each waypoint, marker verification is performed using marker data from world.yaml.
    """
    print("Run mission with RRT‑Connect planning, TSP ordering (3‑opt local search), continuous commands, obstacle avoidance, and marker verification")
    start_pose = scenario.get("drone_start_pose", {"x": 0.0, "y": 0.0, "z": 0.0})
    if start_pose["z"] < TAKE_OFF_HEIGHT:
        start_pose["z"] = TAKE_OFF_HEIGHT

    # Load obstacles from scenario
    obstacles = [obs for key, obs in scenario.get("obstacles", {}).items()]

    # Load world.yaml and get marker information + set of IDs
    world_data = load_world_yaml("config_sim/world/world.yaml")
    markers_list = get_marker_info(world_data)
    marker_ids_set = {m["marker_id"] for m in markers_list}

    # 1. Gather viewpoint poses from scenario
    viewpoint_dict = scenario.get("viewpoint_poses", {})
    view_keys = sorted(viewpoint_dict.keys(), key=lambda x: int(x))
    viewpoints = []
    marker_yaws = []
    for key in view_keys:
        vp = viewpoint_dict[key]
        viewpoints.append([vp["x"], vp["y"], vp["z"]])
        marker_yaws.append(vp["w"])

    # 2. Use Hungarian assignment to map each viewpoint -> exactly one marker ID
    assigned_marker_ids = assign_viewpoints_to_markers(viewpoints, markers_list)

    # 3. Combine start pose with viewpoints
    points = [[start_pose["x"], start_pose["y"], start_pose["z"]]] + viewpoints
    yaw_list = [None] + marker_yaws
    # Insert None for the drone start pose, then the assigned IDs for each viewpoint
    expected_marker_ids = [None] + assigned_marker_ids

    # 4. Solve TSP for these points
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

    # Use a fixed planner range for every segment
    fixed_planner_range = PLANNER_RANGE_SMALL

    for i in range(len(ordered_points) - 1):
        seg_start = ordered_points[i]
        seg_goal = ordered_points[i+1]
        planner_range = fixed_planner_range

        # If yaw for the next viewpoint is None, we compute a default yaw based on travel direction
        dest_yaw = (ordered_yaws[i+1] if ordered_yaws[i+1] is not None 
                    else math.atan2(seg_goal[1]-seg_start[1], seg_goal[0]-seg_start[0]))
        print(f"Planning path from {seg_start} to {seg_goal} with fixed planner range {planner_range}")

        path, planning_time = plan_segment(seg_start, seg_goal, obstacles, bounds, planner_range, PLANNING_TIME_LIMIT)
        if path is None:
            print("No solution found for segment, aborting mission planning.")
            return False
        print(f"Segment planned in {planning_time:.2f} s with {len(path)} states.")
        total_planning_time += planning_time

        planned_path = path
        planned_paths.append(planned_path)
        seg_length = sum(compute_euclidean_distance(planned_path[j], planned_path[j+1]) 
                         for j in range(len(planned_path)-1))
        total_path_length += seg_length

        # Execute the planned path
        current_state = planned_path[0]
        N = len(planned_path)
        for j, point in enumerate(planned_path):
            # Skip if already close
            if compute_euclidean_distance(current_state, point) < 0.1:
                print(f"Skipping point {point} (already reached)")
                current_state = point
                continue
            t = j / (N - 1) if N > 1 else 1.0
            yaw_command = interpolate_angle(
                dest_yaw if j == N - 1 else math.atan2(planned_path[1][1] - planned_path[0][1],
                                                       planned_path[1][0] - planned_path[0][0]),
                dest_yaw, t)
            print(f"Continuously moving to point: {point} with yaw: {yaw_command:.2f} rad")
            success_move = drone_interface.go_to.go_to_point_with_yaw(point, angle=yaw_command, speed=SPEED)
            if not success_move:
                print(f"Failed to move to point: {point}")
                return False
            time.sleep(SLEEP_TIME)
            current_state = point

        # After finishing segment, verify marker
        time.sleep(1.0)  # let drone settle
        current_point = planned_path[-1]
        expected_marker_id = ordered_expected_marker_ids[i+1]
        if expected_marker_id is not None:
            print(f"Expected marker for this waypoint: {expected_marker_id}")
            if not verify_marker_and_adjust(drone_interface,
                                            current_point=current_point,
                                            target_point=seg_goal,
                                            expected_yaw=dest_yaw,
                                            expected_marker_id=expected_marker_id,
                                            marker_ids_set=marker_ids_set,
                                            timeout=1000.0):
                print("Marker verification failed; aborting mission.")
                return False
        else:
            print("No expected marker for this segment; continuing.")

    print(f"Total planning time: {total_planning_time:.2f} s")
    print(f"Total path length: {total_path_length:.2f} m")
    plot_paths(planned_paths, ordered_points, ground_truth)
    return True

# --------------------------
# Main
# --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single drone mission with RRT‑Connect planning, TSP (3‑opt local search) optimization, obstacle avoidance, and marker verification enabled'
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
    uav = SingleNodeDroneMission(drone_id=drone_namespace, use_sim_time=use_sim_time, verbose=verbosity)
    
    # Start a dedicated spinner thread so that subscriptions (and image_callback) are processed continuously.
    spinner_thread = threading.Thread(target=rclpy.spin, args=(uav,), daemon=True)
    spinner_thread.start()
    
    success = drone_start(uav)
    try:
        start_time = time.time()
        if success:
            success = drone_run(uav, scenario)
        duration = time.time() - start_time
        print("---------------------------------")
        print(f"Tour of {scenario_file} took {duration:.2f} seconds")
        print("---------------------------------")
    except KeyboardInterrupt:
        pass
    
    success = drone_end(uav)
    uav.shutdown()
    rclpy.shutdown()
    print("Clean exit")
    exit(0)
