#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Modified Mission Script for Structural Inspection Path Planning with ArUco
#
# This script integrates a TSP solver (now using a basic 3‑opt heuristic) to determine
# the visitation order of viewpoints, uses an OMPL-based RRT‑Connect algorithm to plan
# collision‑free paths, and verifies the continuous path is free of collisions.
# If any segment is in collision, it automatically subdivides the segment by inserting
# intermediate waypoints and replanning.
#
# Now also includes B-spline trajectory planning (see compute_bspline_trajectory).
#
# Assumptions:
# 1. Obstacles are cuboids, axis-aligned, defined by center (x,y,z) and dimensions.
# 2. Viewpoint poses are specified under "viewpoint_poses" in the scenario YAML,
#    with fields x, y, z and w (the desired yaw).
# 3. The drone starting pose is under "drone_start_pose". Its z value is overridden
#    to TAKE_OFF_HEIGHT if too low.
# 4. TSP ordering uses Euclidean distance, now solved via a simple 3‑opt heuristic.
# 5. The OMPL-based RRT‑Connect algorithm is used to compute a continuous 3D path,
#    which is optionally refined by a B-spline for the final flight commands.
# 6. The drone’s yaw is interpolated along the path so that it continuously faces
#    the direction of travel and converges to the marker’s desired orientation.
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
import os  # For robust file paths if needed

from as2_python_api.drone_interface import DroneInterface

# Removed the python_tsp.exact import
# from python_tsp.exact import solve_tsp_dynamic_programming

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import cv2.aruco as aruco
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ompl import base as ob
from ompl import geometric as og

from scipy.interpolate import make_interp_spline

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
    Process the world data and extract a dictionary mapping:
      marker_info[model_name] = (marker_id)
    e.g. "id1" -> 24, "id2" -> 34, etc.
    """
    markers = {}
    for obj in world_data.get("objects", []):
        model_name = obj.get("model_name", "")
        mtype = obj.get("model_type", "")
        # Check if it's an ArUco marker
        if mtype.startswith("aruco_id") and "_marker" in mtype:
            # parse the integer from "aruco_idXX_marker"
            marker_num_str = mtype[len("aruco_id"):mtype.index("_marker")]
            try:
                marker_num = int(marker_num_str)
            except ValueError:
                continue
            markers[model_name] = marker_num
    return markers

# --------------------------
# B-spline Trajectory Planning
# --------------------------
def compute_bspline_trajectory(path, num_points=8):
    """
    Given a list of 3D waypoints (path), generate a B-spline with 'num_points' samples.
    Returns a new list of 3D points representing the B-spline path.
    Debug messages are printed to verify correctness.
    
    If the path has fewer than 4 points, B-spline interpolation may fail or be suboptimal,
    so in that case we skip B-spline and return the original path.
    """
    print(f"Debug: Attempting B-spline with {len(path)} original points.")
    if len(path) < 4:
        print("Debug: Not enough points for cubic B-spline. Returning original path.")
        return path

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    zs = [p[2] for p in path]

    t_original = np.linspace(0, 1, len(path))
    t_new = np.linspace(0, 1, num_points)

    k_spline = min(3, len(path) - 1)

    spline_x = make_interp_spline(t_original, xs, k=k_spline)
    spline_y = make_interp_spline(t_original, ys, k=k_spline)
    spline_z = make_interp_spline(t_original, zs, k=k_spline)

    x_smooth = spline_x(t_new)
    y_smooth = spline_y(t_new)
    z_smooth = spline_z(t_new)

    bspline_path = list(zip(x_smooth, y_smooth, z_smooth))
    print(f"Debug: B-spline path generated with {len(bspline_path)} points (original had {len(path)}).")
    return bspline_path

# --------------------------
# Minimal 3‑opt TSP Solver
# --------------------------
def solve_tsp_dynamic_programming(distance_matrix):
    """
    A basic 3‑opt TSP heuristic.
    Returns (best_route, best_cost).

    1. Start with a simple initial route [0, 1, 2, ..., n-1].
    2. Repeatedly try 3‑opt moves until no improvement.
    3. Return the final route and cost.

    This is not heavily optimized and is meant as a drop-in replacement
    for the old exact solver call.
    """
    n = len(distance_matrix)
    route = list(range(n))  # simple initial route
    best_cost = route_cost(route, distance_matrix)

    improved = True
    iteration = 0
    max_iterations = 300  # you can adjust this limit

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # Attempt all 3‑opt moves
        for i in range(n - 2):
            for j in range(i+1, n - 1):
                for k in range(j+1, n):
                    new_route = do_3opt_move(route, i, j, k)
                    new_cost = route_cost(new_route, distance_matrix)
                    if new_cost < best_cost:
                        route = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Return the final route plus the total cost
    return route, best_cost

def route_cost(route, distance_matrix):
    """Compute the cost of a TSP route with wrap-around."""
    cost = 0.0
    n = len(route)
    for i in range(n):
        cost += distance_matrix[route[i]][route[(i+1) % n]]
    return cost

def do_3opt_move(route, i, j, k):
    """
    Given a route and indices i<j<k, produce a route with
    the 3-opt re-linking of segments. We test one variant:
      - reverse the middle segment
      - reverse the last segment
    You can expand this with more permutations for thoroughness.
    """
    # route is e.g. [0,1,2,3,4,5]
    # i, j, k are boundaries
    # segments: 
    #   segment1 = route[0:i+1]
    #   segment2 = route[i+1:j+1]
    #   segment3 = route[j+1:k+1]
    #   segment4 = route[k+1:]
    segment1 = route[:i+1]
    segment2 = route[i+1:j+1]
    segment3 = route[j+1:k+1]
    segment4 = route[k+1:]

    # A single 3‑opt variation: reverse segment2 and segment3
    # You can add more variations for a more thorough 3‑opt.
    new_route = segment1 + segment2[::-1] + segment3[::-1] + segment4
    return new_route

# --------------------------
# RRT‑Connect Path Planning in 3D
# --------------------------
def plan_rrtconnect(start, goal, obstacles, bounds, planner_range=PLANNER_RANGE_SMALL, planning_time_limit=PLANNING_TIME_LIMIT):
    # ... (unchanged)
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
    # ... (unchanged)
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
    # ... (unchanged)
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
# --------------------------
class SingleNodeDroneMission(DroneInterface):
    """
    Merges ArUco marker detection into the same node that controls the drone.
    Marker detection is subscribed and now actively used for mission verification.
    """
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
        else:
            self.detected_marker_id = None

        aruco.drawDetectedMarkers(cv_image, corners, ids)
        cv2.imshow("Aruco Detection", cv_image)
        cv2.waitKey(1)

    def wait_for_marker(self, timeout=2.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.detected_marker_id is not None:
                return self.detected_marker_id
        return None

# --------------------------
# Marker Verification
# --------------------------
def verify_marker_and_adjust(drone_interface: SingleNodeDroneMission,
                             current_point, target_point, expected_yaw, expected_marker_id,
                             marker_info, timeout=2.0):
    """
    Verify that the expected ArUco marker (based on world.yaml) is detected.
    If the expected marker is not detected, abort the mission (return False).
    """
    if expected_marker_id is None:
        print("No marker ID expected for this viewpoint.")
        return True

    print(f"Waiting for marker id {expected_marker_id}...")
    detected_id = drone_interface.wait_for_marker(timeout=timeout)
    if detected_id is None:
        print("No marker detected within timeout.")
        return False
    if detected_id != expected_marker_id:
        print(f"Detected marker id {detected_id} does not match expected id {expected_marker_id}.")
        return False

    print("Marker verification successful.")
    return True

# --------------------------
# Drone Mission Functions
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
      - Load world.yaml for marker model_name->ID mapping.
      - For each viewpoint in scenario, figure out the expected marker ID by
        matching model_name "idX" in the world data.
      - Solve the TSP using a 3‑opt approach (solve_tsp_dynamic_programming).
      - For each segment between ordered points, plan a collision-free path using 
        OMPL RRT‑Connect. If a path is not collision-free, intermediate waypoints are 
        inserted automatically via recursive subdivision using an arithmetic midpoint.
      - Convert the final path to a B-spline for smooth flight commands.
      - The drone follows the B-spline path with continuous yaw interpolation.
      - ArUco marker verification is performed using marker data from world.yaml.
    """
    print("Run mission with RRT‑Connect planning, 3‑opt TSP, B-spline smoothing, obstacle avoidance, and marker verification")
    start_pose = scenario.get("drone_start_pose", {"x": 0.0, "y": 0.0, "z": 0.0})
    if start_pose["z"] < TAKE_OFF_HEIGHT:
        start_pose["z"] = TAKE_OFF_HEIGHT

    obstacles = [obs for key, obs in scenario.get("obstacles", {}).items()]

    viewpoint_dict = scenario.get("viewpoint_poses", {})
    world_data = load_world_yaml("config_sim/world/world.yaml")
    marker_info = get_marker_info(world_data)

    viewpoints = []
    marker_yaws = []
    expected_marker_ids = []
    for key in sorted(viewpoint_dict.keys(), key=lambda x: int(x)):
        vp = viewpoint_dict[key]
        vx, vy, vz = vp["x"], vp["y"], vp["z"]
        vyaw = vp["w"]
        viewpoints.append([vx, vy, vz])
        marker_yaws.append(vyaw)

        model_name = f"id{key}"
        if model_name in marker_info:
            expected_marker_ids.append(marker_info[model_name])
        else:
            print(f"Warning: No matching model_name '{model_name}' in world.yaml. Setting expected_marker_id=None")
            expected_marker_ids.append(None)

    points = [[start_pose["x"], start_pose["y"], start_pose["z"]]] + viewpoints
    yaw_list = [None] + marker_yaws
    expected_marker_ids = [None] + expected_marker_ids

    distance_matrix = build_distance_matrix(points)
    permutation, tsp_distance = solve_tsp_dynamic_programming(distance_matrix)
    print(f"TSP order: {permutation}, total straight-line distance: {tsp_distance:.2f}")

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
        planner_range = fixed_planner_range

        dest_yaw = (ordered_yaws[i+1] if ordered_yaws[i+1] is not None 
                    else math.atan2(seg_goal[1]-seg_start[1], seg_goal[0]-seg_start[0]))
        print(f"Planning path from {seg_start} to {seg_goal} with fixed planner range {planner_range}")

        path, planning_time = plan_segment(seg_start, seg_goal, obstacles, bounds, planner_range, PLANNING_TIME_LIMIT)
        if path is None:
            print("No solution found for segment, aborting mission planning.")
            return False
        print(f"Segment planned in {planning_time:.2f} s with {len(path)} states.")
        total_planning_time += planning_time

        bspline_path = compute_bspline_trajectory(path, num_points=8)
        planned_paths.append(bspline_path)
        seg_length = sum(compute_euclidean_distance(bspline_path[j], bspline_path[j+1]) 
                         for j in range(len(bspline_path)-1))
        total_path_length += seg_length

        if len(bspline_path) >= 2:
            start_yaw = math.atan2(bspline_path[1][1] - bspline_path[0][1],
                                   bspline_path[1][0] - bspline_path[0][0])
        else:
            start_yaw = dest_yaw

        N = len(bspline_path)
        for j, point in enumerate(bspline_path):
            t = j / (N - 1) if N > 1 else 1.0
            yaw_command = interpolate_angle(start_yaw, dest_yaw, t)
            print(f"Debug: B-spline point {j+1}/{N} -> {point}, yaw={yaw_command:.2f}")
            success_move = drone_interface.go_to.go_to_point_with_yaw(point, angle=yaw_command, speed=SPEED)
            if not success_move:
                print(f"Failed to move to B-spline point: {point}")
                return False
            time.sleep(SLEEP_TIME)

        current_point = bspline_path[-1]
        expected_marker_id = ordered_expected_marker_ids[i+1]
        if expected_marker_id is not None:
            if not verify_marker_and_adjust(drone_interface,
                                            current_point=current_point,
                                            target_point=seg_goal,
                                            expected_yaw=dest_yaw,
                                            expected_marker_id=expected_marker_id,
                                            marker_info=marker_info,
                                            timeout=2.0):
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
        description='Single drone mission with RRT‑Connect planning, 3‑opt TSP, B-spline smoothing, obstacle avoidance, and marker verification'
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
