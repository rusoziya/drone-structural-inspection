#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Modified Mission Script for Structural Inspection Path Planning
#
# This script integrates OMPL’s RRTConnect algorithm to plan a collision‐free
# path between waypoints while avoiding cuboid obstacles, and uses a TSP solver
# (python_tsp) to determine the optimal visiting order of viewpoints.
#
# Assumptions:
# 1. Obstacles are cuboids, axis-aligned, and defined by a center (x,y,z)
#    and dimensions: d (extent in x), w (extent in y), and h (extent in z).
#    The obstacle occupies [x - d/2, x + d/2] in x, [y - w/2, y + w/2] in y,
#    and [z - h/2, z + h/2] in z.
#
# 2. Viewpoint poses are provided in the scenario YAML file under the key
#    "viewpoint_poses" with fields x, y, and z (orientation is ignored for planning).
#
# 3. The drone starting pose is provided in the YAML under "drone_start_pose".
#
# 4. For TSP ordering, the cost between points is computed using the Euclidean
#    distance (i.e., as if the ground truth were straight-line segments).
#
# 5. OMPL’s RRTConnect is used to plan a trajectory between each pair of ordered
#    waypoints. The state validity function checks for collision with any obstacle.
#
# 6. For plotting performance, we log the planned paths and compare them with
#    the ideal ground truth (straight lines between waypoints).
#
# 7. The simulation uses basic sleep calls to mimic drone motion between planned
#    points; in a real application, these would be replaced by actual drone commands.
# ------------------------------------------------------------------------------

import argparse
import time
import math
import yaml
import logging
import numpy as np
import rclpy
from as2_python_api.drone_interface import DroneInterface

# OMPL imports for planning
import ompl.base as ob
import ompl.geometric as og

# TSP solver
from python_tsp.exact import solve_tsp_dynamic_programming

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global constants
TAKE_OFF_HEIGHT = 1.0      # Height in meters
TAKE_OFF_SPEED = 1.0       # Max speed in m/s for takeoff
SLEEP_TIME = 0.5           # Sleep time to simulate movement
SPEED = 1.0                # Max speed in m/s during path execution
LAND_SPEED = 0.5           # Max speed in m/s for landing

# -------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------

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

def plan_rrt_connect(start, goal, obstacles, bounds):
    """
    Use OMPL’s RRTConnect algorithm to plan a path from start to goal.
    
    Parameters:
      - start, goal: Lists with [x, y, z] coordinates.
      - obstacles: List of obstacle dictionaries.
      - bounds: Dictionary with 'low' and 'high' keys, each a 3-element list.
    
    Returns:
      - path: List of [x, y, z] states if a solution is found.
      - planning_time: Time taken by the planner.
    """
    # Create a 3D state space
    space = ob.RealVectorStateSpace(3)
    b = ob.RealVectorBounds(3)
    b.setLow(bounds['low'])
    b.setHigh(bounds['high'])
    space.setBounds(b)
    
    # Initialize the SimpleSetup object
    ss = og.SimpleSetup(space)
    
    # State validity checker: returns False if the state is inside any obstacle.
    def is_state_valid(state):
        x, y, z = state[0], state[1], state[2]
        for obs in obstacles:
            # Check if point is inside the cuboid obstacle.
            if (obs['x'] - obs['d']/2 <= x <= obs['x'] + obs['d']/2 and
                obs['y'] - obs['w']/2 <= y <= obs['y'] + obs['w']/2 and
                obs['z'] - obs['h']/2 <= z <= obs['z'] + obs['h']/2):
                return False
        return True

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
    
    # Set start and goal states
    start_state = ob.State(space)
    start_state()[0], start_state()[1], start_state()[2] = start[0], start[1], start[2]
    
    goal_state = ob.State(space)
    goal_state()[0], goal_state()[1], goal_state()[2] = goal[0], goal[1], goal[2]
    
    ss.setStartAndGoalStates(start_state, goal_state)
    
    # Set RRTConnect as the planner
    planner = og.RRTConnect(ss.getSpaceInformation())
    ss.setPlanner(planner)
    
    start_time = time.time()
    solved = ss.solve(5.0)  # Allow up to 5 seconds per segment
    planning_time = time.time() - start_time
    
    if solved:
        ss.simplifySolution()
        path_obj = ss.getSolutionPath()
        # Convert the OMPL path to a list of [x, y, z] points
        path = []
        for i in range(path_obj.getStateCount()):
            state = path_obj.getState(i)
            point = [state[0], state[1], state[2]]
            path.append(point)
        return path, planning_time
    else:
        return None, planning_time

def plot_paths(planned_paths, waypoints, ground_truth):
    """
    Plot the planned paths, the ordered waypoints, and the ground truth path.
    
    - planned_paths: list of paths (each path is a list of [x, y, z])
    - waypoints: list of ordered waypoints used for planning
    - ground_truth: list of straight-line segments between waypoints
    """
    # 2D Plot (X-Y trajectory)
    fig2d, ax2d = plt.subplots()
    for path in planned_paths:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax2d.plot(xs, ys, label="Planned Path")
    # Plot waypoints
    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    ax2d.plot(wp_x, wp_y, 'ro-', label="Waypoints")
    # Plot ground truth (ideal straight-line segments)
    gt_x = [p[0] for p in ground_truth]
    gt_y = [p[1] for p in ground_truth]
    ax2d.plot(gt_x, gt_y, 'g--', label="Ground Truth")
    ax2d.set_xlabel("X")
    ax2d.set_ylabel("Y")
    ax2d.legend()
    ax2d.set_title("2D Trajectory")
    
    # 3D Plot (X-Y-Z trajectory)
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    for path in planned_paths:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax3d.plot(xs, ys, zs, label="Planned Path")
    # Plot waypoints in 3D
    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    wp_z = [p[2] for p in waypoints]
    ax3d.plot(wp_x, wp_y, wp_z, 'ro-', label="Waypoints")
    # Plot ground truth in 3D
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

# ------------------------------------------------------------------------------
# Drone Mission Functions
# ------------------------------------------------------------------------------

def drone_start(drone_interface: DroneInterface) -> bool:
    """Starts the mission: arms the drone, switches to offboard mode, and takes off."""
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

def drone_end(drone_interface: DroneInterface) -> bool:
    """Ends the mission: lands the drone and switches to manual mode."""
    print('End mission')
    print('Land')
    success = drone_interface.land(speed=LAND_SPEED)
    print(f'Land success: {success}')
    if not success:
        return success
    print('Manual')
    success = drone_interface.manual()
    print(f'Manual success: {success}')
    return success

def drone_run(drone_interface: DroneInterface, scenario):
    """
    Run the mission:
      - Parse the scenario to get the starting pose, obstacles, and viewpoint poses.
      - Solve the TSP to determine the optimal visitation order.
      - For each leg between ordered points, plan a collision-free path with OMPL's
        RRTConnect and simulate the drone moving along the path.
      - Log performance metrics and plot the trajectory.
    """
    print('Run mission with OMPL RRTConnect and TSP ordering')
    
    # Extract starting pose, obstacles, and viewpoint poses from the scenario.
    start_pose = scenario.get("drone_start_pose", {"x": 0.0, "y": 0.0, "z": 0.0})
    obstacles = [obs for key, obs in scenario.get("obstacles", {}).items()]
    viewpoint_dict = scenario.get("viewpoint_poses", {})
    viewpoints = []
    # Sort viewpoint keys numerically
    for key in sorted(viewpoint_dict.keys(), key=lambda x: int(x)):
        vp = viewpoint_dict[key]
        viewpoints.append([vp["x"], vp["y"], vp["z"]])
    
    # Combine starting pose and viewpoints for TSP (starting pose is index 0)
    points = [[start_pose["x"], start_pose["y"], start_pose["z"]]] + viewpoints
    distance_matrix = build_distance_matrix(points)
    
    # Solve the TSP to get an optimal visitation order.
    permutation, tsp_distance = solve_tsp_dynamic_programming(distance_matrix)
    print(f"TSP order: {permutation}, total straight-line distance: {tsp_distance:.2f}")
    
    # Reorder points according to TSP solution.
    ordered_points = [points[i] for i in permutation]
    
    # For ground truth, we assume ideal straight-line segments.
    ground_truth = ordered_points
    
    # Define environment bounds for OMPL planning (adjust as needed).
    bounds = {"low": [-10, -10, 0], "high": [10, 10, 10]}
    
    planned_paths = []
    total_planning_time = 0.0
    total_path_length = 0.0
    
    # Plan and execute path for each segment between ordered waypoints.
    for i in range(len(ordered_points) - 1):
        start = ordered_points[i]
        goal = ordered_points[i + 1]
        print(f"Planning path from {start} to {goal}")
        path, planning_time = plan_rrt_connect(start, goal, obstacles, bounds)
        if path is None:
            print("No solution found for segment, aborting mission planning.")
            return False
        print(f"Segment planned in {planning_time:.2f} s with {len(path)} states.")
        planned_paths.append(path)
        total_planning_time += planning_time
        
        # Compute segment path length.
        seg_length = sum(compute_euclidean_distance(path[j], path[j+1]) for j in range(len(path)-1))
        total_path_length += seg_length
        
        # Simulate drone moving along the planned path.
        for point in path:
            print(f"Moving to point: {point}")
            # In a real implementation, you would call:
            # drone_interface.go_to.go_to_point(point, speed=SPEED)
            time.sleep(SLEEP_TIME)
    
    print(f"Total planning time: {total_planning_time:.2f} s")
    print(f"Total path length: {total_path_length:.2f} m")
    
    # Plot the planned paths, waypoints, and the ideal ground truth.
    plot_paths(planned_paths, ordered_points, ground_truth)
    return True

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single drone mission with OMPL RRTConnect and TSP optimization')
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
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print(f'Running mission for drone {drone_namespace} using scenario {scenario_file}')
    
    # Load the scenario file.
    scenario = load_scenario(scenario_file)
    
    rclpy.init()
    
    uav = DroneInterface(
        drone_id=drone_namespace,
        use_sim_time=use_sim_time,
        verbose=verbosity)
    
    success = drone_start(uav)
    if success:
        success = drone_run(uav, scenario)
    success = drone_end(uav)
    
    uav.shutdown()
    rclpy.shutdown()
    print('Clean exit')
