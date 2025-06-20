#!/usr/bin/env python3
"""
Modified Drone Mission Script with OMPL RRTConnect Path Planning, TSP Ordering,
and Performance Plot Saving.

Pre-run instructions:
1. Build the workspace:
     cd ~/project_gazebo_ws
     colcon build --symlink-install
2. Source the install setup:
     source install/setup.bash
3. Run the mission script, for example:
     python3 mission_scenario.py scenarios/scenario1.yaml

Modifications:
- Integrated OMPL’s RRTConnect algorithm for collision-free planning between waypoints.
- Loaded scenario parameters (drone start pose, obstacles, viewpoints) from a YAML file.
- Used python_tsp to determine an optimal visitation order (TSP) for the viewpoints.
- Logged performance data (positions and timestamps) during simulated path following.
- Plotted and then saved performance graphs (2D XY, 3D path, position vs. time) after display.
- Debug messages indicate the save location of each plot.
  
Assumptions:
- Obstacles are defined as cuboids with a center (x, y, z) and dimensions (w, d, h).
- The drone’s state space is simplified as a 3D real vector with bounds [-10, 10] on all axes.
- The drone "follows" the computed path via discrete point-to-point moves simulated with a sleep.
- OMPL and python_tsp (with Python bindings) are installed and available.
- The DroneInterface provides methods: arm, offboard, takeoff, go_to, land, manual, etc.

"""

import argparse
from time import sleep, time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import rclpy

# OMPL and TSP imports
import ompl.base as ob
import ompl.geometric as og
from python_tsp.exact import solve_tsp_dynamic_programming

# Import the drone interface (assumed available)
from as2_python_api.drone_interface import DroneInterface

# Global constants
TAKE_OFF_HEIGHT = 1.0  # meters
TAKE_OFF_SPEED = 1.0   # m/s
SLEEP_TIME = 0.5       # seconds between moves
SPEED = 1.0            # m/s (used for simulation of movement)
LAND_SPEED = 0.5       # m/s

# ------------------ Helper Functions ------------------

def is_state_valid(state, obstacles):
    """
    Check if a given OMPL state is valid (i.e., collision-free).
    Assumes each obstacle is a cuboid with center (x,y,z) and dimensions w (width), d (depth), h (height).
    """
    x, y, z = state[0], state[1], state[2]
    for obs in obstacles:
        hx = obs['w'] / 2.0
        hy = obs['d'] / 2.0
        hz = obs['h'] / 2.0
        if (obs['x'] - hx <= x <= obs['x'] + hx and
            obs['y'] - hy <= y <= obs['y'] + hy and
            obs['z'] - hz <= z <= obs['z'] + hz):
            return False
    return True

def plan_rrt_path(start_point, goal_point, obstacles):
    """
    Use OMPL's RRTConnect planner to compute a path from start_point to goal_point.
    Returns the path as a list of [x, y, z] points and the planning time.
    """
    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-10)
    bounds.setHigh(10)
    space.setBounds(bounds)
    
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(lambda state: is_state_valid(state, obstacles)))
    
    start = ob.State(space)
    start[0], start[1], start[2] = start_point[0], start_point[1], start_point[2]
    
    goal = ob.State(space)
    goal[0], goal[1], goal[2] = goal_point[0], goal_point[1], goal_point[2]
    
    ss.setStartAndGoalStates(start, goal)
    
    planner = og.RRTConnect(ss.getSpaceInformation())
    ss.setPlanner(planner)
    
    solved = ss.solve(5.0)
    if solved:
        ss.simplifySolution()
        path = ss.getSolutionPath()
        pts = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            pts.append([state[0], state[1], state[2]])
        planning_time = ss.getLastPlanComputationTime()
        return pts, planning_time
    else:
        return None, None

def read_scenario(scenario_file):
    """
    Read the scenario YAML file and extract:
      - drone_start_pose: dictionary with x, y, z
      - obstacles: a list of obstacle definitions (each as a dict)
      - viewpoint_poses: a list of [x, y, z] for each target viewpoint
    """
    print(f"Reading scenario {scenario_file}")
    with open(scenario_file, 'r') as f:
        scenario = yaml.safe_load(f)
    start_pose = scenario.get('drone_start_pose', {'x': 0.0, 'y': 0.0, 'z': 0.0})
    obstacles = []
    if 'obstacles' in scenario:
        for key, obs in scenario['obstacles'].items():
            obstacles.append(obs)
    viewpoint_poses = []
    if 'viewpoint_poses' in scenario:
        for key, vp in scenario['viewpoint_poses'].items():
            viewpoint_poses.append([vp['x'], vp['y'], vp['z']])
    return start_pose, obstacles, viewpoint_poses

def solve_tsp_ordering(start, waypoints):
    """
    Solve the Travelling Salesman Problem to determine the optimal ordering of waypoints.
    Returns the ordered list of waypoints and the total TSP tour distance.
    """
    points = [[start['x'], start['y'], start['z']]] + waypoints
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
    permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
    zero_index = permutation.index(0)
    ordered_perm = permutation[zero_index:] + permutation[:zero_index]
    ordered_waypoints = [waypoints[i - 1] for i in ordered_perm if i != 0]
    return ordered_waypoints, distance

def plot_performance(overall_path, ordered_waypoints, start_pose, position_log, time_log):
    """
    Plot and save:
      - The 2D XY path (planned vs. ground truth straight-line between waypoints)
      - The 3D planned path vs. ground truth
      - X and Y positions over time

    The plots are displayed and then saved as PNG files in the same folder as the script.
    Debug messages are printed to the terminal.
    """
    overall_path = np.array(overall_path)
    
    # Plot 2D XY Path
    fig1 = plt.figure()
    plt.plot(overall_path[:, 0], overall_path[:, 1], marker='o', label="Planned Path")
    gt_points = np.array([[start_pose['x'], start_pose['y']]] + [[wp[0], wp[1]] for wp in ordered_waypoints])
    plt.plot(gt_points[:, 0], gt_points[:, 1], 'r--', label="Ground Truth")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D XY Path")
    plt.legend()
    
    # Plot 3D Path
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot(overall_path[:, 0], overall_path[:, 1], overall_path[:, 2], marker='o', label="Planned Path")
    gt_points_3d = np.array([[start_pose['x'], start_pose['y'], start_pose['z']]] + ordered_waypoints)
    ax.plot(gt_points_3d[:, 0], gt_points_3d[:, 1], gt_points_3d[:, 2], 'r--', label="Ground Truth")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Path")
    ax.legend()
    
    # Plot X and Y positions over time
    fig3 = plt.figure()
    position_log = np.array(position_log)
    plt.plot(time_log, position_log[:, 0], label="X Position")
    plt.plot(time_log, position_log[:, 1], label="Y Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Position vs. Time")
    plt.legend()
    
    # Display the plots
    plt.show()
    
    # Save the figures in the same folder as this script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig1_path = os.path.join(script_dir, "2D_XY_Path.png")
    fig2_path = os.path.join(script_dir, "3D_Path.png")
    fig3_path = os.path.join(script_dir, "Position_vs_Time.png")
    
    fig1.savefig(fig1_path)
    print(f"Debug: Saved 2D XY Path plot as {fig1_path}")
    fig2.savefig(fig2_path)
    print(f"Debug: Saved 3D Path plot as {fig2_path}")
    fig3.savefig(fig3_path)
    print(f"Debug: Saved Position vs Time plot as {fig3_path}")

# ------------------ Drone Mission Functions ------------------

def drone_start(drone_interface: DroneInterface) -> bool:
    """
    Start the drone mission: arm, switch to offboard, and take off.
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

def drone_run(drone_interface: DroneInterface, scenario_file: str) -> bool:
    """
    Run the mission:
      - Read scenario (start pose, obstacles, viewpoints)
      - Compute TSP ordering for the viewpoints
      - For each ordered waypoint, plan a path using RRTConnect and simulate following it.
      - Log positions and timing for performance evaluation.
    """
    print('Run mission using OMPL RRTConnect and TSP ordering')
    
    start_pose, obstacles, viewpoint_poses = read_scenario(scenario_file)
    print("Scenario loaded.")
    
    ordered_waypoints, tsp_distance = solve_tsp_ordering(start_pose, viewpoint_poses)
    print("TSP ordering of waypoints:", ordered_waypoints)
    
    current_pos = [start_pose['x'], start_pose['y'], start_pose['z']]
    overall_path = [current_pos]
    total_planning_time = 0.0
    total_path_length = 0.0
    
    position_log = []
    time_log = []
    start_time = time()
    
    # For each waypoint, plan a path and simulate movement
    for idx, waypoint in enumerate(ordered_waypoints):
        print(f"Planning path from {current_pos} to {waypoint}")
        path_segment, planning_time = plan_rrt_path(current_pos, waypoint, obstacles)
        if path_segment is None:
            print("Failed to plan path to waypoint", waypoint)
            return False
        print("Planned path segment:", path_segment)
        total_planning_time += planning_time
        
        seg_distance = sum(np.linalg.norm(np.array(path_segment[i]) - np.array(path_segment[i-1]))
                           for i in range(1, len(path_segment)))
        total_path_length += seg_distance
        
        for pt in path_segment:
            print(f"Moving to {pt}")
            sleep(SLEEP_TIME)
            current_pos = pt
            overall_path.append(pt)
            position_log.append(pt)
            time_log.append(time() - start_time)
    
    print("Mission completed.")
    print("Total planning time: {:.2f} s, Total path length: {:.2f} m".format(total_planning_time, total_path_length))
    
    # Plot and save performance graphs
    plot_performance(overall_path, ordered_waypoints, start_pose, position_log, time_log)
    
    return True

def drone_end(drone_interface: DroneInterface) -> bool:
    """
    End the mission: land the drone and switch to manual.
    """
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

# ------------------ Main ------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single drone mission with OMPL RRTConnect path planning')
    parser.add_argument('-n', '--namespace',
                        type=str, default='drone0',
                        help='ID of the drone to be used in the mission')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time',
                        action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('--scenario',
                        type=str,
                        default='scenarios/scenario1.yaml',
                        help='Path to scenario YAML file')
    
    args = parser.parse_args()
    drone_namespace = args.namespace
    verbosity = args.verbose
    use_sim_time = args.use_sim_time
    scenario_file = args.scenario
    
    print(f'Running mission for drone {drone_namespace} using scenario file {scenario_file}')
    
    rclpy.init()
    uav = DroneInterface(drone_id=drone_namespace,
                         use_sim_time=use_sim_time,
                         verbose=verbosity)
    
    success = drone_start(uav)
    if success:
        success = drone_run(uav, scenario_file)
    success = drone_end(uav)
    
    uav.shutdown()
    rclpy.shutdown()
    print('Clean exit')
    exit(0)
