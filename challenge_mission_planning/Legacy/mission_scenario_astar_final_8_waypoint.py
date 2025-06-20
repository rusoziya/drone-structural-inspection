#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Modified Mission Script for Structural Inspection Path Planning with ArUco
#
# This script integrates a TSP local search 3‑opt solver to determine the optimal visitation order
# of viewpoints and uses an A* algorithm to plan collision‑free paths between waypoints while avoiding 
# cuboid obstacles. In addition, the planner verifies that the continuous path (using a fine interpolation)
# is free of collisions. If any segment is in collision, it automatically subdivides the segment by inserting
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
# 5. An A* algorithm is used to compute a continuous 3D path, which is then used directly (no post‐smoothing is performed).
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
SAFETY_MARGIN = 0.5        # Additional margin (in meters) added around each obstacle 

# Collision checking parameters (used when interpolating along the planned path)
COLLISION_CHECK_RESOLUTION = 0.5  # Step size for interpolation (in meters)

# Recursive planning parameters for subdividing segments in collision 
MAX_RECURSION_DEPTH = 100000     # Maximum times a segment wwill be subdivided if in collision 

# A* planner parameters 
PLANNING_TIME_LIMIT = 1.0         # Time limit for planning each segment (in seconds) 
ASTAR_RESOLUTION_DEFAULT = 0.5     # Default grid resolution for A* (in meters)

# ------------------------
# Global Metrics and Data Logging Variables
# ------------------------
fallback_count = 0
segment_planning_times = []  # list of planning times per segment
segment_lengths = []         # list of segment lengths
global_astar_tree_data = []  # list to store A* tree data for each segment

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
from matplotlib.lines import Line2D

# For ArUco detection (now enabled in mission logic)
import cv2
import cv2.aruco as aruco
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# OMPL imports (no longer used for planning but kept for compatibility)
from ompl import base as ob
from ompl import geometric as og

# For the assignment
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

# ------------------------
# OUTPUT DIRECTORY SETUP
# ------------------------
script_path = os.path.abspath(__file__)
script_name = os.path.splitext(os.path.basename(script_path))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(script_path), script_name)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

    assigned_marker_ids = [None] * n_view
    for i in range(len(row_ind)):
        v_idx = row_ind[i]
        m_idx = col_ind[i]
        assigned_marker_ids[v_idx] = markers_list[m_idx]["marker_id"]

    return assigned_marker_ids

# ------------------------
# TSP Local Search 3‑opt Solver
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
# A* Path Planning
# --------------------------
def plan_astar(start, goal, obstacles, bounds, resolution, planning_time_limit=PLANNING_TIME_LIMIT):
    """
    Plan a collision-free path between start and goal using a grid-based A* search.
    Before starting the grid search, the function checks if the direct path is collision-free.
    If so, it returns [start, goal] immediately. Otherwise, the search is performed on a grid
    with the given resolution (which is set to 0.5 m here).
    Additionally, during the search, if from any node the direct line to the goal is free,
    the algorithm shortcuts to the goal.
    Returns a list of 3D points (the path) and the planning time.
    """
    start_time = time.time()
    # If the straight-line path is free, skip intermediate waypoints.
    if is_path_collision_free([start, goal], obstacles):
        planning_time = time.time() - start_time
        return [start, goal], planning_time

    low_x, low_y, low_z = bounds["low"][0], bounds["low"][1], bounds["low"][2]
    high_x, high_y, high_z = bounds["high"][0], bounds["high"][1], bounds["high"][2]

    def world_to_grid(point):
        i = int(round((point[0] - low_x) / resolution))
        j = int(round((point[1] - low_y) / resolution))
        k = int(round((point[2] - low_z) / resolution))
        return (i, j, k)
    
    def grid_to_world(cell):
        x = low_x + cell[0] * resolution
        y = low_y + cell[1] * resolution
        z = low_z + cell[2] * resolution
        return (x, y, z)
    
    start_cell = world_to_grid(start)
    goal_cell = world_to_grid(goal)

    max_i = int(round((high_x - low_x) / resolution))
    max_j = int(round((high_y - low_y) / resolution))
    max_k = int(round((high_z - low_z) / resolution))
    
    from heapq import heappush, heappop
    open_set = []
    heappush(open_set, (0, start_cell))
    came_from = {}
    cost_so_far = {start_cell: 0}
    
    # For tree visualization, record visited nodes and edges.
    visited_nodes = set()
    tree_edges = []
    
    found = False
    while open_set:
        if time.time() - start_time > planning_time_limit:
            print("A* planning time limit exceeded")
            break
        current_priority, current = heappop(open_set)
        visited_nodes.add(current)
        # Check if current cell is goal
        if current == goal_cell:
            found = True
            break
        # Shortcut: if direct path from current to goal is free, jump to goal.
        if is_path_collision_free([grid_to_world(current), goal], obstacles):
            came_from[goal_cell] = current
            found = True
            current = goal_cell
            break
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if (neighbor[0] < 0 or neighbor[0] > max_i or
                        neighbor[1] < 0 or neighbor[1] > max_j or
                        neighbor[2] < 0 or neighbor[2] > max_k):
                        continue
                    neighbor_world = grid_to_world(neighbor)
                    if not is_state_valid_cuboids(neighbor_world, obstacles):
                        continue
                    current_world = grid_to_world(current)
                    if not is_path_collision_free([current_world, neighbor_world], obstacles, resolution=resolution/2):
                        continue
                    move_cost = math.sqrt((dx*resolution)**2 + (dy*resolution)**2 + (dz*resolution)**2)
                    new_cost = cost_so_far[current] + move_cost
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        neighbor_world = grid_to_world(neighbor)
                        goal_world = goal  # goal is given in world coordinates
                        heuristic = math.sqrt((neighbor_world[0]-goal_world[0])**2 +
                                              (neighbor_world[1]-goal_world[1])**2 +
                                              (neighbor_world[2]-goal_world[2])**2)
                        priority = new_cost + heuristic
                        heappush(open_set, (priority, neighbor))
                        came_from[neighbor] = current
                        tree_edges.append((grid_to_world(current), grid_to_world(neighbor)))
    planning_time = time.time() - start_time
    if not found:
        return None, planning_time
    # Reconstruct path
    path_cells = []
    current = goal_cell
    while current != start_cell:
        path_cells.append(current)
        current = came_from[current]
    path_cells.append(start_cell)
    path_cells.reverse()
    path_states = [grid_to_world(cell) for cell in path_cells]
    tree_data = {"nodes": [grid_to_world(cell) for cell in visited_nodes], "edges": tree_edges}
    global_astar_tree_data.append(tree_data)
    return path_states, planning_time

# --------------------------
# A* Path Planning-based Segment Planner
# --------------------------
def plan_segment(start, goal, obstacles, bounds, astar_resolution,
                 planning_time_limit, recursion_depth=MAX_RECURSION_DEPTH):
    """
    Recursively plan a collision-free path between 'start' and 'goal' using A*,
    subdividing if collisions are found. This function ensures that if a segment
    is subdivided at 'mid_point', both sub-paths share that exact point, so there
    is no gap in the final trajectory.
    """
    path, ptime = plan_astar(
        start, goal, obstacles, bounds,
        resolution=astar_resolution,
        planning_time_limit=planning_time_limit
    )
    if path is None:
        return None, ptime

    # If the A* path is fully collision-free (checked at fine resolution),
    # return it as-is
    if is_path_collision_free(path, obstacles):
        return path, ptime
    else:
        # We must subdivide around the arithmetic midpoint
        if recursion_depth <= 0:
            print("Max recursion reached; using the current path even if near obstacles.")
            return path, ptime

        mid_point = [(s + g) / 2.0 for s, g in zip(start, goal)]
        if not is_state_valid_cuboids(mid_point, obstacles):
            print("Arithmetic midpoint is invalid; aborting segment.")
            return None, ptime

        print(f"Subdividing segment around midpoint: {mid_point}")

        path1, time1 = plan_segment(
            start, mid_point, obstacles, bounds,
            astar_resolution, planning_time_limit,
            recursion_depth - 1
        )
        path2, time2 = plan_segment(
            mid_point, goal, obstacles, bounds,
            astar_resolution, planning_time_limit,
            recursion_depth - 1
        )

        if path1 is None or path2 is None:
            # If either sub-path fails, the entire segment fails
            return None, ptime + (time1 or 0) + (time2 or 0)

        # -------------------------------
        #  FIX: Force both sub-paths to meet exactly at mid_point
        # -------------------------------
        import numpy as np

        # If path1[-1] is close to mid_point, snap it exactly:
        if np.allclose(path1[-1], mid_point, atol=1e-8):
            path1[-1] = mid_point
        else:
            # If for some reason path1 doesn't end near mid_point, forcibly append it
            path1.append(mid_point)

        # If path2[0] is close to mid_point, snap it exactly:
        if np.allclose(path2[0], mid_point, atol=1e-8):
            path2[0] = mid_point
        else:
            # If for some reason path2 doesn't start near mid_point, forcibly insert it
            path2.insert(0, mid_point)

        # Now combine, skipping the duplicated midpoint in path1
        combined = path1[:-1] + path2
        return combined, ptime + time1 + time2

def compute_path_smoothness(path):
    """
    Compute a smoothness metric for a given path.
    Here, smoothness is estimated as the sum of absolute changes in travel direction (in radians).
    """
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

# --------------------------
# Plotting Functions
# --------------------------
def plot_costs(planning_times, path_lengths):
    segments = list(range(1, len(planning_times)+1))
    cumulative_cost = np.cumsum(np.array(planning_times) + np.array(path_lengths))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title("Segment Costs")

    ax1.bar(segments, planning_times, color='orange')
    ax1.set_xlabel("Segment", fontsize=14)
    ax1.set_ylabel("Planning Time (s)", fontsize=14)
    ax1.set_title("Planning Time per Segment", fontsize=16)
    legend_ax1 = ax1.legend(["Planning Time"], fontsize=10)
    legend_ax1.get_frame().set_alpha(0.6)

    ax2.plot(segments, cumulative_cost, marker='o', color='purple')
    ax2.set_xlabel("Segment", fontsize=14)
    ax2.set_ylabel("Cumulative Cost (s + m)", fontsize=14)
    ax2.set_title("Cumulative Mission Cost", fontsize=16)
    legend_ax2 = ax2.legend(["Cumulative Cost"], fontsize=10)
    legend_ax2.get_frame().set_alpha(0.6)

    plt.tight_layout()
    file_path_costs = os.path.join(OUTPUT_DIR, "Segment_Costs.png")
    plt.savefig(file_path_costs, dpi=300)
    plt.show()

def plot_astar_tree(astar_tree_data, waypoints=None, obstacles=None, planned_paths=None):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("A* Tree (2D Projection)")
    
    plotted_astar_nodes_legend = False
    
    for tree in astar_tree_data:
        nodes = tree.get("nodes", [])
        edges = tree.get("edges", [])
        if nodes:
            xs = [n[0] for n in nodes]
            ys = [n[1] for n in nodes]
            if not plotted_astar_nodes_legend:
                ax.scatter(xs, ys, c='blue', s=10, label="A* Nodes")
                plotted_astar_nodes_legend = True
            else:
                ax.scatter(xs, ys, c='blue', s=10)
        for edge in edges:
            (x1, y1, _), (x2, y2, _) = edge
            ax.plot([x1, x2], [y1, y2], 'c-', linewidth=0.5)

    if obstacles is not None:
        for obs in obstacles:
            ox, oy = obs["x"], obs["y"]
            dx, dy = obs["d"], obs["w"]
            lower_left = (ox - dx/2 - SAFETY_MARGIN, oy - dy/2 - SAFETY_MARGIN)
            rect = patches.Rectangle(
                lower_left, dx + 2*SAFETY_MARGIN, dy + 2*SAFETY_MARGIN,
                linewidth=1, edgecolor='k', facecolor='gray', alpha=0.3
            )
            ax.add_patch(rect)

    if waypoints is not None:
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]
        ax.plot(wp_x, wp_y, 'ro-', label="Waypoints", linewidth=2)

    if planned_paths is not None:
        for idx, path in enumerate(planned_paths):
            xs = [pt[0] for pt in path]
            ys = [pt[1] for pt in path]
            if idx == 0:
                ax.plot(xs, ys, color='black', linewidth=2, label="Drone Path")
            else:
                ax.plot(xs, ys, color='black', linewidth=2)

    ax.set_title("A* Tree (2D Projection)", fontsize=16)
    ax.set_xlabel("X (m)", fontsize=14)
    ax.set_ylabel("Y (m)", fontsize=14)
    legend_2d_astar = ax.legend(fontsize=6)
    legend_2d_astar.get_frame().set_alpha(0.6)

    plt.tight_layout()
    file_path_astar = os.path.join(OUTPUT_DIR, "Astar_Tree.png")
    plt.savefig(file_path_astar, dpi=300)
    plt.show()

def plot_astar_tree_3d(astar_tree_data, waypoints=None, obstacles=None, planned_paths=None):
    fig = plt.figure()
    fig.canvas.manager.set_window_title("A* Tree (3D Projection)")
    ax3d = fig.add_subplot(111, projection='3d')

    plotted_astar_nodes_legend = False

    for tree in astar_tree_data:
        nodes = tree.get("nodes", [])
        edges = tree.get("edges", [])
        if nodes:
            xs = [n[0] for n in nodes]
            ys = [n[1] for n in nodes]
            zs = [n[2] for n in nodes]
            if not plotted_astar_nodes_legend:
                ax3d.scatter(xs, ys, zs, c='blue', s=10, label="A* Nodes")
                plotted_astar_nodes_legend = True
            else:
                ax3d.scatter(xs, ys, zs, c='blue', s=10)
        for edge in edges:
            (x1, y1, z1), (x2, y2, z2) = edge
            ax3d.plot([x1, x2], [y1, y2], [z1, z2], 'c-', linewidth=0.5)

    if obstacles is not None:
        for obs in obstacles:
            ox, oy, oz = obs["x"], obs["y"], obs["z"]
            dx, dy, dz = obs["d"], obs["w"], obs["h"]
            x = ox - dx/2 - SAFETY_MARGIN
            y = oy - dy/2 - SAFETY_MARGIN
            z = oz - dz/2 - SAFETY_MARGIN
            cuboid = [
                [x, y, z],
                [x+dx+2*SAFETY_MARGIN, y, z],
                [x+dx+2*SAFETY_MARGIN, y+dy+2*SAFETY_MARGIN, z],
                [x, y+dy+2*SAFETY_MARGIN, z],
                [x, y, z+dz+2*SAFETY_MARGIN],
                [x+dx+2*SAFETY_MARGIN, y, z+dz+2*SAFETY_MARGIN],
                [x+dx+2*SAFETY_MARGIN, y+dy+2*SAFETY_MARGIN, z+dz+2*SAFETY_MARGIN],
                [x, y+dy+2*SAFETY_MARGIN, z+dz+2*SAFETY_MARGIN]
            ]
            edges = [
                (0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7)
            ]
            for e in edges:
                pt1 = cuboid[e[0]]
                pt2 = cuboid[e[1]]
                ax3d.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k-', alpha=0.5)

    if waypoints is not None:
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]
        wp_z = [p[2] for p in waypoints]
        ax3d.plot(wp_x, wp_y, wp_z, 'ro-', label="Waypoints", linewidth=2)

    if planned_paths is not None:
        for idx, path in enumerate(planned_paths):
            xs = [pt[0] for pt in path]
            ys = [pt[1] for pt in path]
            zs = [pt[2] for pt in path]
            if idx == 0:
                ax3d.plot(xs, ys, zs, color='black', linewidth=2, label="Drone Path")
            else:
                ax3d.plot(xs, ys, zs, color='black', linewidth=2)

    ax3d.set_title("A* Tree (3D Projection)", fontsize=16)
    ax3d.set_xlabel("X (m)", fontsize=14)
    ax3d.set_ylabel("Y (m)", fontsize=14)
    ax3d.set_zlabel("Z (m)", fontsize=14)

    legend_3d_astar = ax3d.legend(fontsize=6)
    legend_3d_astar.get_frame().set_alpha(0.6)

    plt.tight_layout()
    file_path_astar_3d = os.path.join(OUTPUT_DIR, "Astar_Tree_3D.png")
    plt.savefig(file_path_astar_3d, dpi=300)
    plt.show()


def plot_paths(planned_paths, waypoints, obstacles=None, markers_list=None):
    """
    Plots:
      - The planned paths in both 2D and 3D.
      - Waypoints as red markers with numeric labels (no connecting lines).
      - Obstacles as nested shapes:
         * A red wireframe for the real obstacle.
         * A black wireframe (with reduced alpha) for the safety margin.
      - Each path segment in 3D has a different color, and we snap endpoints
        to avoid tiny visual gaps.
    """
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    # Safety margin from global
    global SAFETY_MARGIN

    # ----------------------------
    # HELPER FUNCTIONS
    # ----------------------------
    def draw_obstacle_2d(ax, ox, oy, dx, dy):
        rx = ox - dx/2
        ry = oy - dy/2
        mx = rx - SAFETY_MARGIN
        my = ry - SAFETY_MARGIN
        # Real obstacle outline (red)
        real_rect = patches.Rectangle(
            (rx, ry),
            dx,
            dy,
            linewidth=1,
            edgecolor='red',
            facecolor='none',
            label='_nolegend_'
        )
        ax.add_patch(real_rect)
        # Margin outline (black, alpha=0.2)
        margin_rect = patches.Rectangle(
            (mx, my),
            dx + 2*SAFETY_MARGIN,
            dy + 2*SAFETY_MARGIN,
            linewidth=0.5,
            edgecolor='black',
            facecolor='none',
            alpha=0.2,
            label='_nolegend_'
        )
        ax.add_patch(margin_rect)

    def draw_obstacle_3d_wireframe(ax3d, ox, oy, oz, dx, dy, dz):
        """
        Draw the real obstacle in red, and its margin in black wireframe with alpha=0.2.
        """
        rx1 = ox - dx/2
        rx2 = ox + dx/2
        ry1 = oy - dy/2
        ry2 = oy + dy/2
        rz1 = oz - dz/2
        rz2 = oz + dz/2

        mx1 = rx1 - SAFETY_MARGIN
        mx2 = rx2 + SAFETY_MARGIN
        my1 = ry1 - SAFETY_MARGIN
        my2 = ry2 + SAFETY_MARGIN
        mz1 = rz1 - SAFETY_MARGIN
        mz2 = rz2 + SAFETY_MARGIN

        # Real obstacle edges (red)
        # Bottom face
        ax3d.plot([rx1, rx2], [ry1, ry1], [rz1, rz1], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx2], [ry1, ry2], [rz1, rz1], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx1], [ry2, ry2], [rz1, rz1], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx1, rx1], [ry2, ry1], [rz1, rz1], 'r-', linewidth=1, label='_nolegend_')
        # Top face
        ax3d.plot([rx1, rx2], [ry1, ry1], [rz2, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx2], [ry1, ry2], [rz2, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx1], [ry2, ry2], [rz2, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx1, rx1], [ry2, ry1], [rz2, rz2], 'r-', linewidth=1, label='_nolegend_')
        # Vertical edges
        ax3d.plot([rx1, rx1], [ry1, ry1], [rz1, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx2], [ry1, ry1], [rz1, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx2], [ry2, ry2], [rz1, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx1, rx1], [ry2, ry2], [rz1, rz2], 'r-', linewidth=1, label='_nolegend_')

        # Margin edges (black, alpha=0.2)
        # Bottom face
        ax3d.plot([mx1, mx2], [my1, my1], [mz1, mz1], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx2], [my1, my2], [mz1, mz1], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx1], [my2, my2], [mz1, mz1], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx1, mx1], [my2, my1], [mz1, mz1], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        # Top face
        ax3d.plot([mx1, mx2], [my1, my1], [mz2, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx2], [my1, my2], [mz2, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx1], [my2, my2], [mz2, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx1, mx1], [my2, my1], [mz2, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        # Vertical edges
        ax3d.plot([mx1, mx1], [my1, my1], [mz1, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx2], [my1, my1], [mz1, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx2], [my2, my2], [mz1, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx1, mx1], [my2, my2], [mz1, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')

    # ----------------------------
    # 2D Trajectory
    # ----------------------------
    fig2d, ax2d = plt.subplots()
    fig2d.canvas.manager.set_window_title("2D Trajectory")

    for idx, path in enumerate(planned_paths):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax2d.plot(xs, ys, label=f"Planned Path {idx+1}", linewidth=2)

    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    ax2d.scatter(wp_x, wp_y, c='r', marker='o', label="Waypoints")

    for i, (x, y) in enumerate(zip(wp_x, wp_y)):
        ax2d.text(x, y + 0.2, f"{i+1}", ha='center', va='bottom', fontsize=8, color='blue')

    # Draw 2D obstacles
    if obstacles:
        for obs in obstacles:
            ox, oy = obs["x"], obs["y"]
            dx, dy = obs["d"], obs["w"]
            draw_obstacle_2d(ax2d, ox, oy, dx, dy)

    handles_2d, labels_2d = ax2d.get_legend_handles_labels()
    real_patch_2d = patches.Patch(facecolor='none', edgecolor='red')
    margin_patch_2d = patches.Patch(facecolor='none', edgecolor='black', alpha=0.2, linewidth=0.5)

    if obstacles:
        handles_2d.append(real_patch_2d)
        labels_2d.append("Real Obstacle")
        handles_2d.append(margin_patch_2d)
        labels_2d.append("Margin")

    ax2d.set_xlabel("X (m)", fontsize=14)
    ax2d.set_ylabel("Y (m)", fontsize=14)
    ax2d.set_title("2D Trajectory of Planned Paths", fontsize=16)

    legend_2d = ax2d.legend(handles=handles_2d, labels=labels_2d, fontsize=6)
    legend_2d.get_frame().set_alpha(0.6)

    plt.tight_layout()
    file_path_2d = os.path.join(OUTPUT_DIR, "2D_Trajectory.png")
    plt.savefig(file_path_2d, dpi=300)
    plt.show()

    # ----------------------------
    # 3D Trajectory
    # ----------------------------
    fig3d = plt.figure()
    fig3d.canvas.manager.set_window_title("3D Trajectory")
    ax3d = fig3d.add_subplot(111, projection='3d')

    # Use a colormap so each path segment has a unique color
    import matplotlib.cm as cm
    colors = cm.get_cmap("tab10", len(planned_paths))

    # Snap each segment’s start to the previous segment’s end, so no small gap
    for idx, path in enumerate(planned_paths):
        if idx > 0:
            prev_path = planned_paths[idx - 1]
            if np.allclose(prev_path[-1], path[0], atol=1e-8):
                path[0] = prev_path[-1]
            else:
                path.insert(0, prev_path[-1])

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax3d.plot(xs, ys, zs,
                  color=colors(idx),
                  linewidth=2,
                  label=f"Planned Path {idx+1}")

    # Plot waypoints
    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    wp_z = [p[2] for p in waypoints]
    ax3d.scatter(wp_x, wp_y, wp_z, c='r', marker='o', label="Waypoints")

    for i, (x, y, z) in enumerate(zip(wp_x, wp_y, wp_z)):
        ax3d.text(x, y, z + 0.2, f"{i+1}", fontsize=8, color='blue')

    # Draw 3D obstacles
    if obstacles:
        for obs in obstacles:
            ox, oy, oz = obs["x"], obs["y"], obs["z"]
            dx, dy, dz = obs["d"], obs["w"], obs["h"]
            draw_obstacle_3d_wireframe(ax3d, ox, oy, oz, dx, dy, dz)

    # Build 3D legend
    handles_3d, labels_3d = ax3d.get_legend_handles_labels()
    real_line = Line2D([0], [0], color='red', linewidth=1)
    margin_line = Line2D([0], [0], color='black', linewidth=0.5, alpha=0.4)

    if obstacles:
        handles_3d.append(real_line)
        labels_3d.append("Real Obstacle")
        handles_3d.append(margin_line)
        labels_3d.append("Margin")

    ax3d.set_xlabel("X (m)", fontsize=14)
    ax3d.set_ylabel("Y (m)", fontsize=14)
    ax3d.set_zlabel("Z (m)", fontsize=14)
    ax3d.set_title("3D Trajectory of Planned Paths", fontsize=16)
    ax3d.set_box_aspect([1, 1, 1])

    legend_3d = ax3d.legend(handles=handles_3d, labels=labels_3d, fontsize=6)
    legend_3d.get_frame().set_alpha(0.6)

    plt.tight_layout()
    file_path_3d = os.path.join(OUTPUT_DIR, "3D_Trajectory.png")
    plt.savefig(file_path_3d, dpi=300)
    plt.show()

def log_metrics_to_file(metrics, filename="metrics.json"):
    file_path = os.path.join(OUTPUT_DIR, filename)
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics logged to {file_path}")

# --------------------------
# Single-Node ArUco Detection + Drone Interface
# --------------------------
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
                    print(f"[DEBUG] Marker id {self.detected_marker_id} does not match expected marker {expected_marker_id}. Ignoring and waiting...")
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

# --------------------------
# Marker Verification
# --------------------------
def verify_marker_and_adjust(drone_interface: SingleNodeDroneMission,
                             current_point, target_point, expected_yaw, expected_marker_id,
                             marker_ids_set, timeout=5.0):
    if expected_marker_id not in marker_ids_set:
        print(f"Expected marker id {expected_marker_id} not found in world.yaml.")
        return False

    print(f"Waiting for marker id {expected_marker_id} at waypoint...")
    detected_id = drone_interface.wait_for_expected_marker(expected_marker_id, timeout=timeout)
    if detected_id is None:
        print("Marker verification fallback triggered; proceeding without marker verification.")
        return True

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
    print("Manual")
    success = drone_interface.manual()
    print(f"Manual success: {success}")
    return success

def drone_run(drone_interface: SingleNodeDroneMission, scenario: dict):
    """
    Run the mission:
      - Parse the scenario for starting pose, obstacles, and viewpoint poses.
      - Assign each viewpoint to a unique marker (Hungarian assignment).
      - Solve the TSP for optimal visitation using a local search 3‑opt solver.
      - For each segment between ordered points, plan a collision-free path using 
        an A* planner. If a path is not collision-free, intermediate waypoints are 
        inserted automatically via recursive subdivision using an arithmetic midpoint.
      - The drone follows the computed path with continuous yaw interpolation.
      - After fully reaching each waypoint, marker verification is performed.
      - After visiting the last waypoint, the drone plans a return path to the starting pose and lands immediately.
    """
    print("Run mission with A* planning, TSP (3‑opt local search) optimization, continuous commands, obstacle avoidance, and marker verification")
    mission_start = time.time()  # <--- ADDED: Record mission start time

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

    bounds = {"low": [-10, -10, 0], "high": [10, 10, 5]}

    planned_paths = []
    total_planning_time = 0.0
    total_path_length = 0.0
    fixed_astar_resolution = ASTAR_RESOLUTION_DEFAULT

    for i in range(len(ordered_points) - 1):
        seg_start = ordered_points[i]
        seg_goal = ordered_points[i+1]
        astar_resolution = fixed_astar_resolution

        dest_yaw = (ordered_yaws[i+1] if ordered_yaws[i+1] is not None 
                    else math.atan2(seg_goal[1]-seg_start[1], seg_goal[0]-seg_start[0]))
        print(f"Planning path from {seg_start} to {seg_goal} with grid resolution {astar_resolution}")

        path, planning_time = plan_segment(seg_start, seg_goal, obstacles, bounds, astar_resolution, PLANNING_TIME_LIMIT)
        if path is None:
            print("No solution found for segment, aborting mission planning.")
            return False, None, None
        print(f"Segment planned in {planning_time:.2f} s with {len(path)} states.")
        total_planning_time += planning_time
        planned_paths.append(path)
        seg_length = sum(compute_euclidean_distance(path[j], path[j+1]) for j in range(len(path)-1))
        total_path_length += seg_length

        segment_planning_times.append(planning_time)
        segment_lengths.append(seg_length)

        current_state = path[0]
        N = len(path)
        for j, point in enumerate(path):
            if compute_euclidean_distance(current_state, point) < 0.1:
                print(f"Skipping point {point} (already reached)")
                current_state = point
                continue
            t = j / (N - 1) if N > 1 else 1.0
            yaw_command = interpolate_angle(
                dest_yaw if j == N - 1 else math.atan2(path[1][1] - path[0][1],
                                                       path[1][0] - path[0][0]),
                dest_yaw, t)
            print(f"Continuously moving to point: {point} with yaw: {yaw_command:.2f} rad")
            success_move = drone_interface.go_to.go_to_point_with_yaw(point, angle=yaw_command, speed=SPEED)
            if not success_move:
                print(f"Failed to move to point: {point}")
                return False, None, None
            time.sleep(SLEEP_TIME)
            current_state = point

        time.sleep(1.0)
        current_point = path[-1]
        expected_marker_id = ordered_expected_marker_ids[i+1]
        if expected_marker_id is not None:
            print(f"Expected marker for this waypoint: {expected_marker_id}")
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

    return_start = ordered_points[0]
    last_waypoint = ordered_points[-1]
    print("Planning return path from last waypoint to starting position")
    path, planning_time = plan_segment(last_waypoint, return_start, obstacles, bounds, fixed_astar_resolution, PLANNING_TIME_LIMIT)
    if path is None:
        print("No solution found for return segment, aborting mission planning.")
        return False, None, None
    print(f"Return segment planned in {planning_time:.2f} s with {len(path)} states.")
    total_planning_time += planning_time
    planned_paths.append(path)
    seg_length = sum(compute_euclidean_distance(path[j], path[j+1]) for j in range(len(path)-1))
    total_path_length += seg_length
    segment_planning_times.append(planning_time)
    segment_lengths.append(seg_length)
    
    current_state = path[0]
    N = len(path)
    for j, point in enumerate(path):
        if compute_euclidean_distance(current_state, point) < 0.1:
            print(f"Skipping point {point} (already reached)")
            current_state = point
            continue
        t = j / (N - 1) if N > 1 else 1.0
        yaw_command = math.atan2(return_start[1]-current_state[1], return_start[0]-current_state[0])
        print(f"Return path moving to point: {point} with yaw: {yaw_command:.2f} rad")
        success_move = drone_interface.go_to.go_to_point_with_yaw(point, angle=yaw_command, speed=SPEED)
        if not success_move:
            print(f"Failed to move to point: {point} on return path")
            return False, None, None
        time.sleep(SLEEP_TIME)
        current_state = point
    print("Return to starting position complete. Landing now.")
    land_success = drone_interface.land(speed=LAND_SPEED)
    print(f"Landing success: {land_success}")

    total_ground_truth_path_length = sum(
        compute_euclidean_distance(ordered_points[i], ordered_points[i+1]) 
        for i in range(len(ordered_points)-1)
    )
    smoothness_list = [compute_path_smoothness(path) for path in planned_paths]
    energy_estimate_list = [segment_lengths[i] + 0.5 * smoothness_list[i] for i in range(len(segment_lengths))]
    mission_duration = time.time() - mission_start  # <--- ADDED: Compute total mission time
    
    metrics = {
        "total_planning_time": total_planning_time,
        "total_path_length": total_path_length,
        "segment_planning_times": segment_planning_times,
        "segment_lengths": segment_lengths,
        "fallback_count": drone_interface.fallback_count,
        "energy_estimate": energy_estimate_list,
        "path_smoothness": smoothness_list,
        "shortest_possible_path_length": tsp_distance,
        "total_ground_truth_path_length": total_ground_truth_path_length,
        "total_mission_time": mission_duration  # <--- ADDED: Total mission time metric
    }
    
    log_metrics_to_file(metrics)
    
    plot_data = {
         "planned_paths": planned_paths,
         "ordered_points": ordered_points,
         "segment_planning_times": segment_planning_times,
         "segment_lengths": segment_lengths,
         "obstacles": obstacles,
         "markers_list": markers_list,
         "fallback_count": drone_interface.fallback_count
    }
    
    return True, plot_data, metrics

# --------------------------
# Main
# --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single drone mission with A* planning, TSP (3‑opt local search) optimization, obstacle avoidance, and marker verification enabled'
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
        plot_paths(
            plot_data["planned_paths"],
            plot_data["ordered_points"],
            obstacles=plot_data["obstacles"],
            markers_list=plot_data["markers_list"]
        )
        plot_costs(
            plot_data["segment_planning_times"],
            plot_data["segment_lengths"]
        )
        plot_astar_tree(
            global_astar_tree_data,
            waypoints=plot_data["ordered_points"],
            obstacles=plot_data["obstacles"],
            planned_paths=plot_data["planned_paths"]
        )
        plot_astar_tree_3d(
            global_astar_tree_data,
            waypoints=plot_data["ordered_points"],
            obstacles=plot_data["obstacles"],
            planned_paths=plot_data["planned_paths"]
        )
    
    os.system("./stop.bash")
    print("Clean exit")
    exit(0)
