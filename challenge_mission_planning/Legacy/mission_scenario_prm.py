#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Modified Mission Script for Structural Inspection Path Planning with ArUco
#
# This script integrates a TSP local search 3‑opt solver to determine the optimal visitation order
# of viewpoints and uses an OMPL-based PRM* algorithm to plan collision‑free
# paths between waypoints while avoiding cuboid obstacles. In addition, the planner
# verifies that the continuous path (using a fine interpolation) is free of collisions.
# If any segment is in collision, it automatically subdivides the segment by inserting
# intermediate waypoints and replanning.
#
# The key modification here is that we call planner.setMaxNearestNeighbors(20)
# to speed up roadmap construction while keeping "reasonably near-optimal" solutions.
# ------------------------------------------------------------------------------

# ------------------------
# Configuration (Modifiable Parameters)
# ------------------------

# Drone motion parameters
TAKE_OFF_HEIGHT = 1.0
TAKE_OFF_SPEED = 1.0
SLEEP_TIME = 0.05
SPEED = 1.0
LAND_SPEED = 0.5

# Obstacle avoidance parameters 
SAFETY_MARGIN = 0.8

# Collision checking parameters
COLLISION_CHECK_RESOLUTION = 0.5

# Recursive planning parameters
MAX_RECURSION_DEPTH = 100000

# PRM* planning parameters
PLANNING_TIME_LIMIT = 25.0
PLANNER_RANGE_SMALL = 1.0    # Not used by PRM*, but kept for consistency
PLANNER_RANGE_LARGE = 1.5    # Unused

# ------------------------
# Global Metrics and Data Logging
# ------------------------
fallback_count = 0
segment_planning_times = []
segment_lengths = []
global_prm_roadmap_data = []

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
import threading
import json
import os
import cv2
import cv2.aruco as aruco

from as2_python_api.drone_interface import DroneInterface
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# OMPL imports for PRM*
from ompl import base as ob
from ompl import geometric as og

# For TSP assignment
from scipy.optimize import linear_sum_assignment

# ------------------------
# Helper Functions
# ------------------------
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

def assign_viewpoints_to_markers(viewpoints, markers_list):
    n_view = len(viewpoints)
    n_mark = len(markers_list)
    if n_mark < n_view:
        print("[WARNING] Fewer markers than viewpoints! Some viewpoints may remain unassigned.")

    cost_matrix = np.zeros((n_view, n_mark))
    for i in range(n_view):
        vx, vy, vz = viewpoints[i]
        for j in range(n_mark):
            mx, my, mz = markers_list[j]["xyz"]
            dist = math.sqrt((vx - mx)**2 + (vy - my)**2 + (vz - mz)**2)
            cost_matrix[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assigned_marker_ids = [None] * n_view
    for i in range(len(row_ind)):
        v_idx = row_ind[i]
        m_idx = col_ind[i]
        assigned_marker_ids[v_idx] = markers_list[m_idx]["marker_id"]

    return assigned_marker_ids

# ------------------------
# TSP Local Search 3‑opt
# ------------------------
def apply_3opt_move(tour, i, j, k, distance_matrix):
    A = tour[:i]
    B = tour[i:j]
    C = tour[j:k]
    D = tour[k:]
    candidates = [
        A + B + C + D,
        A + B[::-1] + C + D,
        A + B + C[::-1] + D,
        A + B[::-1] + C[::-1] + D,
        A + C + B + D,
        A + C + B[::-1] + D,
        A + C[::-1] + B + D,
        A + C[::-1] + B[::-1] + D
    ]
    best_candidate = candidates[0]
    best_cost = sum(distance_matrix[best_candidate[m]][best_candidate[m+1]] 
                    for m in range(len(best_candidate)-1))
    for cand in candidates[1:]:
        cost = sum(distance_matrix[cand[m]][cand[m+1]] 
                   for m in range(len(cand)-1))
        if cost < best_cost:
            best_candidate = cand
            best_cost = cost
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
# PRM* Path Planning
# --------------------------
def plan_prmstar(start, goal, obstacles, bounds, planner_time_limit=PLANNING_TIME_LIMIT):
    """
    Create a PRM* planner, but to keep it faster:
      - setMaxNearestNeighbors(20) to cap how many neighbors each node tries to connect.
    """
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

    # Create a PRM* planner
    planner = og.PRMstar(si)
    # Limit the maximum number of neighbors to speed up roadmap construction
    planner.setMaxNearestNeighbors(20)

    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(planner_time_limit)
    if solved:
        # Extract the roadmap for visualization
        planner_data = ob.PlannerData(si)
        planner.getPlannerData(planner_data)
        roadmap_nodes = []
        roadmap_edges = []
        num_vertices = planner_data.numVertices()

        for i in range(num_vertices):
            vertex = planner_data.getVertex(i)
            state = vertex.getState()
            point = [state[0], state[1], state[2]]
            roadmap_nodes.append(point)

        for i in range(num_vertices):
            edge_list = []
            planner_data.getEdges(i, edge_list)
            for j in edge_list:
                v1 = planner_data.getVertex(i).getState()
                v2 = planner_data.getVertex(j).getState()
                p1 = [v1[0], v1[1], v1[2]]
                p2 = [v2[0], v2[1], v2[2]]
                if i < j:
                    roadmap_edges.append((p1, p2))

        global_prm_roadmap_data.append({"nodes": roadmap_nodes, "edges": roadmap_edges})

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

def plan_segment(start, goal, obstacles, bounds, planner_time_limit=PLANNING_TIME_LIMIT, recursion_depth=MAX_RECURSION_DEPTH):
    path, ptime = plan_prmstar(start, goal, obstacles, bounds, planner_time_limit)
    if path is None:
        return None, ptime
    if is_path_collision_free(path, obstacles):
        return path, ptime
    else:
        if recursion_depth <= 0:
            print("Max recursion reached; using current path anyway.")
            return path, ptime
        mid_point = [(s + g)/2.0 for s, g in zip(start, goal)]
        if not is_state_valid_cuboids(mid_point, obstacles):
            print("Midpoint invalid; abort segment.")
            return None, ptime
        print("Subdividing segment with midpoint:", mid_point)
        path1, time1 = plan_segment(start, mid_point, obstacles, bounds, planner_time_limit, recursion_depth-1)
        path2, time2 = plan_segment(mid_point, goal, obstacles, bounds, planner_time_limit, recursion_depth-1)
        if path1 is None or path2 is None:
            return None, ptime + time1 + time2
        combined = path1[:-1] + path2
        return combined, ptime + time1 + time2

def compute_path_smoothness(path):
    if len(path) < 3:
        return 0.0
    smoothness = 0.0
    for i in range(1, len(path)-1):
        p_prev = np.array(path[i-1])
        p_curr = np.array(path[i])
        p_next = np.array(path[i+1])
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        diff = (angle2 - angle1 + math.pi) % (2*math.pi) - math.pi
        smoothness += abs(diff)
    return smoothness

# --------------------------
# Plotting Functions
# --------------------------
def plot_paths(planned_paths, waypoints, ground_truth, obstacles=None, markers_list=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2D
    fig2d, ax2d = plt.subplots()
    fig2d.canvas.manager.set_window_title("2D Trajectory")
    for idx, path in enumerate(planned_paths):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax2d.plot(xs, ys, label=f"Planned Path {idx+1}", linewidth=2)
    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    ax2d.plot(wp_x, wp_y, 'ro-', label="Waypoints", linewidth=2)
    gt_x = [p[0] for p in ground_truth]
    gt_y = [p[1] for p in ground_truth]
    ax2d.plot(gt_x, gt_y, 'g--', label="Ground Truth", linewidth=2)

    if obstacles is not None:
        for obs in obstacles:
            ox, oy = obs["x"], obs["y"]
            dx, dy = obs["d"], obs["w"]
            lower_left = (ox - dx/2 - SAFETY_MARGIN, oy - dy/2 - SAFETY_MARGIN)
            rect = patches.Rectangle(
                lower_left, dx+2*SAFETY_MARGIN, dy+2*SAFETY_MARGIN,
                linewidth=1, edgecolor='k', facecolor='gray', alpha=0.3
            )
            ax2d.add_patch(rect)

    ax2d.set_xlabel("X (m)", fontsize=14)
    ax2d.set_ylabel("Y (m)", fontsize=14)
    ax2d.set_title("2D Trajectory", fontsize=16)
    legend_2d = ax2d.legend(fontsize=6)
    legend_2d.get_frame().set_alpha(0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "2D_Trajectory.png"), dpi=300)
    plt.show()

    # 3D
    fig3d = plt.figure()
    fig3d.canvas.manager.set_window_title("3D Trajectory")
    ax3d = fig3d.add_subplot(111, projection='3d')
    for idx, path in enumerate(planned_paths):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax3d.plot(xs, ys, zs, label=f"Planned Path {idx+1}", linewidth=2)

    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    wp_z = [p[2] for p in waypoints]
    ax3d.plot(wp_x, wp_y, wp_z, 'ro-', label="Waypoints", linewidth=2)
    gt_x = [p[0] for p in ground_truth]
    gt_y = [p[1] for p in ground_truth]
    gt_z = [p[2] for p in ground_truth]
    ax3d.plot(gt_x, gt_y, gt_z, 'g--', label="Ground Truth", linewidth=2)

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

    ax3d.set_xlabel("X (m)", fontsize=14)
    ax3d.set_ylabel("Y (m)", fontsize=14)
    ax3d.set_zlabel("Z (m)", fontsize=14)
    ax3d.set_title("3D Trajectory", fontsize=16)
    legend_3d = ax3d.legend(fontsize=6)
    legend_3d.get_frame().set_alpha(0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "3D_Trajectory.png"), dpi=300)
    plt.show()

def plot_costs(planning_times, path_lengths):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    segments = list(range(1, len(planning_times)+1))
    cumulative_cost = np.cumsum(np.array(planning_times) + np.array(path_lengths))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title("Segment Costs")

    # Planning Time
    ax1.bar(segments, planning_times, color='orange')
    ax1.set_xlabel("Segment", fontsize=14)
    ax1.set_ylabel("Planning Time (s)", fontsize=14)
    ax1.set_title("Planning Time per Segment", fontsize=16)
    ax1.legend(["Planning Time"], fontsize=10)

    # Cumulative Cost
    ax2.plot(segments, cumulative_cost, marker='o', color='purple')
    ax2.set_xlabel("Segment", fontsize=14)
    ax2.set_ylabel("Cumulative Cost (s + m)", fontsize=14)
    ax2.set_title("Cumulative Mission Cost", fontsize=16)
    ax2.legend(["Cumulative Cost"], fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Segment_Costs.png"), dpi=300)
    plt.show()

def plot_prm_roadmap(global_roadmap_data, waypoints=None, obstacles=None, planned_paths=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("PRM* Roadmap (2D)")

    plotted_nodes_legend = False
    for roadmap in global_roadmap_data:
        nodes = roadmap.get("nodes", [])
        edges = roadmap.get("edges", [])
        if nodes:
            xs = [n[0] for n in nodes]
            ys = [n[1] for n in nodes]
            if not plotted_nodes_legend:
                ax.scatter(xs, ys, c='blue', s=10, label="Roadmap Nodes")
                plotted_nodes_legend = True
            else:
                ax.scatter(xs, ys, c='blue', s=10)
        for edge in edges:
            (x1, y1, _), (x2, y2, _) = edge
            ax.plot([x1, x2], [y1, y2], 'c-', linewidth=0.5)

    if obstacles:
        for obs in obstacles:
            ox, oy = obs["x"], obs["y"]
            dx, dy = obs["d"], obs["w"]
            lower_left = (ox - dx/2 - SAFETY_MARGIN, oy - dy/2 - SAFETY_MARGIN)
            rect = patches.Rectangle(
                lower_left, dx+2*SAFETY_MARGIN, dy+2*SAFETY_MARGIN,
                linewidth=1, edgecolor='k', facecolor='gray', alpha=0.3
            )
            ax.add_patch(rect)

    if waypoints:
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]
        ax.plot(wp_x, wp_y, 'ro-', label="Waypoints", linewidth=2)

    if planned_paths:
        for idx, path in enumerate(planned_paths):
            xs = [pt[0] for pt in path]
            ys = [pt[1] for pt in path]
            if idx == 0:
                ax.plot(xs, ys, color='black', linewidth=2, label="Drone Path")
            else:
                ax.plot(xs, ys, color='black', linewidth=2)

    ax.set_title("PRM* Roadmap (2D)", fontsize=16)
    ax.set_xlabel("X (m)", fontsize=14)
    ax.set_ylabel("Y (m)", fontsize=14)
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "PRM_Roadmap_2D.png"), dpi=300)
    plt.show()

def plot_prm_roadmap_3d(global_roadmap_data, waypoints=None, obstacles=None, planned_paths=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure()
    fig.canvas.manager.set_window_title("PRM* Roadmap (3D)")
    ax3d = fig.add_subplot(111, projection='3d')

    plotted_nodes_legend = False
    for roadmap in global_roadmap_data:
        nodes = roadmap.get("nodes", [])
        edges = roadmap.get("edges", [])
        if nodes:
            xs = [n[0] for n in nodes]
            ys = [n[1] for n in nodes]
            zs = [n[2] for n in nodes]
            if not plotted_nodes_legend:
                ax3d.scatter(xs, ys, zs, c='blue', s=10, label="Roadmap Nodes")
                plotted_nodes_legend = True
            else:
                ax3d.scatter(xs, ys, zs, c='blue', s=10)
        for edge in edges:
            (x1, y1, z1), (x2, y2, z2) = edge
            ax3d.plot([x1, x2], [y1, y2], [z1, z2], 'c-', linewidth=0.5)

    if obstacles:
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

    if waypoints:
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]
        wp_z = [p[2] for p in waypoints]
        ax3d.plot(wp_x, wp_y, wp_z, 'ro-', label="Waypoints", linewidth=2)

    if planned_paths:
        for idx, path in enumerate(planned_paths):
            xs = [pt[0] for pt in path]
            ys = [pt[1] for pt in path]
            zs = [pt[2] for pt in path]
            if idx == 0:
                ax3d.plot(xs, ys, zs, color='black', linewidth=2, label="Drone Path")
            else:
                ax3d.plot(xs, ys, zs, color='black', linewidth=2)

    ax3d.set_title("PRM* Roadmap (3D)", fontsize=16)
    ax3d.set_xlabel("X (m)", fontsize=14)
    ax3d.set_ylabel("Y (m)", fontsize=14)
    ax3d.set_zlabel("Z (m)", fontsize=14)
    ax3d.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "PRM_Roadmap_3D.png"), dpi=300)
    plt.show()

def log_metrics_to_file(metrics, filename="metrics.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics logged to {file_path}")

# --------------------------
# Single-Node ArUco + Drone Interface
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
                    print(f"[DEBUG] Marker id {self.detected_marker_id} != expected {expected_marker_id}. Waiting...")
                    self.detected_marker_id = None
        print("Timeout: ignoring marker detection failure.")
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
        print(f"Expected marker id {expected_marker_id} not found in world data.")
        return False
    print(f"Waiting for marker id {expected_marker_id} at waypoint...")
    detected_id = drone_interface.wait_for_expected_marker(expected_marker_id, timeout=timeout)
    if detected_id is None:
        print("Marker verification fallback; continuing anyway.")
        return True
    print("Marker verification success.")
    return True

# --------------------------
# Drone Mission
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
    print("Running mission with PRM* (limited neighbors), TSP(3-opt), obstacle avoidance, marker verification")

    start_pose = scenario.get("drone_start_pose", {"x":0.0, "y":0.0, "z":0.0})
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
    print(f"TSP order: {permutation}, total distance: {tsp_distance:.2f}")

    ordered_points = [points[i] for i in permutation]
    ordered_yaws = [yaw_list[i] for i in permutation]
    ordered_expected_marker_ids = [expected_marker_ids[i] for i in permutation]

    ground_truth = ordered_points
    bounds = {"low": [-10, -10, 0], "high": [10, 10, 5]}

    planned_paths = []
    total_planning_time = 0.0
    total_path_length = 0.0

    for i in range(len(ordered_points) - 1):
        seg_start = ordered_points[i]
        seg_goal = ordered_points[i+1]
        print(f"Planning from {seg_start} to {seg_goal} (PRM*)")
        path, planning_time = plan_segment(seg_start, seg_goal, obstacles, bounds, PLANNING_TIME_LIMIT)
        if path is None:
            print("No solution for segment. Aborting.")
            return False, None, None
        print(f"Segment planned in {planning_time:.2f}s, states={len(path)}")
        total_planning_time += planning_time
        planned_paths.append(path)
        seg_length = sum(compute_euclidean_distance(path[j], path[j+1]) for j in range(len(path)-1))
        total_path_length += seg_length
        segment_planning_times.append(planning_time)
        segment_lengths.append(seg_length)

        current_state = path[0]
        N = len(path)

        # Final yaw for approach
        if i+1 < len(ordered_yaws) and ordered_yaws[i+1] is not None:
            dest_yaw = ordered_yaws[i+1]
        else:
            dx = seg_goal[0] - seg_start[0]
            dy = seg_goal[1] - seg_start[1]
            dest_yaw = math.atan2(dy, dx)

        for j, point in enumerate(path):
            if compute_euclidean_distance(current_state, point) < 0.1:
                print(f"Skipping {point} (already at location)")
                current_state = point
                continue
            t = j / (N - 1) if N > 1 else 1.0
            init_yaw = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0]) if N>1 else dest_yaw
            yaw_command = interpolate_angle(init_yaw, dest_yaw, t)
            print(f"Going to {point} with yaw={yaw_command:.2f}")
            success_move = drone_interface.go_to.go_to_point_with_yaw(point, angle=yaw_command, speed=SPEED)
            if not success_move:
                print(f"Failed to move to {point}")
                return False, None, None
            time.sleep(SLEEP_TIME)
            current_state = point

        time.sleep(1.0)
        current_point = path[-1]
        expected_marker_id = ordered_expected_marker_ids[i+1]
        if expected_marker_id is not None:
            print(f"Expected marker ID: {expected_marker_id}")
            if not verify_marker_and_adjust(
                drone_interface, current_point, seg_goal, dest_yaw,
                expected_marker_id, marker_ids_set
            ):
                print("Marker verification failed. Aborting.")
                return False, None, None
        else:
            print("No marker expected here.")

    # Return to start
    return_start = ordered_points[0]
    last_waypoint = ordered_points[-1]
    print("Plan return path to start.")
    path, planning_time = plan_segment(last_waypoint, return_start, obstacles, bounds, PLANNING_TIME_LIMIT)
    if path is None:
        print("No solution for return. Aborting.")
        return False, None, None
    print(f"Return segment in {planning_time:.2f}s, states={len(path)}")
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
            print(f"Skipping {point}")
            current_state = point
            continue
        yaw_command = math.atan2(return_start[1]-current_state[1], return_start[0]-current_state[0])
        print(f"Return: going to {point} yaw={yaw_command:.2f}")
        success_move = drone_interface.go_to.go_to_point_with_yaw(point, angle=yaw_command, speed=SPEED)
        if not success_move:
            print(f"Fail moving to {point}")
            return False, None, None
        time.sleep(SLEEP_TIME)
        current_state = point

    print("Landing...")
    land_success = drone_interface.land(speed=LAND_SPEED)
    print(f"Landing success: {land_success}")

    total_ground_truth_path_length = sum(
        compute_euclidean_distance(ordered_points[i], ordered_points[i+1]) 
        for i in range(len(ordered_points)-1)
    )
    smoothness_list = [compute_path_smoothness(path) for path in planned_paths]
    energy_estimate_list = [segment_lengths[i] + 0.5 * smoothness_list[i] for i in range(len(segment_lengths))]
    
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

# --------------------------
# Main
# --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single drone mission with PRM*, TSP (3‑opt), obstacle avoidance, and marker verification'
    )
    parser.add_argument('-n', '--namespace', type=str, default='drone0',
                        help='ID of the drone')
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
    print(f'Running mission for drone {drone_namespace}, scenario={scenario_file}')

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

    # Post-flight plotting
    if success and plot_data is not None:
        plot_paths(
            plot_data["planned_paths"],
            plot_data["ordered_points"],
            plot_data["ground_truth"],
            obstacles=plot_data["obstacles"],
            markers_list=plot_data["markers_list"]
        )
        plot_costs(plot_data["segment_planning_times"], plot_data["segment_lengths"])
        plot_prm_roadmap(
            global_prm_roadmap_data,
            waypoints=plot_data["ordered_points"],
            obstacles=plot_data["obstacles"],
            planned_paths=plot_data["planned_paths"]
        )
        plot_prm_roadmap_3d(
            global_prm_roadmap_data,
            waypoints=plot_data["ordered_points"],
            obstacles=plot_data["obstacles"],
            planned_paths=plot_data["planned_paths"]
        )

    print("Clean exit")
    exit(0)
