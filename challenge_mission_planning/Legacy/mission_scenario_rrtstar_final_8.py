#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Modified Mission Script for Structural Inspection Path Planning with ArUco
#
# This script integrates a TSP local search 3‑opt solver to determine the optimal visitation order
# of viewpoints and uses an OMPL‑based RRT* algorithm to plan collision‑free
# paths between waypoints while avoiding cuboid obstacles. In addition, the planner
# verifies that the continuous path (using a fine interpolation) is free of collisions.
# If any segment is in collision, it automatically subdivides the segment by inserting
# intermediate waypoints and replanning.
#
# A new B‑Spline smoothing function is integrated here that smooths the continuous path
# returned by RRT*. Instead of sending individual waypoint commands, the dense,
# smooth path is converted to a ROS Path message and sent as a single command to the
# TrajectoryGenerationModule. This module handles true continuous trajectory following.
#
# *CHANGE*: Instead of following the path in "path facing" mode and then issuing a separate
# final yaw adjustment, we now use trajectory generation with yaw so that the drone is always
# oriented toward the desired waypoint yaw. The final single‑point yaw command has been removed.
#
# Assumptions:
# 1. Obstacles are cuboids, axis‑aligned, defined by center (x,y,z) and dimensions (d, w, h).
# 2. Viewpoint poses are specified in the scenario YAML with fields x, y, z and w.
# 3. The drone starting pose is provided and its z value is raised to TAKE_OFF_HEIGHT if needed.
# 4. TSP ordering is computed using Euclidean distances.
# 5. RRT* computes a continuous 3D path which is then smoothed.
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

SAFETY_MARGIN = 0.50        # Additional safety margin (m) around obstacles

COLLISION_CHECK_RESOLUTION = 0.5  # Step size (m) for collision checking
MAX_RECURSION_DEPTH = 100000      # Maximum subdivisions if a segment is in collision

PLANNING_TIME_LIMIT = 2       # Planning time limit per segment (s)
PLANNER_RANGE_SMALL = 1.0         # Fixed planner range used for all segments

# ------------------------
# Global Metrics and Data Logging Variables
# ------------------------
fallback_count = 0
segment_planning_times = []  # list of planning times per segment
segment_lengths = []         # list of segment lengths
global_rrt_tree_data = []    # list to store RRT tree data for each segment

# ------------------------
# Output Folder Setup
# ------------------------
import os
script_path = os.path.abspath(__file__)
script_name = os.path.splitext(os.path.basename(script_path))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(script_path), script_name)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
import threading
import json
import os  # already imported above
import pyautogui

from as2_python_api.drone_interface import DroneInterface

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

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
    num_points = int(8 + 8*avg_curvature)
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

def plan_rrtstar(start, goal, obstacles, bounds,
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

    # Use RRT* instead of RRT‑Connect
    planner = og.RRTstar(si)
    if hasattr(planner, 'setRange'):
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
    path, ptime = plan_rrtstar(start, goal, obstacles, bounds,
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

# --------------------------
# Plotting Functions
# --------------------------
def plot_paths(planned_paths, waypoints, obstacles=None, markers_list=None):
    """
    Plots:
      - The planned paths in both 2D and 3D.
      - Waypoints as red markers with numeric labels (no connecting lines).
      - Obstacles as nested shapes:
         * A red wireframe for the real obstacle.
         * A black wireframe (with reduced alpha) for the safety margin.
      - Uses a simplified approach for 3D obstacles (wireframes) to avoid Poly3DCollection issues.
    """

    # ----------------------------
    # HELPER FUNCTIONS
    # ----------------------------

    def draw_obstacle_2d(ax, ox, oy, dx, dy):
        # Real obstacle rectangle (red outline)
        real_rect = patches.Rectangle(
            (ox - dx/2, oy - dy/2),
            dx,
            dy,
            linewidth=1,
            edgecolor='red',
            facecolor='none',
            label='_nolegend_',
            zorder=10  # <-- higher zorder
        )
        ax.add_patch(real_rect)

        # Obstacle + margin rectangle (black outline, lower alpha)
        margin_rect = patches.Rectangle(
            (ox - dx/2 - SAFETY_MARGIN, oy - dy/2 - SAFETY_MARGIN),
            dx + 2*SAFETY_MARGIN,
            dy + 2*SAFETY_MARGIN,
            linewidth=0.5,
            edgecolor='black',
            facecolor='none',
            alpha=0.2,
            label='_nolegend_',
            zorder=9  # <-- slightly below real obstacle
        )
        ax.add_patch(margin_rect)

    def draw_obstacle_3d_wireframe(ax3d, ox, oy, oz, dx, dy, dz):
        """
        Draws obstacles as simple wireframes in 3D to avoid Poly3DCollection issues.
        Red lines = real obstacle, black lines (with alpha=0.2, linewidth=0.5) = margin.
        """
        # Real obstacle corners
        rx1 = ox - dx/2
        rx2 = ox + dx/2
        ry1 = oy - dy/2
        ry2 = oy + dy/2
        rz1 = oz - dz/2
        rz2 = oz + dz/2

        # Margin corners
        mx1 = rx1 - SAFETY_MARGIN
        mx2 = rx2 + SAFETY_MARGIN
        my1 = ry1 - SAFETY_MARGIN
        my2 = ry2 + SAFETY_MARGIN
        mz1 = rz1 - SAFETY_MARGIN
        mz2 = rz2 + SAFETY_MARGIN

        # -- Real obstacle wireframe (red) --
        # bottom face
        ax3d.plot([rx1, rx2], [ry1, ry1], [rz1, rz1], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx2], [ry1, ry2], [rz1, rz1], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx1], [ry2, ry2], [rz1, rz1], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx1, rx1], [ry2, ry1], [rz1, rz1], 'r-', linewidth=1, label='_nolegend_')

        # top face
        ax3d.plot([rx1, rx2], [ry1, ry1], [rz2, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx2], [ry1, ry2], [rz2, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx1], [ry2, ry2], [rz2, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx1, rx1], [ry2, ry1], [rz2, rz2], 'r-', linewidth=1, label='_nolegend_')

        # vertical edges
        ax3d.plot([rx1, rx1], [ry1, ry1], [rz1, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx2], [ry1, ry1], [rz1, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx2, rx2], [ry2, ry2], [rz1, rz2], 'r-', linewidth=1, label='_nolegend_')
        ax3d.plot([rx1, rx1], [ry2, ry2], [rz1, rz2], 'r-', linewidth=1, label='_nolegend_')

        # -- Margin wireframe (black, barely visible) --
        # bottom face
        ax3d.plot([mx1, mx2], [my1, my1], [mz1, mz1], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx2], [my1, my2], [mz1, mz1], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx1], [my2, my2], [mz1, mz1], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx1, mx1], [my2, my1], [mz1, mz1], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')

        # top face
        ax3d.plot([mx1, mx2], [my1, my1], [mz2, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx2], [my1, my2], [mz2, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx1], [my2, my2], [mz2, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx1, mx1], [my2, my1], [mz2, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')

        # vertical edges
        ax3d.plot([mx1, mx1], [my1, my1], [mz1, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx2], [my1, my1], [mz1, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx2, mx2], [my2, my2], [mz1, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')
        ax3d.plot([mx1, mx1], [my2, my2], [mz1, mz2], 'k-', linewidth=0.5, alpha=0.2, label='_nolegend_')

    # ----------------------------
    # 2D Trajectory Plot
    # ----------------------------
    fig2d, ax2d = plt.subplots()
    fig2d.canvas.manager.set_window_title("2D Trajectory")

    # Plot each planned path
    for idx, path in enumerate(planned_paths):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax2d.plot(xs, ys, label=f"Planned Path {idx+1}", linewidth=2)

    # Plot waypoints as red dots
    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    ax2d.scatter(wp_x, wp_y, c='r', marker='o', label="Waypoints")

    # Annotate each waypoint
    for i, (x, y) in enumerate(zip(wp_x, wp_y)):
        ax2d.text(x, y + 0.2, f"{i+1}", ha='center', va='bottom', fontsize=8, color='blue')

    # Draw obstacles in 2D
    if obstacles:
        for obs in obstacles:
            ox, oy = obs["x"], obs["y"]
            dx, dy = obs["d"], obs["w"]
            draw_obstacle_2d(ax2d, ox, oy, dx, dy)

    # Build legend, adding single "Real Obstacle" & "Margin" handles
    handles_2d, labels_2d = ax2d.get_legend_handles_labels()

    # Custom patches for 2D legend (only once)
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

    legend_2d = ax2d.legend(handles_2d, labels_2d, fontsize=6)
    legend_2d.get_frame().set_alpha(0.6)

    plt.tight_layout()
    file_path_2d = os.path.join(OUTPUT_DIR, "2D_Trajectory.png")
    plt.savefig(file_path_2d, dpi=300)
    plt.show()

    # ----------------------------
    # 3D Trajectory Plot
    # ----------------------------
    fig3d = plt.figure()
    fig3d.canvas.manager.set_window_title("3D Trajectory")
    ax3d = fig3d.add_subplot(111, projection='3d')

    # Plot each planned path
    for idx, path in enumerate(planned_paths):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax3d.plot(xs, ys, zs, label=f"Planned Path {idx+1}", linewidth=2)

    # Plot waypoints in 3D
    wp_z = [p[2] for p in waypoints]
    ax3d.scatter(wp_x, wp_y, wp_z, c='r', marker='o', label="Waypoints")

    # Annotate each waypoint
    for i, (x, y, z) in enumerate(zip(wp_x, wp_y, wp_z)):
        ax3d.text(x, y, z + 0.2, f"{i+1}", fontsize=8, color='blue')

    # Draw obstacles in 3D (using simple wireframe approach)
    if obstacles:
        for obs in obstacles:
            ox, oy, oz = obs["x"], obs["y"], obs["z"]
            dx, dy, dz = obs["d"], obs["w"], obs["h"]
            draw_obstacle_3d_wireframe(ax3d, ox, oy, oz, dx, dy, dz)

    # Legend for 3D
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

    legend_3d = ax3d.legend(handles_3d, labels_3d, fontsize=6)
    legend_3d.get_frame().set_alpha(0.6)

    ax3d.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    file_path_3d = os.path.join(OUTPUT_DIR, "3D_Trajectory.png")
    plt.savefig(file_path_3d, dpi=300)
    plt.show()


def plot_rrt_tree(rrt_tree_data, waypoints=None, obstacles=None, planned_paths=None):
    """
    Plot the 2D projection of the RRT* tree along with obstacles, waypoints, and drone paths.
    RRT tree nodes are drawn in the background without legend entries.
    """
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("RRT* Tree (2D Projection)")
    
    # Draw RRT tree nodes and edges (background)
    for tree in rrt_tree_data:
        nodes = tree.get("nodes", [])
        edges = tree.get("edges", [])
        if nodes:
            xs = [n[0] for n in nodes]
            ys = [n[1] for n in nodes]
            ax.scatter(xs, ys, c='blue', s=10, zorder=1)  # no label
        for edge in edges:
            (x1, y1, _), (x2, y2, _) = edge
            ax.plot([x1, x2], [y1, y2], 'c-', linewidth=0.5, zorder=1)
    
    # Plot obstacles
    if obstacles is not None:
        for obs in obstacles:
            ox, oy = obs["x"], obs["y"]
            dx, dy = obs["d"], obs["w"]
            lower_left = (ox - dx/2 - SAFETY_MARGIN, oy - dy/2 - SAFETY_MARGIN)
            rect = patches.Rectangle(lower_left, dx + 2*SAFETY_MARGIN, dy + 2*SAFETY_MARGIN,
                                     linewidth=1, edgecolor='k', facecolor='gray', alpha=0.3, zorder=2)
            ax.add_patch(rect)
    
    # Plot waypoints
    if waypoints is not None:
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]
        ax.plot(wp_x, wp_y, 'ro-', label="Waypoints", linewidth=2, zorder=3)
    
    # Plot drone path(s) on top
    if planned_paths is not None:
        for idx, path in enumerate(planned_paths):
            xs = [pt[0] for pt in path]
            ys = [pt[1] for pt in path]
            if idx == 0:
                ax.plot(xs, ys, color='black', linewidth=2, label="Drone Path", zorder=4)
            else:
                ax.plot(xs, ys, color='black', linewidth=2, zorder=4)
    
    ax.set_title("RRT* Tree (2D Projection)", fontsize=16)
    ax.set_xlabel("X (m)", fontsize=14)
    ax.set_ylabel("Y (m)", fontsize=14)
    ax.legend(fontsize=6)
    
    plt.tight_layout()
    file_path_rrt = os.path.join(OUTPUT_DIR, "RRTstar_Tree.png")
    plt.savefig(file_path_rrt, dpi=300)
    plt.show()

def plot_rrt_tree_3d(rrt_tree_data, waypoints=None, obstacles=None, planned_paths=None):
    """
    Plot the 3D projection of the RRT* tree along with obstacles, waypoints, and drone paths.
    RRT tree nodes are drawn in the background without legend entries.
    """
    fig = plt.figure()
    fig.canvas.manager.set_window_title("RRT* Tree (3D Projection)")
    ax3d = fig.add_subplot(111, projection='3d')
    
    # Draw RRT tree nodes and edges (background)
    for tree in rrt_tree_data:
        nodes = tree.get("nodes", [])
        edges = tree.get("edges", [])
        if nodes:
            xs = [n[0] for n in nodes]
            ys = [n[1] for n in nodes]
            zs = [n[2] for n in nodes]
            ax3d.scatter(xs, ys, zs, c='blue', s=10)  # no label
        for edge in edges:
            (x1, y1, z1), (x2, y2, z2) = edge
            ax3d.plot([x1, x2], [y1, y2], [z1, z2], 'c-', linewidth=0.5)
    
    # Plot obstacles
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
    
    # Plot waypoints
    if waypoints is not None:
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]
        wp_z = [p[2] for p in waypoints]
        ax3d.plot(wp_x, wp_y, wp_z, 'ro-', label="Waypoints", linewidth=2)
    
    # Plot drone path(s) on top
    if planned_paths is not None:
        for idx, path in enumerate(planned_paths):
            xs = [pt[0] for pt in path]
            ys = [pt[1] for pt in path]
            zs = [pt[2] for pt in path]
            if idx == 0:
                ax3d.plot(xs, ys, zs, color='black', linewidth=2, label="Drone Path")
            else:
                ax3d.plot(xs, ys, zs, color='black', linewidth=2)
    
    ax3d.set_title("RRT* Tree (3D Projection)", fontsize=16)
    ax3d.set_xlabel("X (m)", fontsize=14)
    ax3d.set_ylabel("Y (m)", fontsize=14)
    ax3d.set_zlabel("Z (m)", fontsize=14)
    ax3d.legend(fontsize=6)
    
    plt.tight_layout()
    file_path_rrt_3d = os.path.join(OUTPUT_DIR, "RRTstar_Tree_3D.png")
    plt.savefig(file_path_rrt_3d, dpi=300)
    plt.show()

def plot_costs(planning_times, path_lengths):
    segments = list(range(1, len(planning_times)+1))
    cumulative_cost = np.cumsum(np.array(planning_times) + np.array(path_lengths))
    
    # Create two subplots side by side (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title("Segment Costs")

    # First subplot: Planning Time per Segment
    ax1.bar(segments, planning_times, color='orange')
    ax1.set_xlabel("Segment", fontsize=14)
    ax1.set_ylabel("Planning Time (s)", fontsize=14)
    ax1.set_title("Planning Time per Segment", fontsize=16)
    ax1.legend(["Planning Time"], fontsize=10)

    # Second subplot: Cumulative Mission Cost
    ax2.plot(segments, cumulative_cost, marker='o', color='purple')
    ax2.set_xlabel("Segment", fontsize=14)
    ax2.set_ylabel("Cumulative Cost (s + m)", fontsize=14)
    ax2.set_title("Cumulative Mission Cost", fontsize=16)
    ax2.legend(["Cumulative Cost"], fontsize=10)

    plt.tight_layout()
    file_path_costs = os.path.join(OUTPUT_DIR, "Segment_Costs.png")
    plt.savefig(file_path_costs, dpi=300)
    plt.show()

def log_metrics_to_file(metrics, filename="metrics.json"):
    file_path = os.path.join(OUTPUT_DIR, filename)
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
        from as2_python_api.modules.trajectory_generation_module import TrajectoryGenerationModule
        self.trajectory_generation = TrajectoryGenerationModule(self)

    def image_callback(self, msg):
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        # Convert to grayscale and detect markers
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

        # Draw detected markers
        aruco.drawDetectedMarkers(cv_image, corners, ids)

        # 1) Create a resizable window
        cv2.namedWindow("ArUco Detection", cv2.WINDOW_NORMAL)

        # 2) Get screen dimensions
        screen_width, screen_height = pyautogui.size()

        # 3) Calculate half the screen width & height => 1/4 total screen area
        window_width = screen_width // 2
        window_height = screen_height // 2

        # 4) Resize and optionally move the window
        cv2.resizeWindow("ArUco Detection", window_width, window_height)
        # cv2.moveWindow("ArUco Detection", 0, 0)  # e.g., top-left corner

        # Show the image
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
      - Plans RRT* segments
      - Sends each segment using trajectory generation with yaw so that the drone is
        always oriented toward the desired waypoint yaw.
    """
    print("Run mission with RRT*, TSP (3‑opt), and trajectory generation with yaw.")
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
    fixed_planner_range = PLANNER_RANGE_SMALL

    for i in range(len(ordered_points) - 1):
        seg_start = ordered_points[i]
        seg_goal = ordered_points[i+1]

        # The final yaw we want at the end of the segment
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

        # B‑Spline smoothing
        smoothed_path = smooth_path_bspline(path)
        planned_paths.append(smoothed_path)
        seg_length = sum(compute_euclidean_distance(smoothed_path[j], smoothed_path[j+1])
                         for j in range(len(smoothed_path)-1))
        total_path_length += seg_length
        segment_planning_times.append(planning_time)
        segment_lengths.append(seg_length)

        # Use trajectory generation with yaw for the entire segment
        path_msg = create_path_msg(smoothed_path, frame_id="earth")
        print(f"Following segment with yaw (final yaw: {dest_yaw:.2f} rad) from {seg_start} to {seg_goal}.")
        success_traj = drone_interface.trajectory_generation.traj_generation_with_yaw(
            path=path_msg,
            speed=SPEED,
            angle=dest_yaw,
            frame_id="earth"
        )
        if not success_traj:
            print("Trajectory generation with yaw failed.")
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

    # Compute desired yaw for the return path (from last waypoint back to start)
    dest_yaw_return = math.atan2(return_start[1]-last_waypoint[1], return_start[0]-last_waypoint[0])
    path_msg = create_path_msg(smoothed_path, frame_id="earth")
    print(f"Following return path with yaw (final yaw: {dest_yaw_return:.2f} rad).")
    success_traj = drone_interface.trajectory_generation.traj_generation_with_yaw(
        path=path_msg,
        speed=SPEED,
        angle=dest_yaw_return,
        frame_id="earth"
    )
    if not success_traj:
        print("Trajectory generation for return path failed.")
        return False, None, None

    time.sleep(1.0)

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
        "obstacles": obstacles,
        "markers_list": markers_list,
        "fallback_count": drone_interface.fallback_count,
        "segment_planning_times": segment_planning_times,
        "segment_lengths": segment_lengths
    }

    return True, plot_data, metrics

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single drone mission with RRT*, TSP (3‑opt), obstacle avoidance, and trajectory generation enabled'
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

    # Update metrics with total mission duration and log again
    if plot_data is not None and metrics is not None:
        metrics["mission_duration"] = duration
        log_metrics_to_file(metrics)  # update the metrics file with duration

        # -- RELEVANT FIX: remove the extra "ground_truth" argument --
        # Now the function call matches the signature: plot_paths(planned_paths, waypoints, obstacles, markers_list).
        plot_paths(
            plot_data["planned_paths"],
            plot_data["ordered_points"],
            obstacles=plot_data["obstacles"],
            markers_list=plot_data["markers_list"]
        )
        plot_costs(plot_data["segment_planning_times"], plot_data["segment_lengths"])

        # Uncomment if you want to see the RRT tree plots:
        # plot_rrt_tree(global_rrt_tree_data,
        #               waypoints=plot_data["ordered_points"],
        #               obstacles=plot_data["obstacles"],
        #               planned_paths=plot_data["planned_paths"])
        # plot_rrt_tree_3d(global_rrt_tree_data,
        #                  waypoints=plot_data["ordered_points"],
        #                  obstacles=plot_data["obstacles"],
        #                  planned_paths=plot_data["planned_paths"])

    # After all plot windows are closed, invoke the external stop script for a clean exit.
    os.system("./stop.bash")
    print("Clean exit")
    exit(0)
