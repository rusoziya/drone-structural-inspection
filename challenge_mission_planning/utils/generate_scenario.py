import random
import yaml
import argparse

def is_point_inside_obstacle(point, obstacle, buffer=1.0):
    """
    Check if a point is inside or within a buffer zone around an obstacle.
    
    Args:
        point (tuple): (x, y, z) coordinates of the point.
        obstacle (dict): Obstacle dictionary with keys 'x', 'y', 'z', 'h', 'w', 'd'.
        buffer (float): Additional buffer distance around the obstacle (default 1.0).
    
    Returns:
        bool: True if the point is inside the obstacle or its buffer zone, False otherwise.
    """
    x, y, z = point
    ox, oy, oz = obstacle['x'], obstacle['y'], obstacle['z']
    h, w, d = obstacle['h'], obstacle['w'], obstacle['d']
    
    return (
        ox - w / 2 - buffer <= x <= ox + w / 2 + buffer and
        oy - d / 2 - buffer <= y <= oy + d / 2 + buffer and
        oz - buffer <= z <= oz + h + buffer
    )


def generate_random_pose(bounds, obstacles, min_distance=1.0):
    """Generate a random pose that is not inside any obstacle."""
    while True:
        x = random.uniform(bounds['x_min'], bounds['x_max'])
        y = random.uniform(bounds['y_min'], bounds['y_max'])
        z = random.uniform(bounds['z_min'], bounds['z_max'])
        
        # Check against all obstacles
        if all(not is_point_inside_obstacle((x, y, z), obs, min_distance) for obs in obstacles):
            return {"x": x, "y": y, "z": z, "r": 0.0, "p": 0.0, "w": random.uniform(0, 3.14)}

def generate_scenario(num_viewpoints, num_obstacles, bounds):
    """Generate a scenario configuration."""
    scenario = {
        "name": f"scenario{random.randint(1, 1000)}",
        "drone_start_pose": {"x": 0.0, "y": 0.0, "z": 0.0},
        "viewpoint_poses": {},
        "obstacles": {}
    }
    
    # Generate random obstacles
    for i in range(1, num_obstacles + 1):
        obstacle = {
            "x": random.uniform(bounds['x_min'], bounds['x_max']),
            "y": random.uniform(bounds['y_min'], bounds['y_max']),
            "z": random.uniform(0.0, bounds['z_max']),
            "h": random.uniform(0.5, 5.0),
            "w": random.uniform(0.5, 2.0),
            "d": random.uniform(0.5, 2.0)
        }
        scenario["obstacles"][i] = obstacle
    
    # Generate random viewpoints
    for i in range(1, num_viewpoints + 1):
        pose = generate_random_pose(bounds, scenario["obstacles"].values())
        scenario["viewpoint_poses"][i] = pose
    
    return scenario

def save_scenario_to_yaml(scenario, file_path):
    """Save the scenario to a YAML file."""
    with open(file_path, 'w') as file:
        yaml.dump(scenario, file, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="Generate a random Gazebo scenario with obstacles and viewpoints.")
    parser.add_argument("--num_viewpoints", type=int, default=5, help="Number of random viewpoints to generate.")
    parser.add_argument("--num_obstacles", type=int, default=30, help="Number of random obstacles to generate.")
    parser.add_argument("--x_min", type=float, default=-10.0, help="Minimum x-coordinate for generation bounds.")
    parser.add_argument("--x_max", type=float, default=10.0, help="Maximum x-coordinate for generation bounds.")
    parser.add_argument("--y_min", type=float, default=-10.0, help="Minimum y-coordinate for generation bounds.")
    parser.add_argument("--y_max", type=float, default=10.0, help="Maximum y-coordinate for generation bounds.")
    parser.add_argument("--z_min", type=float, default=1.0, help="Minimum z-coordinate for generation bounds.")
    parser.add_argument("--z_max", type=float, default=5.0, help="Maximum z-coordinate for generation bounds.")
    parser.add_argument("--output_file", type=str, default="scenario.yaml", help="Output YAML file for the scenario.")
    
    args = parser.parse_args()

    # Define bounds for the random generation
    bounds = {
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "z_min": args.z_min,
        "z_max": args.z_max
    }
    
    # Generate scenario
    scenario = generate_scenario(args.num_viewpoints, args.num_obstacles, bounds)
    
    # Save to a YAML file
    save_scenario_to_yaml(scenario, args.output_file)
    print(f"Scenario saved to {args.output_file}")

if __name__ == "__main__":
    main()
