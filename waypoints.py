import numpy as np
from metadrive.component.road_network import Road

def get_next_relative_waypoints(agent, n=15):
    waypoints = get_waypoints(agent, delta_s=10, n=n)
    transformed_waypoints = []
    for wp in waypoints:
        x_rel, y_rel = global_to_agent(agent.position[0], agent.position[1], agent.heading_theta, wp[0], wp[1])
        transformed_waypoints.append((x_rel, y_rel, 0.5))
    
    return transformed_waypoints

def get_waypoints(agent, delta_s=10, n = 15): # CHANGE THIS TO START AT AGENT CURRENT POSITION
    roads = []
    navigation = agent.navigation
    checkpoints = navigation.checkpoints

    # Get the roads from agent current position to goal
    start_index = 0
    while checkpoints[start_index] != navigation.current_road.start_node:
        start_index += 1
        if start_index >= len(checkpoints):
            break

    for i in range(start_index, len(checkpoints)-1):
        road = Road(checkpoints[i], checkpoints[i+1])
        roads.append(road)

    waypoints = []
    dist_since_last = 0.0

    for idx, road in enumerate(roads):
        lanes = road.get_lanes(navigation.map.road_network)
        lane = lanes[0]

        # Position of the center of the road
        middle = (len(lanes) / 2 - 0.5) * lane.width

        # longitudinal start on this lane
        if idx == 0:
            s_start, _ = lane.local_coordinates(agent.position)
        else:
            s_start = 0.0

        lane_len = lane.length
        available = lane_len - s_start

        dist_to_next = (delta_s - dist_since_last) if dist_since_last > 0 else delta_s

        if available < dist_to_next:
            dist_since_last += available
            continue

        s = s_start + dist_to_next
        x, y = lane.position(s, middle)
        waypoints.append((x, y, 0.5))

        rem = lane_len - s
        n_more = int(np.floor(rem / delta_s))
        for k in range(1, n_more + 1):
            s_i = s + k * delta_s
            x_i, y_i = lane.position(s_i, middle)
            waypoints.append((x_i, y_i, 0.5))

        dist_since_last = rem - n_more * delta_s

        if len(waypoints) >= n:
            break

    return waypoints[:15]


import math

def global_to_agent(agent_x, agent_y, agent_theta, point_x, point_y):
    """
    Transforms a point from global coordinates into the agent's local frame.

    Parameters:
    - agent_x, agent_y: float. Agent's position in global frame.
    - agent_theta: float. Agent's yaw in radians (0 along global +X, positive CCW).
    - point_x, point_y: float. Point's position in global frame.

    Returns:
    - (x_rel, y_rel): tuple of floats. Point's coordinates in the agent's frame.
      x_rel is the forward distance, y_rel is the lateral (left-right) distance
      relative to where the agent is facing.
    """
    # Translate so agent is at origin
    dx = point_x - agent_x
    dy = point_y - agent_y

    # Rotate by -agent_theta
    x_rel =  dx * math.cos(agent_theta) + dy * math.sin(agent_theta)
    y_rel = -dx * math.sin(agent_theta) + dy * math.cos(agent_theta)

    return x_rel, y_rel
