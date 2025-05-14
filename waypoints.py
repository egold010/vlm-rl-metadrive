import numpy as np
from metadrive.component.road_network import Road

def get_waypoints(agent, delta_s=10):
    roads = []
    navigation = agent.navigation
    checkpoints = navigation.checkpoints

    # Get the roads from start to goal
    for i in range(len(checkpoints)-1):
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

    return waypoints

def get_next_relative_waypoints(agent, n=15):
    pass