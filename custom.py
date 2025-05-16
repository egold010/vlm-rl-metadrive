"""
Sensors
"""
from metadrive.component.sensors.base_sensor import BaseSensor
from metadrive.component.sensors.semantic_camera import SemanticCamera
from waypoints import get_next_relative_waypoints

class SegCamera(SemanticCamera):
    """
    Semantic segmentation camera sensor.
    """
    HPR = (0, -90, 0)
    POS = (0, 0, 32)

    def __init__(self, width, height, engine, cuda=False):
        self.WIDTH, self.HEIGHT = width, height
        super(SegCamera, self).__init__(width, height, engine)

    def perceive(self, to_float=True, new_parent_node = None, position=None, hpr=None):
        return super(SegCamera, self).perceive(to_float, new_parent_node, position=self.POS, hpr=self.HPR)
    
class EgoState(BaseSensor):
    """
    EgoState sensor.
    """
    def __init__(self, agent_name, engine, cuda=False):
        self.engine = engine
        self.agent_name = agent_name

    def perceive(self):
        agent = self.engine.agents[self.agent_name]
        steer = agent.steering
        throttle = agent.throttle_brake
        speed = agent.speed_km_h
        return steer, throttle, speed
    
class WaypointSensor(BaseSensor):
    """
    Waypoint sensor.
    """
    def __init__(self, agent_name, n, engine, cuda=False):
        self.n = n
        self.engine = engine
        self.agent_name = agent_name

    def perceive(self):
        agent = self.engine.agents[self.agent_name]
        return get_next_relative_waypoints(agent, n=15)

"""
Observations
"""
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.observation_base import BaseObservation
from gym.spaces import Box
import gym
import numpy as np
from agent_utils import *

class AgentObservation(BaseObservation):
    def __init__(self, config):
        super(AgentObservation, self).__init__(config)
        assert config["norm_pixel"] is False
        assert config["stack_size"] == 1
        self.seg_obs = ImageObservation(config, "seg_camera", config["norm_pixel"])

    @property
    def observation_space(self):
        seg_stack = self.seg_obs.observation_space
        os = {}

        low, high = [], []
        low.append(-1), high.append(1) # steer
        low.append(-1), high.append(1) # throttle
        low.append(-80), high.append(80) # speed
        os['vehicle_measures'] = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        os['waypoints'] = gym.spaces.Box(low=-50, high=50, shape=(15, 2), dtype=np.float32)
        os["seg_camera"] = gym.spaces.Box(0, 1, shape=(get_num_classes(), *seg_stack.shape[-2:]), dtype=np.uint8),

        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        os = {}

        seg_cam = self.engine.get_sensor("seg_camera").cam
        agent = seg_cam.getParent()
        original_position = seg_cam.getPos()
        heading, pitch, roll = seg_cam.getHpr()
        seg_img = self.seg_obs.observe(agent, position=original_position, hpr=[heading, pitch, roll])
        assert seg_img.ndim == 4
        assert seg_img.shape[-1] == 1
        assert seg_img.dtype == np.uint8
        seg_img = seg_img[..., 0]
        seg_img = seg_img[..., ::-1]  # BGR -> RGB
        seg_img = one_hot_encode_semantic_map(seg_img).astype(np.float32)
        os["seg_camera"] = seg_img

        ego = self.engine.get_sensor("ego")
        steer, throttle, speed = ego.perceive()

        vehicle_measures = []
        vehicle_measures.append(steer)
        vehicle_measures.append(throttle)
        vehicle_measures.append(speed)
        os["vehicle_measures"] = vehicle_measures

        waypoint_sensor = self.engine.get_sensor("waypoint")
        waypoints = waypoint_sensor.perceive()
        os["waypoints"] = np.array(waypoints, dtype=np.float32)

        return os