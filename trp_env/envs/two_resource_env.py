import logging
import math
import os
import tempfile
import xml.etree.ElementTree as ET
import inspect
from collections import deque

import numpy as np

from enum import Enum, auto

import mujoco

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE, MujocoEnv
from gymnasium import utils

BIG = 1e6
DEFAULT_CAMERA_CONFIG = {}


class FoodClass(Enum):
    BLUE = auto()
    RED = auto()


class AgentType(Enum):
    WALKER = auto()
    SWIMMER = auto()
    BALL = auto()


def qtoeuler(q):
    """ quaternion to Euler angle

    :param q: quaternion
    :return:
    """
    phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    # theta = -np.pi/2 + 2*math.atan2(math.sqrt(1 + 2*(q[0]*q[2] - q[1]*q[3])), math.sqrt(1 - 2*(q[0]*q[2]-q[1]*q[3])))
    psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([phi, theta, psi])


def eulertoq(euler):
    phi, theta, psi = euler
    qx = np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qy = np.sin(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qz = np.cos(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2)
    qw = np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2)
    return np.array([qx, qy, qz, qw])


class TwoResourceEnv(MujocoEnv, utils.EzPickle):
    MODEL_CLASS = None
    ORI_IND = None
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps"  : 20,
    }
    
    def __init__(self,
                 ego_obs=True,
                 n_blue=14,  # same with Konidaris & Barto
                 n_red=8,  # same with Konidaris & Barto
                 activity_range=42,  # approx. same with Konidaris & Barto
                 robot_object_spacing=2.,
                 catch_range=1.,
                 n_bins=10,
                 sensor_range=12.5,  # approx. same with Konidaris & Barto
                 sensor_span=2 * math.pi,
                 coef_inner_rew=0.,
                 coef_main_rew=100.,
                 coef_ctrl_cost=0.001,
                 coef_head_angle=0.005,
                 dying_cost=-10,
                 max_episode_steps=np.inf,
                 show_sensor_range=False,
                 reward_setting="homeostatic_shaped",
                 reward_bias=None,
                 internal_reset="random",
                 internal_random_range=(-1. / 6, 1. / 6),
                 blue_nutrient=(0.1, 0),
                 red_nutrient=(0, 0.1),
                 show_move_line=False,
                 on_texture=False,
                 recognition_obs=False,
                 vision=False,
                 width=64,
                 height=64,
                 random_position=False,
                 *args, **kwargs):
        """

        :param int n_blue:  Number of greens in each episode
        :param int n_red: Number of reds in each episode
        :param float activity_range: he span for generating objects (x, y in [-range, range])
        :param float robot_object_spacing: Number of objects in each episode
        :param float catch_range: Minimum distance range to catch an object
        :param int n_bins: Number of objects in each episode
        :param float sensor_range: Maximum sensor range (how far it can go)
        :param float sensor_span: Maximum sensor span (how wide it can span), in radians
        :param coef_inner_rew:
        :param coef_main_rew:
        :param coef_cost:
        :param coef_head_angle:
        :param dying_cost:
        :param max_episode_steps:
        :param show_sensor_range: Show range sensor. Default OFF
        :param reward_setting: Setting of the reward definitions. "homeostatic", "homeostatic_shaped", "one", "homeostatic_biased" or "greedy". "homeostatic_shaped" is default. "greedy is not a homeostatic setting"
        :param reward_bias: biasing reward with constant. new_reward = reward + reward_bias
        :param internal_reset: resetting rule of the internal nutrient state. "setpoint" or "random".
        :param internal_random_range: if reset condition is "random", use this region for initialize all internal variables
        :param blue_nutrient: setting of the nutrient update if the agent took a blue food
        :param red_nutrient: setting of the nutrient update if the agent took a red food
        :param show_move_line: render the movement of the agent in the environment
        :param regognition_obs: True if exteroception representation is (distance array + object recignition array)
        :param vision: enable vision outputs
        :param width: vision width
        :param height: vision height
        :param random_position: set random position at environment reset
        :param args:
        :param kwargs:
        """
        self.n_blue = n_blue
        self.n_red = n_red
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.coef_inner_rew = coef_inner_rew
        self.coef_main_rew = coef_main_rew
        self.coef_ctrl_cost = coef_ctrl_cost
        self.coef_head_angle = coef_head_angle
        self.dying_cost = dying_cost
        self._max_episode_steps = max_episode_steps
        self.show_sensor_range = show_sensor_range
        self.reward_setting = reward_setting
        self.reward_bias = reward_bias if reward_bias else 0.
        self.internal_reset = internal_reset
        self.internal_random_range = internal_random_range
        self.show_move_line = show_move_line
        self.recognition_obs = recognition_obs
        self.random_position_at_reset = random_position
        
        self.objects = []
        
        # Internal state
        self._target_internal_state = np.array([0.0, 0.0])  # [Blue, Red]
        
        if self.internal_reset in {"setpoint", "random"}:
            self.internal_state = {
                FoodClass.BLUE: 0.0,
                FoodClass.RED : 0.0,
            }
        else:
            raise ValueError('internal_reset should be "setpoint" or "random"')
        
        self.prev_interoception = self.get_interoception()
        self.blue_nutrient = blue_nutrient
        self.red_nutrient = red_nutrient
        self.default_metabolic_update = 0.00015
        self.survival_area = 1.0
        
        utils.EzPickle.__init__(**locals())
        
        # for openai baseline
        self.reward_range = (-float('inf'), float('inf'))
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        import pathlib
        p = pathlib.Path(inspect.getfile(self.__class__))
        MODEL_DIR = os.path.join(p.parent, "models", model_cls.FILE)
        
        tree = ET.parse(MODEL_DIR)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3"
        )
        
        if on_texture:
            asset = tree.find(".//asset")
            ET.SubElement(
                asset, "texture", dict(
                    name="grass_texture",
                    type="2d",
                    file=os.path.dirname(__file__) + "/models/texture/grass.png",
                    width="100",
                    height="100",
                )
            )
            ET.SubElement(
                asset, "material", dict(
                    name="grass",
                    texture="grass_texture",
                    texrepeat="20 20"
                )
            )
            
            ET.SubElement(
                asset, "texture", dict(
                    name="wall_texture",
                    type="cube",
                    file=os.path.dirname(__file__) + "/models/texture/jari.png",
                    width="100",
                    height="100",
                )
            )
            ET.SubElement(
                asset, "material", dict(
                    name="wall",
                    texture="wall_texture",
                    texrepeat="2 1"
                )
            )
            
            worldbody.find("geom").set("material", "grass")
            
            attrs = dict(
                type="box", conaffinity="1", material="wall", condim="3"
            )
        
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall1",
                pos="0 -%d 1" % walldist,
                size="%d.5 0.5 2" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall2",
                pos="0 %d 1" % walldist,
                size="%d.5 0.5 2" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall3",
                pos="-%d 0 1" % walldist,
                size="0.5 %d.5 2" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall4",
                pos="%d 0 1" % walldist,
                size="0.5 %d.5 2" % walldist))
        
        # NOTE: NamedTemporaryFile cannot be reopened by name while still open on
        # Windows, so use mkstemp (close the handle) and clean up manually.
        fd, file_path = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        try:
            tree.write(file_path)

            # build mujoco
            self.wrapped_env = model_cls(
                file_path,
                vision=vision,
                width=width,
                height=height,
                **kwargs
            )
        finally:
            try:
                os.remove(file_path)
            except OSError:
                pass
        
        # optimization, caching obs spaces
        ub = BIG * np.ones(self.get_current_obs().shape, dtype=np.float32)
        self.obs_space = spaces.Box(ub * -1, ub)
        ub = BIG * np.ones(self.get_current_robot_obs().shape, dtype=np.float32)
        self.robot_obs_space = spaces.Box(ub * -1, ub)
        
        self._step = 0
        
        self.num_blue_eaten = 0
        self.num_red_eaten = 0
        
        # visualization
        self.agent_positions = deque(maxlen=300)
    
    @property
    def dim_intero(self):
        return np.prod(self._target_internal_state.shape)
    
    def reset_internal_state(self):
        if self.internal_reset == "setpoint":
            self.internal_state = {
                FoodClass.BLUE: 0.0,
                FoodClass.RED : 0.0,
            }
        elif self.internal_reset == "random":
            self.internal_state = {
                FoodClass.BLUE: self.wrapped_env.np_random.uniform(self.internal_random_range[0],
                                                                   self.internal_random_range[1]),
                FoodClass.RED : self.wrapped_env.np_random.uniform(self.internal_random_range[0],
                                                                   self.internal_random_range[1]),
            }
        else:
            raise ValueError('internal_reset should be "setpoint" or "random"')
    
    def reset(self, seed=None, return_info=True, options=None, n_blue=None, n_red=None):
        self._step = 0
        self.num_blue_eaten = 0
        self.num_red_eaten = 0
        
        if self.random_position_at_reset:
            L = self.activity_range - self.activity_range * 0.1
            self.wrapped_env.init_qpos[:2] = self.np_random.uniform(-L, L, size=2)
            
            random_angle = self.np_random.uniform(0, 2 * np.pi)
            q = eulertoq(np.array([0, 0, random_angle]))
            if self.wrapped_env.TYPE in {AgentType.WALKER, AgentType.BALL}:
                self.wrapped_env.init_qpos[3:3 + 4] = q
            elif self.wrapped_env.TYPE is AgentType.SWIMMER:
                self.wrapped_env.init_qpos[2:2 + 2] = q[:2]
            else:
                raise ValueError()
        
        self.wrapped_env.reset(seed=seed)
        
        self.reset_internal_state()
        self.prev_interoception = self.get_interoception()
        self.agent_positions.clear()
        
        if n_blue is not None:
            self.n_blue = n_blue
        
        if n_red is not None:
            self.n_red = n_red
        
        assert self.n_red + self.n_blue < (self.activity_range + 1) ** 2
        
        self.objects = []
        existing = set()
        while len(self.objects) < self.n_blue:
            x = self.wrapped_env.np_random.integers(-self.activity_range / 2,
                                                    self.activity_range / 2 + 1) * 2
            y = self.wrapped_env.np_random.integers(-self.activity_range / 2,
                                                    self.activity_range / 2 + 1) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = FoodClass.BLUE
            self.objects.append((x, y, typ))
            existing.add((x, y))
        while len(self.objects) < self.n_blue + self.n_red:
            x = self.wrapped_env.np_random.integers(-self.activity_range / 2,
                                                    self.activity_range / 2 + 1) * 2
            y = self.wrapped_env.np_random.integers(-self.activity_range / 2,
                                                    self.activity_range / 2 + 1) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = FoodClass.RED
            self.objects.append((x, y, typ))
            existing.add((x, y))
        
        info = {"interoception": self.get_interoception()}
        
        return (self.get_current_obs(), info) if return_info else self.get_current_obs()
    
    def generate_new_object(self, type_gen):
        existing = set()
        for object in self.objects:
            existing.add((object[0], object[1]))
        
        while True:
            x = self.wrapped_env.np_random.integers(-self.activity_range / 2,
                                                    self.activity_range / 2) * 2
            y = self.wrapped_env.np_random.integers(-self.activity_range / 2,
                                                    self.activity_range / 2) * 2
            if (x, y) in existing:
                continue
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            return (x, y, type_gen)
    
    def step_internal_state_default(self):
        self.internal_state[FoodClass.RED] -= self.default_metabolic_update
        self.internal_state[FoodClass.BLUE] -= self.default_metabolic_update
    
    def update_by_food(self, is_red, is_blue):
        """
        A metabolic update of the agent internal state
        :param is_red:
        :param is_blue:
        :return:
        """
        
        assert is_red or is_blue, "one of food should be True"
        
        if is_red:
            self.internal_state[FoodClass.BLUE] += self.red_nutrient[0]
            self.internal_state[FoodClass.RED] += self.red_nutrient[1]
        if is_blue:
            self.internal_state[FoodClass.BLUE] += self.blue_nutrient[0]
            self.internal_state[FoodClass.RED] += self.blue_nutrient[1]
    
    def step(self, action: np.ndarray):
        self.prev_interoception = self.get_interoception()
        _, inner_rew, terminated, truncated, info = self.wrapped_env.step(action)
        truncated = False
        
        info['inner_rew'] = inner_rew
        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]
        self.agent_positions.append(np.array(com, np.float32))
        info['com'] = com
        
        if terminated:
            info['outer_rew'] = 0
            info["interoception"] = self.get_interoception()
            return self.get_current_obs(), self.dying_cost, terminated, truncated, info  # give a -10 rew if the robot dies
        
        #  Default Metabolic update
        self.step_internal_state_default()
        
        # Food-Eating
        new_objs = []
        self.num_blue_eaten = 0
        self.num_red_eaten = 0
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ is FoodClass.BLUE:
                    self.update_by_food(is_red=False, is_blue=True)
                    self.num_blue_eaten += 1
                elif typ is FoodClass.RED:
                    self.update_by_food(is_red=True, is_blue=False)
                    self.num_red_eaten += 1
                new_objs.append(self.generate_new_object(type_gen=typ))
            else:
                new_objs.append(obj)
        
        self.objects = new_objs
        
        info["interoception"] = self.get_interoception()
        
        terminated = np.max(np.abs(self.get_interoception())) > self.survival_area
        
        info["food_eaten"] = (self.num_blue_eaten, self.num_red_eaten)
        
        self._step += 1
        terminated = terminated or self._step >= self._max_episode_steps
        
        reward, info_rew = self.get_reward(reward_setting=self.reward_setting,
                                           action=action,
                                           done=terminated,
                                           num_blue_eaten=self.num_blue_eaten,
                                           num_red_eaten=self.num_red_eaten)
        
        info.update(info_rew)
        
        return self.get_current_obs(), reward, terminated, truncated, info
    
    def get_reward(self, reward_setting, action, done, num_blue_eaten=None, num_red_eaten=None):
        # Motor Cost
        lb, ub = self.wrapped_env.action_space.low, self.wrapped_env.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = -.5 * np.square(action / scaling).sum()
        
        # Local Posture Cost
        if self.wrapped_env.TYPE is AgentType.WALKER:
            euler = qtoeuler(self.wrapped_env.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4])
            euler_stand = qtoeuler([1.0, 0.0, 0.0, 0.0])  # quaternion of standing state
            head_angle_cost = -np.square(euler[:2] - euler_stand[:2]).sum()  # Drop yaw
        else:
            head_angle_cost = 0.
        
        total_cost = self.coef_ctrl_cost * ctrl_cost + self.coef_head_angle * head_angle_cost
        
        # Main Reward
        info = {"reward_module": None}
        
        def drive(intero, target):
            drive_module = -1 * (intero - target) ** 2
            d_ = drive_module.sum()
            return d_, drive_module
        
        if reward_setting == "homeostatic":
            d, dm = drive(self.prev_interoception, self._target_internal_state)
            main_reward = d
            info["reward_module"] = np.concatenate([self.coef_main_rew * dm, [total_cost]])
        
        elif reward_setting == "homeostatic_shaped":
            d, dm = drive(self.get_interoception(), self._target_internal_state)
            d_prev, dm_prev = drive(self.prev_interoception, self._target_internal_state)
            main_reward = d - d_prev
            info["reward_module"] = np.concatenate([self.coef_main_rew * (dm - dm_prev), [total_cost]])
        
        elif reward_setting == "one":
            # From continual-Cartpole setting from the lecture of Doina Precup (EEML 2021).
            if done:
                main_reward = -1.
            else:
                main_reward = 0.
        
        elif reward_setting == "homeostatic_biased":
            d, dm = drive(self.prev_interoception, self._target_internal_state)
            main_reward = d + self.reward_bias
            info["reward_module"] = np.concatenate([self.coef_main_rew * dm, [total_cost]])
        
        elif reward_setting == "greedy":
            if num_red_eaten is None or num_blue_eaten is None:
                raise ValueError
            main_reward = num_blue_eaten + num_red_eaten
        else:
            raise ValueError
        
        reward = self.coef_main_rew * main_reward + total_cost
        
        return reward, info
    
    def get_readings(self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        blue_readings = np.zeros(self.n_bins)
        red_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins
        
        ori = self.get_ori()
        
        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb;
                ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ is FoodClass.BLUE:
                blue_readings[bin_number] = intensity
            elif typ is FoodClass.RED:
                red_readings[bin_number] = intensity
        return blue_readings, red_readings
    
    def get_recog_based_readings(self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        depth_readings = np.zeros(self.n_bins)
        blue_readings = np.zeros(self.n_bins)
        red_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins
        
        ori = self.get_ori()
        
        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb;
                ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            
            # Set depth readings
            depth_readings[bin_number] = intensity
            # Set object recognition results
            if typ is FoodClass.BLUE:
                blue_readings[bin_number] = 1.0
            elif typ is FoodClass.RED:
                red_readings[bin_number] = 1.0
        
        return depth_readings, blue_readings, red_readings
    
    def get_interoception(self):
        return np.array(list(self.internal_state.values()), dtype=np.float32)
    
    def get_current_robot_obs(self):
        return self.wrapped_env.get_current_obs()
    
    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env.get_current_obs()
        
        if self.recognition_obs:
            depth_readings, blue_readings, red_readings = self.get_recog_based_readings()
            exteroception = np.concatenate([depth_readings, blue_readings, red_readings])
        else:
            blue_readings, red_readings = self.get_readings()
            exteroception = np.concatenate([blue_readings, red_readings])
        
        interoception = self.get_interoception()
        return np.concatenate([self_obs, exteroception, interoception], dtype=np.float32)
    
    @property
    def multi_modal_dims(self):
        self_obs_dim = len(self.wrapped_env.get_current_obs())
        
        green_readings, red_readings = self.get_readings()
        readings_dim = len(green_readings) + len(red_readings)
        
        interoception_dim = len(self.get_interoception())
        
        # (proprioception, exteroception, interoception)
        return tuple([self_obs_dim, readings_dim, interoception_dim])
    
    @property
    def observation_space(self):
        return self.obs_space
    
    # space of only the robot observations (they go first in the get current obs)
    @property
    def robot_observation_space(self):
        return self.robot_obs_space
    
    @property
    def action_space(self):
        return self.wrapped_env.action_space
    
    @property
    def dt(self):
        return self.wrapped_env.dt
    
    def close(self):
        if self.wrapped_env.mujoco_renderer is not None:
            self.wrapped_env.mujoco_renderer.close()
    
    def get_ori(self):
        """
        First it tries to use a get_ori from the wrapped env. If not successfull, falls
        back to the default based on the ORI_IND specified in Maze (not accurate for quaternions)
        """
        obj = self.wrapped_env
        while not hasattr(obj, 'get_ori') and hasattr(obj, 'wrapped_env'):
            obj = obj.wrapped_env
        try:
            return obj.get_ori()
        except (NotImplementedError, AttributeError) as e:
            pass
        return self.wrapped_env.data.qpos[self.__class__.ORI_IND]
    
    def get_image(self,
                  mode='human',
                  camera_id=None,
                  camera_name=None
                  ):
        
        if mode == "rgbd_array":
            viewers = [self.wrapped_env.mujoco_renderer._get_viewer(render_mode="rgb_array")]
            viewers.append(self.wrapped_env.mujoco_renderer._get_viewer(render_mode="depth_array"))
        else:
            viewers = [self.wrapped_env.mujoco_renderer._get_viewer(render_mode=mode)]
        
        # Show Sensor Range
        if self.show_sensor_range:
            
            robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
            ori = self.get_ori()
            
            sensor_range = np.linspace(start=-self.sensor_span * 0.5,
                                       stop=self.sensor_span * 0.5,
                                       num=self.n_bins,
                                       endpoint=True)
            for direction in sensor_range:
                from scipy.spatial.transform import Rotation
                def euler2mat(euler):
                    r = Rotation.from_euler('xyz', euler, degrees=False)
                    return r.as_matrix()
                
                ox = robot_x + self.sensor_range * math.cos(direction + ori)
                oy = robot_y + self.sensor_range * math.sin(direction + ori)
                viewers[0].add_marker(
                    pos=np.array([ox, oy, 0.5]),
                    label=" ",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=(0.1, 0.1, 0.1),
                    rgba=(0, 1, 0, 0.8)
                )
                viewers[0].add_marker(
                    pos=np.array([robot_x, robot_y, 0.5]),
                    label=" ",
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    mat=euler2mat([0, np.pi/2, np.arctan2(oy, ox)]),
                    size=(0.1, 0.1, self.sensor_range),
                    rgba=(0, 1, 0, 0.8)
                )
        
        # show movement of the agent
        if self.show_move_line:
            for pos in self.agent_positions:
                for v in viewers:
                    v.add_marker(pos=pos,
                                 label=" ",
                                 type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                 size=(0.05, 0.05, 0.05),
                                 rgba=(1, 0, 0, 0.3),
                                 emission=1)
        
        # Show food
        for obj in self.objects:
            ox, oy, typ = obj
            rgba = None
            if typ is FoodClass.RED:
                rgba = (1, 0, 0, 1)
            elif typ is FoodClass.BLUE:
                rgba = (0, 0, 1, 1)
            
            if rgba:
                for v in viewers:
                    v.add_marker(pos=np.array([ox, oy, 0.5]),
                                 label=" ",
                                 type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                 size=(0.5, 0.5, 0.5),
                                 rgba=rgba)
        
        if mode == "rgbd_array":
            im = self.wrapped_env.mujoco_renderer.render(
                "rgb_array",
                camera_id,
                camera_name, )
            im_d = self.wrapped_env.mujoco_renderer.render(
                "depth_array",
                camera_id,
                camera_name, )[:, :, np.newaxis]
            
            im = np.dstack((im, im_d))
        else:
            im = self.wrapped_env.mujoco_renderer.render(
                mode,
                camera_id,
                camera_name,
            )
        
        # delete unnecessary markers: https://github.com/openai/mujoco-py/issues/423#issuecomment-513846867
        for v in viewers:
            del v._markers[:]
        
        return im
    
    def render(
            self,
            mode='human',
            camera_id=None,
            camera_name=None
    ):
        return self.get_image(mode=mode, camera_id=camera_id, camera_name=camera_name)
