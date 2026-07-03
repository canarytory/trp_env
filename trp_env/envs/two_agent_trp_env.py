"""Two-agent Two-Resource-Problem environment (state-based, homeostatic RL).

Two low-gear Ants live inside a *single* MuJoCo simulation so that they
physically collide with each other (one can knock the other over, block a
path, etc.). Food has no physics -- exactly as in the single-agent
``TwoResourceEnv`` it is tracked in Python and "eaten" by proximity -- but the
food field is *shared*, so the two agents compete for the same resources.

Design notes
------------
* Both Ant bodies are placed in one ``mujoco`` model. Body/joint/geom names are
  prefixed with ``ant0_`` / ``ant1_`` to avoid clashes, and the two Ants are
  given contype/conaffinity bitmasks so that they collide with each other and
  with the walls/floor, but NOT with themselves (this keeps each Ant's own
  dynamics identical to the single-agent env):

      ant0 geoms: contype=1 conaffinity=2
      ant1 geoms: contype=2 conaffinity=1
      floor/walls: contype=1 conaffinity=1   (unchanged)

* Each agent has its own interoception (internal nutrient state), its own food
  eating, its own reward and its own observation. The observation adds a third
  sensor channel for the *other* Ant on top of the blue/red food channels:

      obs = [ proprioception(27),
              blue_readings(n_bins), red_readings(n_bins),
              other_ant_readings(n_bins),
              interoception(2) ]

* The MuJoCo sim is shared, so an episode ends (and BOTH agents reset) as soon
  as either agent dies or the step limit is hit. Per-agent rewards/returns are
  still tracked separately, which is what the independent learners consume.

The class deliberately does NOT go through ``gym.make`` (its step signature is
multi-agent); construct it directly via the builder in
``util/env_two_state_modern.py``.
"""
import copy
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import mujoco
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from trp_env.envs.two_resource_env import (
    FoodClass, AgentType, qtoeuler,
)
from trp_env.envs.ant_trp_env import q_inv, q_mult

BIG = 1e6
DEFAULT_CAMERA_CONFIG = {"distance": 25.0}

# per-Ant qpos / qvel layout (free joint 7 + 8 hinges = 15 qpos; 6 + 8 = 14 qvel)
QPOS_PER_ANT = 15
QVEL_PER_ANT = 14
N_ACT_PER_ANT = 8


def _prefix_tree(body, prefix):
    """Prefix every ``name`` attribute in an element subtree in place."""
    for el in body.iter():
        if "name" in el.attrib:
            el.attrib["name"] = prefix + el.attrib["name"]


def _set_collision(body, contype, conaffinity):
    """Force contype/conaffinity on every geom in a subtree."""
    for geom in body.iter("geom"):
        geom.attrib["contype"] = str(contype)
        geom.attrib["conaffinity"] = str(conaffinity)


def build_two_ant_model_xml(base_xml_path, activity_range, spawn_positions,
                            collision_masks):
    """Duplicate the single-Ant model into a two-Ant model + surrounding walls.

    :param base_xml_path: path to the single-ant xml (low_gear_ratio_ant.xml)
    :param activity_range: half-size of the arena (walls placed at +/- range+1)
    :param spawn_positions: list of (x, y) spawn positions, one per Ant
    :param collision_masks: list of (contype, conaffinity) per Ant
    :return: path to a written temp xml (caller is responsible for deleting)
    """
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    worldbody = root.find(".//worldbody")

    # drop the rllab init_qpos custom field (single-ant sized, unused by gym)
    custom = root.find(".//custom")
    if custom is not None:
        root.remove(custom)

    # detach the template torso body
    template = None
    for body in list(worldbody.findall("body")):
        if body.attrib.get("name") == "torso":
            template = body
            worldbody.remove(body)
            break
    if template is None:
        raise RuntimeError("could not find <body name='torso'> in base xml")

    # find the actuator block and detach its motors as templates
    actuator = root.find(".//actuator")
    motor_templates = list(actuator) if actuator is not None else []
    for m in list(actuator):
        actuator.remove(m)

    n_ants = len(spawn_positions)
    for i in range(n_ants):
        prefix = f"ant{i}_"
        ox, oy = spawn_positions[i]
        contype, conaffinity = collision_masks[i]

        ant = copy.deepcopy(template)
        _prefix_tree(ant, prefix)
        _set_collision(ant, contype, conaffinity)
        # start standing at the requested (x, y)
        ant.attrib["pos"] = f"{ox} {oy} 0.75"
        worldbody.append(ant)

        # duplicate the actuators, retargeting them to this Ant's joints
        for m in motor_templates:
            mm = copy.deepcopy(m)
            mm.attrib["joint"] = prefix + mm.attrib["joint"]
            if "name" in mm.attrib:
                mm.attrib["name"] = prefix + mm.attrib["name"]
            actuator.append(mm)

    # surrounding walls (mirrors TwoResourceEnv)
    attrs = dict(type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3")
    walldist = activity_range + 1
    ET.SubElement(worldbody, "geom", dict(attrs, name="wall1",
                  pos="0 -%d 1" % walldist, size="%d.5 0.5 2" % walldist))
    ET.SubElement(worldbody, "geom", dict(attrs, name="wall2",
                  pos="0 %d 1" % walldist, size="%d.5 0.5 2" % walldist))
    ET.SubElement(worldbody, "geom", dict(attrs, name="wall3",
                  pos="-%d 0 1" % walldist, size="0.5 %d.5 2" % walldist))
    ET.SubElement(worldbody, "geom", dict(attrs, name="wall4",
                  pos="%d 0 1" % walldist, size="0.5 %d.5 2" % walldist))

    fd, file_path = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    tree.write(file_path)
    return file_path


class _TwoAntMujocoEnv(MujocoEnv, utils.EzPickle):
    """Thin MuJoCo wrapper holding both Ants; obs handled by the TRP env."""
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(self, xml_path, n_ants=2, width=480, height=480):
        utils.EzPickle.__init__(self, xml_path, n_ants, width, height)
        self.n_ants = n_ants
        frame_skip = 5
        # dummy single-agent obs space; the TRP env builds the real per-agent one
        observation_space = Box(-np.inf, np.inf, (27,), np.float64)
        MujocoEnv.__init__(
            self, xml_path, frame_skip, observation_space,
            width=width, height=height,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )
        assert self.model.nq == n_ants * QPOS_PER_ANT, (self.model.nq, n_ants)
        assert self.model.nv == n_ants * QVEL_PER_ANT, (self.model.nv, n_ants)

    # MujocoEnv requires these; we drive the sim manually via do_simulation.
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        return self._zero_obs(), 0.0, False, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.standard_normal(size=self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._zero_obs()

    @staticmethod
    def _zero_obs():
        return np.zeros(27, dtype=np.float64)

    # ---- per-Ant accessors -------------------------------------------------
    def ant_proprioception(self, idx):
        q0 = idx * QPOS_PER_ANT
        v0 = idx * QVEL_PER_ANT
        qpos_i = self.data.qpos[q0:q0 + QPOS_PER_ANT]
        qvel_i = self.data.qvel[v0:v0 + QVEL_PER_ANT]
        return np.concatenate([qpos_i[2:], qvel_i]).astype(np.float32)

    def ant_com(self, idx):
        return self.get_body_com(f"ant{idx}_torso")

    def ant_quat(self, idx):
        q0 = idx * QPOS_PER_ANT
        return self.data.qpos.flat[q0 + 3:q0 + 7]

    def ant_ori(self, idx):
        rot = self.ant_quat(idx)
        ori = [0, 1, 0, 0]
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]
        return math.atan2(ori[1], ori[0])


class TwoAgentLowGearAntTRPEnv:
    """Two homeostatic low-gear Ants competing for a shared food field."""

    BASE_MODEL = "low_gear_ratio_ant.xml"

    def __init__(self,
                 n_blue=6,
                 n_red=4,
                 activity_range=8.,
                 robot_object_spacing=2.,
                 catch_range=1.,
                 n_bins=20,
                 sensor_range=16.,
                 sensor_span=2 * math.pi,
                 coef_main_rew=100.,
                 coef_ctrl_cost=0.001,
                 coef_head_angle=0.005,
                 dying_cost=-10.,
                 max_episode_steps=np.inf,
                 reward_setting="homeostatic_shaped",
                 reward_bias=None,
                 internal_reset="random",
                 internal_random_range=(-1. / 6, 1. / 6),
                 blue_nutrient=(0.1, 0),
                 red_nutrient=(0, 0.1),
                 spawn_positions=((-3., 0.), (3., 0.)),
                 show_sensor_range=False,
                 show_move_line=False,
                 width=480,
                 height=480):
        self.n_agents = len(spawn_positions)
        assert self.n_agents == 2, "this env is written for exactly two Ants"

        self.n_blue = n_blue
        self.n_red = n_red
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.coef_main_rew = coef_main_rew
        self.coef_ctrl_cost = coef_ctrl_cost
        self.coef_head_angle = coef_head_angle
        self.dying_cost = dying_cost
        self._max_episode_steps = max_episode_steps
        self.reward_setting = reward_setting
        self.reward_bias = reward_bias if reward_bias else 0.
        self.internal_reset = internal_reset
        self.internal_random_range = internal_random_range
        self.blue_nutrient = blue_nutrient
        self.red_nutrient = red_nutrient
        self.show_sensor_range = show_sensor_range
        self.show_move_line = show_move_line

        self._target_internal_state = np.array([0.0, 0.0])  # [Blue, Red]
        self.default_metabolic_update = 0.00015
        self.survival_area = 1.0

        self.internal_state = [
            {FoodClass.BLUE: 0.0, FoodClass.RED: 0.0} for _ in range(self.n_agents)
        ]
        self.prev_interoception = [self.get_interoception(a) for a in range(self.n_agents)]

        # build the two-Ant model and instantiate the shared sim
        import pathlib
        model_dir = pathlib.Path(__file__).parent / "models" / self.BASE_MODEL
        collision_masks = [(1, 2), (2, 1)]  # ant0 / ant1 collide with each other, not selves
        file_path = build_two_ant_model_xml(
            str(model_dir), activity_range, list(spawn_positions), collision_masks,
        )
        try:
            self.sim = _TwoAntMujocoEnv(file_path, n_ants=self.n_agents,
                                        width=width, height=height)
        finally:
            try:
                os.remove(file_path)
            except OSError:
                pass

        self.objects = []
        self._step = 0
        self.num_eaten = [[0, 0] for _ in range(self.n_agents)]
        self.agent_positions = [deque(maxlen=300) for _ in range(self.n_agents)]

        # observation / action spaces (per single agent)
        obs_dim = self.get_agent_obs(0).shape[0]
        ub = BIG * np.ones(obs_dim, dtype=np.float32)
        self.single_observation_space = spaces.Box(-ub, ub, dtype=np.float32)
        self.single_action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_ACT_PER_ANT,), dtype=np.float32,
        )

    # ---- interoception / reward -------------------------------------------
    @property
    def dim_intero(self):
        return int(np.prod(self._target_internal_state.shape))

    def get_interoception(self, agent):
        return np.array(list(self.internal_state[agent].values()), dtype=np.float32)

    def reset_internal_state(self):
        for a in range(self.n_agents):
            if self.internal_reset == "setpoint":
                self.internal_state[a] = {FoodClass.BLUE: 0.0, FoodClass.RED: 0.0}
            elif self.internal_reset == "random":
                lo, hi = self.internal_random_range
                self.internal_state[a] = {
                    FoodClass.BLUE: self.sim.np_random.uniform(lo, hi),
                    FoodClass.RED: self.sim.np_random.uniform(lo, hi),
                }
            else:
                raise ValueError('internal_reset should be "setpoint" or "random"')

    def step_internal_state_default(self):
        for a in range(self.n_agents):
            self.internal_state[a][FoodClass.RED] -= self.default_metabolic_update
            self.internal_state[a][FoodClass.BLUE] -= self.default_metabolic_update

    def update_by_food(self, agent, is_red, is_blue):
        if is_red:
            self.internal_state[agent][FoodClass.BLUE] += self.red_nutrient[0]
            self.internal_state[agent][FoodClass.RED] += self.red_nutrient[1]
        if is_blue:
            self.internal_state[agent][FoodClass.BLUE] += self.blue_nutrient[0]
            self.internal_state[agent][FoodClass.RED] += self.blue_nutrient[1]

    def get_reward(self, agent, action, num_blue_eaten, num_red_eaten):
        # motor cost ([0,1] policy action -> [-1,1] ctrl was applied, cost on that)
        ctrl = 2.0 * np.asarray(action) - 1.0
        ctrl_cost = -.5 * np.square(ctrl).sum()

        # local posture cost from this Ant's quaternion
        euler = qtoeuler(self.sim.ant_quat(agent))
        euler_stand = qtoeuler([1.0, 0.0, 0.0, 0.0])
        head_angle_cost = -np.square(euler[:2] - euler_stand[:2]).sum()

        total_cost = self.coef_ctrl_cost * ctrl_cost + self.coef_head_angle * head_angle_cost

        def drive(intero, target):
            dm = -1 * (intero - target) ** 2
            return dm.sum(), dm

        info = {"reward_module": None}
        rs = self.reward_setting
        target = self._target_internal_state
        if rs == "homeostatic":
            d, dm = drive(self.prev_interoception[agent], target)
            main_reward = d
            info["reward_module"] = np.concatenate([self.coef_main_rew * dm, [total_cost]])
        elif rs == "homeostatic_shaped":
            d, dm = drive(self.get_interoception(agent), target)
            d_prev, dm_prev = drive(self.prev_interoception[agent], target)
            main_reward = d - d_prev
            info["reward_module"] = np.concatenate(
                [self.coef_main_rew * (dm - dm_prev), [total_cost]])
        elif rs == "homeostatic_biased":
            d, dm = drive(self.prev_interoception[agent], target)
            main_reward = d + self.reward_bias
            info["reward_module"] = np.concatenate([self.coef_main_rew * dm, [total_cost]])
        elif rs == "greedy":
            main_reward = num_blue_eaten + num_red_eaten
        elif rs == "one":
            main_reward = 0.
        else:
            raise ValueError(rs)

        reward = self.coef_main_rew * main_reward + total_cost
        return reward, info

    # ---- sensors -----------------------------------------------------------
    def _sensor_readings(self, agent, points):
        """Depth-style readings (n_bins) toward a list of (x, y, tag) points.

        Returns a dict tag -> np.array(n_bins); closer points occlude farther
        ones within the same bin.
        """
        robot_x, robot_y = self.sim.ant_com(agent)[:2]
        ori = self.sim.ant_ori(agent)
        bin_res = self.sensor_span / self.n_bins
        half_span = self.sensor_span * 0.5
        tags = set(t for _, _, t in points)
        out = {t: np.zeros(self.n_bins) for t in tags}
        # farther first so nearer overwrites
        pts = sorted(points, key=lambda o: (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        for ox, oy, tag in pts:
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle -= 2 * math.pi
            if angle < -math.pi:
                angle += 2 * math.pi
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            bin_number = min(bin_number, self.n_bins - 1)
            intensity = 1.0 - dist / self.sensor_range
            out[tag][bin_number] = max(out[tag][bin_number], intensity)
        return out

    def get_food_readings(self, agent):
        pts = [(ox, oy, typ) for ox, oy, typ in self.objects]
        readings = self._sensor_readings(agent, pts)
        blue = readings.get(FoodClass.BLUE, np.zeros(self.n_bins))
        red = readings.get(FoodClass.RED, np.zeros(self.n_bins))
        return blue, red

    def get_other_ant_readings(self, agent):
        other = 1 - agent
        ox, oy = self.sim.ant_com(other)[:2]
        readings = self._sensor_readings(agent, [(ox, oy, "ant")])
        return readings.get("ant", np.zeros(self.n_bins))

    def get_agent_obs(self, agent):
        proprio = self.sim.ant_proprioception(agent)
        blue, red = self.get_food_readings(agent)
        other = self.get_other_ant_readings(agent)
        intero = self.get_interoception(agent)
        return np.concatenate([proprio, blue, red, other, intero]).astype(np.float32)

    def _all_obs(self):
        return np.stack([self.get_agent_obs(a) for a in range(self.n_agents)])

    # ---- food field --------------------------------------------------------
    def _sample_object(self, typ, existing):
        while True:
            x = self.sim.np_random.integers(-self.activity_range / 2,
                                            self.activity_range / 2 + 1) * 2
            y = self.sim.np_random.integers(-self.activity_range / 2,
                                            self.activity_range / 2 + 1) * 2
            if (x, y) in existing:
                continue
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            return (x, y, typ)

    def generate_new_object(self, type_gen):
        existing = set((o[0], o[1]) for o in self.objects)
        return self._sample_object(type_gen, existing)

    def _spawn_food(self):
        self.objects = []
        existing = set()
        for typ, n in ((FoodClass.BLUE, self.n_blue), (FoodClass.RED, self.n_red)):
            for _ in range(n):
                obj = self._sample_object(typ, existing)
                self.objects.append(obj)
                existing.add((obj[0], obj[1]))

    # ---- gym-ish API -------------------------------------------------------
    def reset(self, seed=None):
        self._step = 0
        self.num_eaten = [[0, 0] for _ in range(self.n_agents)]
        for dq in self.agent_positions:
            dq.clear()

        self.sim.reset(seed=seed)
        self.reset_internal_state()
        self.prev_interoception = [self.get_interoception(a) for a in range(self.n_agents)]
        self._spawn_food()

        info = {"interoception": [self.get_interoception(a) for a in range(self.n_agents)]}
        return self._all_obs(), info

    def step(self, actions):
        """actions: array-like (n_agents, 8) in [0, 1]."""
        actions = np.asarray(actions, dtype=np.float64).reshape(self.n_agents, N_ACT_PER_ANT)
        self.prev_interoception = [self.get_interoception(a) for a in range(self.n_agents)]

        # [0,1] policy output -> [-1,1] actuator ctrl, concatenated for both Ants
        ctrl = np.concatenate([2.0 * actions[a] - 1.0 for a in range(self.n_agents)])
        self.sim.do_simulation(ctrl, self.sim.frame_skip)

        coms = [self.sim.ant_com(a) for a in range(self.n_agents)]
        for a in range(self.n_agents):
            self.agent_positions[a].append(np.array(coms[a], np.float32))

        # broken-robot death check (non-finite state)
        broken = not np.isfinite(self.sim.state_vector()).all()

        # metabolism
        self.step_internal_state_default()

        # shared food eating: nearest Ant within catch_range eats
        self.num_eaten = [[0, 0] for _ in range(self.n_agents)]
        new_objs = []
        for ox, oy, typ in self.objects:
            eater, best = None, self.catch_range ** 2
            for a in range(self.n_agents):
                ax, ay = coms[a][:2]
                d2 = (ox - ax) ** 2 + (oy - ay) ** 2
                if d2 < best:
                    best, eater = d2, a
            if eater is not None:
                if typ is FoodClass.BLUE:
                    self.update_by_food(eater, is_red=False, is_blue=True)
                    self.num_eaten[eater][0] += 1
                else:
                    self.update_by_food(eater, is_red=True, is_blue=False)
                    self.num_eaten[eater][1] += 1
                new_objs.append(self.generate_new_object(typ))
            else:
                new_objs.append((ox, oy, typ))
        self.objects = new_objs

        self._step += 1

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        terminated = np.zeros(self.n_agents, dtype=bool)
        infos = []
        for a in range(self.n_agents):
            dead = broken or (np.max(np.abs(self.get_interoception(a))) > self.survival_area)
            terminated[a] = dead
            if dead:
                rewards[a] = self.dying_cost
                infos.append({"dead": True})
            else:
                r, ir = self.get_reward(a, actions[a],
                                        self.num_eaten[a][0], self.num_eaten[a][1])
                rewards[a] = r
                infos.append(ir)

        truncated = self._step >= self._max_episode_steps
        info = {
            "interoception": [self.get_interoception(a) for a in range(self.n_agents)],
            "food_eaten": self.num_eaten,
            "per_agent": infos,
        }
        return self._all_obs(), rewards, terminated, truncated, info

    @property
    def dt(self):
        return self.sim.dt

    def close(self):
        if getattr(self.sim, "mujoco_renderer", None) is not None:
            self.sim.mujoco_renderer.close()

    # ---- rendering ---------------------------------------------------------
    def render(self, mode="rgb_array", camera_id=None, camera_name=None):
        viewer = self.sim.mujoco_renderer._get_viewer(render_mode=mode)
        for obj in self.objects:
            ox, oy, typ = obj
            rgba = (0, 0, 1, 1) if typ is FoodClass.BLUE else (1, 0, 0, 1)
            viewer.add_marker(pos=np.array([ox, oy, 0.5]), label=" ",
                              type=mujoco.mjtGeom.mjGEOM_SPHERE,
                              size=(0.5, 0.5, 0.5), rgba=rgba)
        if self.show_move_line:
            colors = [(1, 0, 0, 0.3), (0, 1, 0, 0.3)]
            for a in range(self.n_agents):
                for pos in self.agent_positions[a]:
                    viewer.add_marker(pos=pos, label=" ",
                                      type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                      size=(0.05, 0.05, 0.05), rgba=colors[a], emission=1)
        im = self.sim.mujoco_renderer.render(mode, camera_id, camera_name)
        del viewer._markers[:]
        return im
