"""RLBench (github.com/stepjam/RLBench) environment wrapper."""
from typing import Sequence
from collections import defaultdict

import numpy as np
import gym

import rlbench
import rlbench.utils
import rlbench.backend
from rlbench.demo import Demo
from rlbench import Environment
from rlbench.backend.scene import Scene
from rlbench.backend.observation import Observation
from rlbench.backend.exceptions import InvalidActionError
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.const import SUPPORTED_ROBOTS


_TESTED_TASKS = ()
_DISABLED_CAMERA = CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False)
_ROBOT = "panda"
_ROBOT_DOF = SUPPORTED_ROBOTS[_ROBOT][2]


class ActionRescale:
    # However Dreamer has its own common.envs.NormalizeAction
    def __init__(self, lower_bound: np.ndarray, upper_bound: np.ndarray):
        lower_bound, upper_bound = map(
            lambda bound: np.asanyarray(bound, dtype=np.float32),
            (lower_bound, upper_bound)
        )
        self._slope = (upper_bound - lower_bound) / 2.
        self._inception = (upper_bound + lower_bound) / 2.

    def __call__(self, action):
        return self._inception + self._slope * action


class PostponedActionMode(MoveArmThenGripper):
    """
    RLBench.Environment requires action mode on initialization
    but workspace boundaries may be set after Scene init
    or after demos generation.
    """
    def __init__(self, arm_action_mode, gripper_action_mode):
        super().__init__(arm_action_mode, gripper_action_mode)
        self._lower_action_bounds = None
        self._upper_action_bounds = None

    def action_bounds(self):
        self._assert_bounds_set()
        return (
            self._lower_action_bounds,
            self._upper_action_bounds
        )

    def set_bounds(self, lower_action_bound: np.ndarray, upper_action_bound: np.ndarray):
        self._lower_action_bounds = lower_action_bound
        self._upper_action_bounds = upper_action_bound

    def _assert_bounds_set(self):
        assert isinstance(self._lower_action_bounds, np.ndarray), "Set action bounds first."


class JointsWithDiscrete(PostponedActionMode):
    def __init__(self):
        super().__init__(
            JointPosition(absolute_mode=False),
            Discrete(attach_grasped_objects=True,
                     detach_before_open=True)
        )

    def action(self, scene: Scene, action: np.ndarray):
        self._assert_bounds_set()
        return super().action(scene, action)


class EndEffectorWithDiscrete(PostponedActionMode):

    def __init__(self):
        super().__init__(
            EndEffectorPoseViaPlanning(
                absolute_mode=True,
                frame="world",
                collision_checking=False
            ),
            Discrete(
                attach_grasped_objects=True,
                detach_before_open=True
            )
        )

    def action(self, scene: Scene, action: np.ndarray):
        self._assert_bounds_set()
        pos, rot, grip = np.split(action, [3, 7], axis=-1)
        pos = np.clip(
            pos,
            a_min=self._lower_action_bounds[:3],
            a_max=self._upper_action_bounds[:3]
        )
        rot /= np.linalg.norm(rot)
        action = np.concatenate([pos, rot, grip], axis=-1)
        return super().action(scene, action)


def _make_observation_config(image_size):
    """There is a rich space for randomization and customization.
    However, only image size is used for now."""
    enabled_camera_config = CameraConfig(image_size=image_size)
    return ObservationConfig(
        left_shoulder_camera=_DISABLED_CAMERA,
        right_shoulder_camera=_DISABLED_CAMERA,
        overhead_camera=_DISABLED_CAMERA,
        wrist_camera=_DISABLED_CAMERA,
        front_camera=enabled_camera_config,
        joint_velocities=True,
        joint_positions=True,
        joint_forces=True,
        gripper_open=True,
        gripper_pose=True,
        gripper_matrix=False,
        gripper_joint_positions=False,
        gripper_touch_forces=False,
        task_low_dim_state=False
    )


def _workspace_bounds(scene: Scene):
    """Bounds from the scene workspace."""
    from itertools import product
    scene_bounds = [
        getattr(scene, f"_workspace_{mode}{axis}")
        for mode, axis in product(("min", "max"), "xyz")
    ]
    scene_bounds = np.float32(scene_bounds)
    return np.split(scene_bounds, 2)


def _bounds_from_demos(demos: Sequence[Demo], fields: Sequence[str]):
    """Iterate over prerecorded demos to infer values bounds
    over requested fields."""
    observations = defaultdict(list)
    for demo in demos:
        for field in fields:
            observations[field].extend([getattr(obs, field) for obs in demo])

    observations = {k: np.float32(v) for k, v in observations.items()}
    bounds = {k: [v.min(axis=0), v.max(axis=0)] for k, v in observations.items()}
    return bounds


def _modify_action_min_max(action_min_max):
    """
    Copied directly from the Stephan's ARM repo
    for proper comparison.
    https://github.com/stepjam/ARM/blob/main/launch.py#L74
    """
    # Make translation bounds a little bigger
    action_min_max[0][0:3] -= np.fabs(action_min_max[0][0:3]) * 0.2
    action_min_max[1][0:3] += np.fabs(action_min_max[1][0:3]) * 0.2
    action_min_max[0][-1] = 0
    action_min_max[1][-1] = 1
    action_min_max[0][3:7] = np.array([-1, -1, -1, 0])
    action_min_max[1][3:7] = np.array([1, 1, 1, 1])
    return action_min_max


# todo: deal with sparse rewards: now dense only available for reach_target
class RLBenchEnv:

    def __init__(self,
                 name: str,
                 size: tuple = (64, 64),
                 action_repeat: int = 1,
                 pn_number: int = 100
                 ):
        obs_config = _make_observation_config(size)
        task = rlbench.utils.name_to_task_class(name)

        self._action_mode = JointsWithDiscrete()
        self._env = Environment(
            self._action_mode,
            obs_config=obs_config,
            headless=True,
            robot_setup=_ROBOT,
            static_positions=False,
            attach_grasped_objects=True,
            shaped_rewards=False, #(name == "reach_target"),
        )
        self._env.launch()
        self._task = self._env.get_task(task)
        self._scene = self._task._scene

        lower_action_bounds, upper_action_bounds = self._get_actions_bounds()
        self._action_mode.set_bounds(lower_action_bounds, upper_action_bounds)
        self._action_rescaler = ActionRescale(
            lower_action_bounds, upper_action_bounds)

        self._action_repeat = action_repeat
        self._size = size
        self._pn_number = pn_number
        self._prev_observation = None

    def reset(self):
        desc, obs = self._task.reset()
        self._prev_observation = obs
        obs = self._observation(obs)
        obs.update(
            is_first=True,
            reward=0.0,
            is_last=False,
            is_terminal=False
        )
        return obs

    def step(self, action):
        action = action["action"]
        assert np.isfinite(action).all(), action
        action = self._action_rescaler(action)

        obs = self._prev_observation
        reward = 0.0
        try:
            for _ in range(self._action_repeat):
                obs, r, done = self._task.step(action)
                reward += r or 0.0
                self._prev_observation = obs
                if done:
                    break
        except InvalidActionError:
            done = True
            reward = 0.

        obs = self._observation(obs)
        obs.update(
            reward=reward,
            is_first=False,
            is_last=done,
            is_terminal=done
        )
        return obs

    @property
    def act_space(self):
        ones = np.ones_like(self._action_mode.action_bounds()[0], dtype=np.float32)
        act_space = gym.spaces.Box(
            low=-ones,
            high=ones,
            dtype=np.float32,
            shape=ones.shape,
        )
        return {"action": act_space}

    @property
    def obs_space(self):
        joint_shape = (_ROBOT_DOF,)
        return {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "depth": gym.spaces.Box(0, np.inf, self._size, dtype=np.float32),
            "flat_point_cloud": gym.spaces.Box(-np.inf, np.inf, self._size + (3,),
                                               dtype=np.float64),
            "point_cloud": gym.spaces.Box(-np.inf, np.inf, (self._pn_number, 3), dtype=np.float64),
            "joint_positions": gym.spaces.Box(-np.inf, np.inf, joint_shape, dtype=np.float64),
            "joint_velocities": gym.spaces.Box(-np.inf, np.inf, joint_shape, dtype=np.float64),
            "joint_forces": gym.spaces.Box(-np.inf, np.inf, joint_shape, dtype=np.float64),
            "gripper_open": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
            "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float64),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "if_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }

    def _observation(self, obs: Observation):
        return {
            "depth": obs.front_depth,
            "image": obs.front_rgb,
            "flat_point_cloud": obs.front_point_cloud,
            "point_cloud": self._get_pc(obs.front_point_cloud),
            "joint_positions": obs.joint_positions,
            "joint_velocities": obs.joint_velocities,
            "joint_forces": obs.joint_forces,
            "gripper_open": np.array(obs.gripper_open, dtype=np.float32)[np.newaxis],
            "gripper_pose": obs.gripper_pose
        }

    def _get_pc(self, flat_pc):
        flat_pc = flat_pc.reshape(-1, 3)
        stride = flat_pc.shape[0] // self._pn_number
        pcd = flat_pc[::stride][:self._pn_number]
        assert pcd.shape[0] == self._pn_number
        return pcd

    def _get_actions_bounds(self):
        demos = self._task.get_demos(amount=7, live_demos=True)
        action_bounds = _bounds_from_demos(demos, ("gripper_pose",))
        lower_bounds, upper_bounds = action_bounds["gripper_pose"]
        # Append gripper bounds.
        lower_bounds = np.concatenate((lower_bounds, [0]), dtype=np.float32)
        upper_bounds = np.concatenate((upper_bounds, [1]), dtype=np.float32)
        lower_bounds, upper_bounds = _modify_action_min_max([lower_bounds, upper_bounds])

        scene_bounds = _workspace_bounds(self._scene)
        lower_bounds[:3] = np.maximum(lower_bounds[:3], scene_bounds[0])
        upper_bounds[:3] = np.minimum(upper_bounds[:3], scene_bounds[1])

        if isinstance(self._action_mode, JointsWithDiscrete):
            lower_bounds[:-1] /= 10.
            upper_bounds[:-1] /= 10.

        return lower_bounds, upper_bounds

    def close(self):
        self._env.shutdown()

    def prepare_demos(self, amount: int = 10):
        demos = self._task.get_demos(amount, live_demos=True)
        episodes = []
        for demo in demos:
            length = len(demo)
            episode = defaultdict(list)
            for i, obs in enumerate(demo):
                is_last = (i == length - 1)
                obs = self._observation(obs)
                arm_action = obs["gripper_pose"]
                gripper_action = obs["gripper_open"] < 0.5  # that's not it
                action = np.concatenate([arm_action, gripper_action], dtype=np.float32)
                obs.update(
                    reward=float(is_last),
                    is_first=(i == 0),
                    is_last=is_last,
                    is_terminal=is_last,
                    action=action
                )
                for k, v in obs.items():
                    episode[k].append(v)

            episodes.append(episode)

        return episodes
