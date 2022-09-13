"""RLBench (github.com/stepjam/RLBench) environment wrapper."""
from typing import Optional, Sequence

import numpy as np
import gym

import rlbench
import rlbench.utils
import rlbench.backend
from rlbench import Environment
from rlbench.action_modes.action_mode import JointPositionActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.const import SUPPORTED_ROBOTS

_TESTED_TASKS = ()
_DISABLED_CAMERA = CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False)
_ROBOT = "panda"
_ROBOT_ACTION_DIM = SUPPORTED_ROBOTS[_ROBOT][2]


class VariableActionMode(JointPositionActionMode):
    def __init__(self, robot_action_dim):
        super().__init__()
        self._robot_act_dim = robot_action_dim

    def action_bounds(self):
        return (
            np.array(self._robot_act_dim * [-0.1] + [0.0]),
            np.array(self._robot_act_dim * [0.1] + [0.04])
        )


class ActionRescale:
    def __init__(self, lower_bound: np.ndarray, upper_bound: np.ndarray):
        lower_bound, upper_bound = map(
            lambda bound: np.asanyarray(bound, dtype=np.float32),
            (lower_bound, upper_bound)
        )
        self._slope = (upper_bound - lower_bound) / 2.
        self._inception = (upper_bound + lower_bound) / 2.

    def __call__(self, action):
        return self._inception + self._slope * action


class EndEffectorWithDiscrete(MoveArmThenGripper):
    _default_scene_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]

    def __init__(self,
                 scene_bounds: Optional[Sequence[float]] = None
                 ):
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
        scene_bounds = list(scene_bounds) if scene_bounds else self._default_scene_bounds
        assert len(scene_bounds) == 6, f"Wrong scene bounds specification: {scene_bounds}"
        self._lower_action_bounds = np.array(scene_bounds[:3] + 4 * [-1] + [0], dtype=np.float32)
        self._upper_action_bounds = np.array(scene_bounds[3:] + 4 * [1] + [1], dtype=np.float32)

    def action(self, scene, action):
        pos, rot, grip = np.split(action, [3, 7], axis=-1)
        rot /= np.linalg.norm(rot)
        action = np.concatenate([pos, rot, grip], axis=-1)
        return super().action(scene, action)

    def action_bounds(self):
        return self._lower_action_bounds, self._upper_action_bounds


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


# todo: deal with sparse rewards: now dense only available for reach_target
class RLBenchEnv:
    def __init__(self, name: str, size: tuple = (64, 64), action_repeat: int = 1,
                 pn_number: int = 100):
        action_mode = EndEffectorWithDiscrete()
        self._action_mode = action_mode
        self._action_rescaler = ActionRescale(*action_mode.action_bounds())

        obs_config = _make_observation_config(size)
        task = rlbench.utils.name_to_task_class(name)

        self._env = Environment(
            action_mode,
            obs_config=obs_config,
            headless=True,
            robot_setup=_ROBOT,
            attach_grasped_objects=True,
            shaped_rewards=(name == "reach_target"),
        )
        self._env.launch()
        self._task = self._env.get_task(task)

        self._action_repeat = action_repeat
        self._size = size
        self._pn_number = pn_number

    def reset(self):
        desc, obs = self._task.reset()
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

        reward = 0.0
        for _ in range(self._action_repeat):
            obs, r, done = self._task.step(action)
            reward += r or 0.0
            if done:
                break

        obs = self._observation(obs)
        obs.update(
            reward=reward,
            is_first=False,
            is_last=done,
            is_terminal=done  # catch invalid IK or PlanningError
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
        pos_shape = self._env.action_shape
        return {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "depth": gym.spaces.Box(0, np.inf, self._size, dtype=np.float32),
            "flat_point_cloud": gym.spaces.Box(-np.inf, np.inf, self._size + (3,),
                                               dtype=np.float64),
            "point_cloud": gym.spaces.Box(-np.inf, np.inf, (self._pn_number, 3), dtype=np.float64),
            "joint_positions": gym.spaces.Box(-np.inf, np.inf, pos_shape, dtype=np.float64),
            "joint_velocities": gym.spaces.Box(-np.inf, np.inf, pos_shape, dtype=np.float64),
            "joint_forces": gym.spaces.Box(-np.inf, np.inf, pos_shape, dtype=np.float64),
            "gripper_open": gym.spaces.Box(0, 1, (), dtype=float),
            "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float64),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "if_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }

    def _observation(self, obs: rlbench.backend.observation.Observation):
        return {
            "depth": obs.front_depth,
            "image": obs.front_rgb,
            "flat_point_cloud": obs.front_point_cloud,
            "point_cloud": self._get_pc(obs.front_point_cloud),
            "joint_positions": obs.joint_positions,
            "joint_velocities": obs.joint_velocities,
            "joint_forces": obs.joint_forces,
            "gripper_open": obs.gripper_open,
            "gripper_pose": obs.gripper_pose
        }

    def _get_pc(self, flat_pc):
        flat_pc = flat_pc.reshape(-1, 3)
        stride = flat_pc.shape[0] // self._pn_number
        pcd = flat_pc[::stride][:self._pn_number]
        assert pcd.shape[0] == self._pn_number
        return pcd

    def close(self):
        self._env.shutdown()
