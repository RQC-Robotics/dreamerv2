"""RLBench (github.com/stepjam/RLBench) environment wrapper."""
import numpy as np
import gym
import rlbench
import rlbench.backend
from rlbench import Environment
from rlbench.action_modes.action_mode import JointPositionActionMode
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench import const

_TESTED_TASKS = ()
_DISABLED_CAMERA = CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False)
_ROBOT = "panda"
_ROBOT_ACTION_DIM = const.SUPPORTED_ROBOTS[_ROBOT][2]


# TODO: Decide how to create action_mode:
#  this includes choices of to use or not IK/planning, absolute/relative control step,
#  discrete/cont gripper action.
def _make_action_mode(*args, **kwargs):
    action_mode = JointPositionActionMode()
    assert hasattr(action_mode, "action_bounds"), ".action_bounds method is not implemented"
    return action_mode


class VariableActionMode(JointPositionActionMode):
    def __init__(self, robot_action_dim):
        super().__init__()
        self._robot_act_dim = robot_action_dim

    def action_bounds(self):
        return (
            np.array(self._robot_act_dim * [-0.1] + [0.0]),
            np.array(self._robot_act_dim * [0.1] + [0.04])
        )


def _make_observation_config(image_size):
    """There is a rich space for randomization and customization.
    However, now only image size is used."""
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


def _rescale_action(action, lower_bound, upper_bound):
    """If default action bounds differ from [-1, 1]^n this rescales it accordingly."""
    return (upper_bound + lower_bound) / 2. + (upper_bound - lower_bound) / 2. * action


# todo: deal with sparse rewards: now dense only available for reach_target
class RLBenchEnv:
    def __init__(self, name: str, size: tuple = (64, 64), action_repeat: int = 1,
                 pn_number: int = 100):
        action_mode = VariableActionMode(_ROBOT_ACTION_DIM)
        obs_config = _make_observation_config(size)
        task = rlbench.utils.name_to_task_class(name)
        self._lower_action_bound, self._upper_action_bound = action_mode.action_bounds()
        self._env = Environment(
            action_mode,
            obs_config=obs_config,
            headless=True,
            robot_setup=_ROBOT,
            shaped_rewards=(name == "reach_target")
        )
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
        rescaled_action = _rescale_action(action, self._lower_action_bound,
                                          self._upper_action_bound)
        reward = 0.0
        for _ in range(self._action_repeat):
            obs, r, done = self._task.step(rescaled_action)
            reward += r or 0.0
            if done:
                break

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
        action = gym.spaces.Box(-1, 1, (_ROBOT_ACTION_DIM + 1,), dtype=np.float32)
        return {"action": action}

    @property
    def obs_space(self):
        pos_shape = self._env.action_shape
        return {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "depth": gym.spaces.Box(0, np.inf, self._size, dtype=np.float32),
            "flat_point_cloud": gym.spaces.Box(-np.inf, np.inf, self._size + (3,),
                                               dtype=np.float64),
            "point_cloud": gym.spaces.Box(-np.inf, np.inf, (self._pn_number, 3), dtype=np.float64),
            "positions": gym.spaces.Box(-np.inf, np.inf, pos_shape, dtype=np.float64),
            "velocities": gym.spaces.Box(-np.inf, np.inf, pos_shape, dtype=np.float64),
            "gripper_open": gym.spaces.Box(0, 1, (), dtype=float),
            "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float64),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "if_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    def _observation(self, obs: rlbench.backend.observation.Observation):
        return {
            "depth": obs.front_depth,
            "image": obs.front_rgb,
            "flat_point_cloud": obs.front_point_cloud,
            "point_cloud": self._get_pc(obs.front_point_cloud),
            "positions": obs.joint_positions,
            "velocities": obs.joint_velocities,
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
