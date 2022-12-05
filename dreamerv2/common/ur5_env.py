from collections import OrderedDict

import numpy as np
import gym.spaces

from ur_env.remote import RemoteEnvClient, Address


def _is_image(key):
    img_keys = ("image", "depth")
    return any(map(lambda k: k in key, img_keys))


class UR5:
    """Only several default rl related methods are expose.
    Otherwise, there is no much control over remote robot.
    """
    def __init__(self, action_repeat: int = 1, size: tuple = (64, 64)):
        address = ("10.201.2.136", 5555)
        self._env = RemoteEnvClient(address)

        self._action_repeat = action_repeat
        self._size = size

    def reset(self):
        ts = self._env.reset()
        obs = self._observation(ts.observation)

        obs.update(
            is_first=True,
            reward=0.,
            is_last=False,
            is_terminal=False,
        )
        return obs

    def step(self, action):
        action = action['action']
        reward = 0.0
        for _ in range(self._action_repeat):
            ts = self._env.step(action)
            reward += ts.reward
            if ts.last():
                break

        obs = self._observation(ts.observation)
        obs.update(
            reward=reward,
            is_first=False,
            is_last=ts.last(),
            is_terminal=(ts.discount == 0.),
        )
        return obs

    def _observation(self, obs):
        # for key, value in obs.items():
        #     if _is_image(key):
        #         img = Image.fromarray(value)
        #         img = img.resize(self._size)
        #         obs[key] = np.asarray(img)
        return obs

    @property
    def obs_space(self):
        obs_spec = OrderedDict()
        for key, spec in self._env.observation_spec().items():
            if spec.dtype == np.uint8:
                low, high = 0, 255
            else:
                low, high = -np.inf, np.inf
            obs_spec[key] = gym.spaces.Box(
                low=low,
                high=high,
                shape=spec.shape,
                dtype=spec.dtype
            )
        return obs_spec

    @property
    def act_space(self):
        spec = self._env.action_spec()
        spec = gym.spaces.Box(
            low=spec.minimum,
            high=spec.maximum,
            shape=spec.shape,
            dtype=spec.dtype
        )
        return {"action": spec}

    def __getattr__(self, item):
        return getattr(self._env, item)

