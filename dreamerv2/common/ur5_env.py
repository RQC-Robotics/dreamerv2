import numpy as np
from PIL import Image

from ur_env.remote import RemoteEnvClient, Address


def _is_image(key):
    img_keys = ("image", "depth")
    return any(map(lambda k: k in key, img_keys))


class UR5:
    """Only several default rl related methods are expose.
    Otherwise, there is no much control over remote robot.
    """
    def __init__(self, action_repeat: int = 1, size: tuple = (64, 64)):
        address = ("10.201.2.136", 4445)
        self._env = RemoteEnvClient(address)
        self._action_repeat = action_repeat
        self._size = size

    def reset(self):
        ts = self._env.reset()
        obs = self._observation(ts.observation)

        obs.update(
            is_first=True,
            reward=ts.reward,
            is_last=False,
            is_terminal=False,
        )
        return obs

    def step(self, action):
        action = action['action']
        action = {'arm': np.concatenate([action, np.zeros((3,))])}
        reward = 0.0
        for _ in range(self._action_repeat):
            obs, r, done, extra = self._env.step(action)
            reward += r
            if done:
                break

        obs = self._observation(obs)
        obs.update(
            reward=reward,
            is_first=False,
            is_last=done,
            is_terminal=done,
        )

        return obs

    def _observation(self, obs):
        for key, value in obs.items():
            if _is_image(key):
                img = Image.fromarray(value)
                img = img.resize(self._size)
                obs[key] = np.asarray(img)
        return obs

    @property
    def obs_space(self):
        obs_space = self._env.observation_space
        for key, space in obs_space.items():
            if _is_image(key):
                new_shape = self._size + tuple(space.shape[2:])
                new_space = type(space)(
                    np.zeros(new_shape, dtype=space.dtype),
                    np.ones(new_shape, dtype=space.dtype),
                    new_shape,
                    space.dtype
                )
                obs_space[key] = new_space

        return obs_space
    
    @property
    def act_space(self):
        return {"action": self._env.action_space}

    def __getattr__(self, item):
        return getattr(self._env, item)

