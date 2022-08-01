from typing import NamedTuple, TypedDict, Union, Iterable

import numpy as np
from dm_control.mujoco.engine import Camera


class IntrinsicParams(NamedTuple):
    width: float
    height: float
    fx: float
    fy: float
    cx: float
    cy: float


class CameraParams(TypedDict):
    width: float
    height: float
    camera_id: Union[int, str]


def list_geom_ids_by_names(physics, geom_names, **render_kwargs):
    geom_ids = physics.render(segmentation=True, **render_kwargs)[..., 0]
    geom_ids = np.unique(geom_ids).tolist()

    def _predicate(geom_id):
        return any(
            map(
                lambda name: name in physics.model.id2name(geom_id, 'geom'),
                geom_names
            )
        )

    in_and_out = ([], [])
    for _id in geom_ids:
        in_and_out[not _predicate(_id)].append(_id)
    return in_and_out


def point_cloud_from_depth_map(depth, intrinsic_params):
    assert depth.shape == (intrinsic_params.height, intrinsic_params.width), "Incompatible shapes"
    yy, xx = np.mgrid[:intrinsic_params.height, :intrinsic_params.width]
    xx = (xx - intrinsic_params.cx) / intrinsic_params.fx * depth
    yy = (yy - intrinsic_params.cy) / intrinsic_params.fy * depth

    return np.stack([xx, yy, depth], axis=-1).reshape(-1, 3)


def camera_params_from_physics(physics, render_kwargs: CameraParams):
    """Get required params from physics"""
    # todo: rotation + translation
    cam = Camera(physics, **render_kwargs)
    image, focal, rotation, translation = cam.matrices()
    intrinsic = (image @ focal)[:, :-1]
    extrinsic_mat = rotation @ translation

    intrinsic_params = IntrinsicParams(
        height=render_kwargs['height'],
        width=render_kwargs['width'],
        fx=-intrinsic[0, 0],
        fy=intrinsic[1, 1],
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2]
    )

    return intrinsic_params, extrinsic_mat


# TODO: global transformation
# TODO: append rgb inputs
class PointCloudGenerator:
    def __init__(self,
                 pn_number: int,
                 cameras_params: Iterable[CameraParams],
                 stride: int = -1
                 ):

        self.stride = stride
        self.pn_number = pn_number
        self._cams = cameras_params
        self._bad_geom_names = ('wall', 'ground', 'floor')

    def __call__(self, physics):
        """Merge cameras views to single point cloud."""
        pcd = np.concatenate([
            self.call(physics, cam) for cam in self._cams
        ])
        pcd = self._apply_stride(pcd)

        return self._to_fixed_number(pcd).astype(np.float16)

    def call(self, physics, render_kwargs: CameraParams):
        """Per camera pcd generation."""
        depth = physics.render(depth=True, **render_kwargs)

        intrinsic_params, extrinsic_mat = camera_params_from_physics(physics, render_kwargs)
        pcd = point_cloud_from_depth_map(depth, intrinsic_params)
        mask = self._mask(physics, pcd, render_kwargs)
        pcd = pcd[mask]

        # TODO: fix transformation
        inv_rotation = physics.named.data.cam_xmat[render_kwargs['camera_id']].reshape(3, 3)
        translation = physics.named.data.cam_xpos[render_kwargs['camera_id']]
        # pcd = np.einsum('ij, nj -> ni', inv_rotation, pcd)
        # pcd += translations
        # inv_ex = np.linalg.inv(extrinsic_mat)[:-1]
        # ones = np.ones_like(pcd[:, -1:])
        # pcd = np.concatenate((pcd, ones), axis=-1)
        # pcd = np.einsum('ij, nj -> ni', inv_ex, pcd)

        return pcd

    def _apply_stride(self, pcd):
        if self.stride < 0:
            adaptive_stride = pcd.shape[0] // self.pn_number
            return pcd[::max(adaptive_stride, 1)]
        else:
            return pcd[::self.stride]

    def _mask(self, physics, point_cloud, render_kwargs):
        segmentation = physics.render(segmentation=True, **render_kwargs)
        _, geom_ids = list_geom_ids_by_names(physics, self._bad_geom_names, **render_kwargs)
        geom_ids.remove(-1) # sky renders infinity
        segmentation = np.isin(segmentation[..., 0].flatten(), geom_ids)

        truncate = point_cloud[..., 2] < 10.
        return np.logical_and(segmentation, truncate)

    def _to_fixed_number(self, pcd):
        n = pcd.shape[0]
        if n == 0:
            pcd = np.zeros((self.pn_number, 3))
        elif n <= self.pn_number:
            pcd = np.pad(pcd, ((0, self.pn_number - n), (0, 0)), mode='edge')
        else:
            pcd = np.random.permutation(pcd)[:self.pn_number]
        return pcd
