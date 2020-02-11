# augmentations for 2D and 2.5D
import random
import numpy as np
import SimpleITK as sitk

from src.utils.itk_tools import rotate_translate_scale2d


# Now only support list or tuple
class Compose(object):
    def __init__(self, augmentations):
        self._augmentations = augmentations

    def __call__(self, nda, nda_type):
        for _a in self._augmentations:
            nda = _a(nda, nda_type)
        return nda


# flip axis is array axis
class NPRandomFlip(object):

    def __init__(self, axis=0, do_probability=0.5):
        assert isinstance(axis, (int, tuple, list)), "Axis value type must be int, tuple or list."
        self._axis = axis
        self._p = do_probability

    def __call__(self, nda, nda_type):
        if random.random() < self._p:
            # set flip params
            if isinstance(self._axis, int):
                flip_axis = self._axis
                flip_axis_dim3 = self._axis + 1
            elif isinstance(self._axis, (tuple, list)):
                flip_axis = random.sample(self._axis, random.randint(1, len(self._axis)))
                flip_axis_dim3 = tuple([i + 1 for i in flip_axis])
            else:
                flip_axis = None
                flip_axis_dim3 = None
            # initialize out
            out = []
            # do flip
            for a, t in zip(nda, nda_type):
                if a.ndim == 3:
                    out.append(np.copy(np.flip(a, flip_axis_dim3)))
                else:
                    out.append(np.copy(np.flip(a, flip_axis)))
            out = tuple(out)
            return out
        else:
            return nda


# scale, translate, rotation
class ITKRandomRotateTranslateScale(object):

    def __init__(self, theta=0, tx=0, ty=0, scale=0, do_probability=0.5):
        self._theta = theta * np.pi / 180.0
        self._tx = tx
        self._ty = ty
        self._scale = scale
        self._p = do_probability

    def __call__(self, nda, nda_type):
        if random.random() < self._p:
            # set transform params
            transform_params = [(np.random.rand() * 2 - 1) * self._scale + 1,
                                (np.random.rand() * 2 - 1) * self._theta,
                                (np.random.rand() * 2 - 1) * self._tx,
                                (np.random.rand() * 2 - 1) * self._ty,
                                ]
            # initialize out
            out = []
            # do
            for a, t in zip(nda, nda_type):
                interpolator = "Linear" if t == "image" else "NearestNeighbor"
                default_v = float(np.amin(a)) if t == "image" else 0
                if a.ndim == 3:
                    tmp = []
                    for i in a.shape[0]:
                        tmp.append(sitk.GetArrayFromImage(rotate_translate_scale2d(
                            sitk.GetImageFromArray(a[i], isVector=False), transform_params, interpolator, default_v)))
                    out.append(np.stack(tmp, axis=0))
                else:
                    out.append(sitk.GetArrayFromImage(
                        rotate_translate_scale2d(
                            sitk.GetImageFromArray(a, isVector=False), transform_params, interpolator, default_v)))
            out = tuple(out)
            return out
        else:
            return nda
