import random
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from src.utils.itk_tools import rotate_translate_scale3d


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
            elif isinstance(self._axis, (tuple, list)):
                flip_axis = random.sample(self._axis, random.randint(1, len(self._axis)))
            else:
                flip_axis = None
            # initialize out
            out = []
            # do flip
            for a, t in zip(nda, nda_type):
                out.append(np.copy(np.flip(a, flip_axis)))
            out = tuple(out)
            return out
        else:
            return nda


# scale, translate, rotation
class ITKRandomRotateTranslateScale(object):

    def __init__(self, theta_x=0, theta_y=0, theta_z=0, tx=0, ty=0, tz=0, scale=0, do_probability=0.5):
        self._theta_x = theta_x * np.pi / 180.0
        self._theta_y = theta_y * np.pi / 180.0
        self._theta_z = theta_z * np.pi / 180.0
        self._tx = tx
        self._ty = ty
        self._tz = tz
        self._scale = scale
        self._p = do_probability

    def __call__(self, nda, nda_type):
        if random.random() < self._p:
            # set transform params
            transform_params = [(np.random.rand() * 2 - 1) * self._theta_x,
                                (np.random.rand() * 2 - 1) * self._theta_y,
                                (np.random.rand() * 2 - 1) * self._theta_z,
                                (np.random.rand() * 2 - 1) * self._tx,
                                (np.random.rand() * 2 - 1) * self._ty,
                                (np.random.rand() * 2 - 1) * self._tz,
                                (np.random.rand() * 2 - 1) * self._scale + 1]
            # initialize out
            out = []
            # do
            for a, t in zip(nda, nda_type):
                interpolator = "Linear" if t == "image" else "NearestNeighbor"
                default_v = float(np.amin(a)) if t == "image" else 0
                out.append(sitk.GetArrayFromImage(
                    rotate_translate_scale3d(
                        sitk.GetImageFromArray(a, isVector=False), transform_params, interpolator, default_v)))
            out = tuple(out)
            return out
        else:
            return nda


# Elastic deformation
# https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py
class NPRandomGridDeform(object):

    def __init__(self, sigma=3, points=3, do_probability=0.5):
        self._sigma = sigma  # sigma = standard deviation of the normal distribution
        self._pts = points  # points = number of points of the each side of the square grid
        self._p = do_probability

    def __call__(self, nda, nda_type):
        if random.random() < self._p:
            shape = nda[0].shape
            # creates the grid of coordinates of the points of the image (an ndim array per dimension)
            coordinates = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
            xi = np.meshgrid(np.linspace(0, self._pts - 1, shape[0]), np.linspace(0, self._pts - 1, shape[1]),
                             np.linspace(0, self._pts - 1, shape[2]), indexing='ij')
            grid = [self._pts, self._pts, self._pts]
            # creates the deformation along each dimension and then add it to the coordinates
            for i in range(len(shape)):
                yi = np.random.randn(*grid) * self._sigma  # creating the displacement at the control points
                y = map_coordinates(yi, xi, order=3).reshape(shape)
                coordinates[i] = np.add(coordinates[i], y)  # adding the displacement
            # initialize out
            out = []
            # do
            for a, t in zip(nda, nda_type):
                order = 3 if t == "image" else 0
                out.append(map_coordinates(a, coordinates, order=order, cval=np.amin(nda)).reshape(shape))
            out = tuple(out)
            return out
        else:
            return nda
