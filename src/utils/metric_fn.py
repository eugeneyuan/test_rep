# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure


class Metrics(object):

    def __init__(self, voxel_spacing=None, num_classes=2, metric_list=('dice', ), activation_fn='sigmoid', activated=True):
        self.voxel_spacing = voxel_spacing
        self.num_classes = num_classes
        self.metric_list = metric_list
        self.activation_fn = activation_fn
        self.activated = activated

    @staticmethod
    def get_metric_fn(metric_name):
        if metric_name.lower() == 'dice':
            return dc
        elif metric_name.lower() == 'hausdorff distance':
            return hd
        elif metric_name.lower() == 'average symmetric surface distance':
            return assd
        else:
            raise NotImplementedError

    def compute_metric(self, inputs, targets, metric_name):

        if metric_name.lower() == 'dice':
            return dc(inputs, targets)
        elif metric_name.lower() == 'hausdorff distance':
            return hd(inputs, targets, voxelspacing=self.voxel_spacing)
        elif metric_name.lower() == 'average symmetric surface distance':
            return assd(inputs, targets, voxelspacing=self.voxel_spacing)
        else:
            raise NotImplementedError

    def compute_class_metrics(self, inputs, targets):
        bs = inputs.size(0)
        metric_array = np.zeros((self.num_classes-1, len(self.metric_list)), dtype=np.float16)

        if self.activation_fn == 'sigmoid':
            if not self.activated:
                inputs = torch.sigmoid(inputs)
            preds = np.asarray(inputs.data.cpu().numpy() > 0.5)
            preds = np.squeeze(preds, axis=1)
        elif self.activation_fn == 'softmax':
            if not self.activated:
                inputs = F.softmax(inputs, dim=1)
            _, preds = inputs.data.cpu().max(1)
            preds = preds.numpy()
        else:
            raise Exception("Param activation_fn should be 'sigmoid' or 'softmax'.")
        targets = targets.data.cpu().numpy()

        for b_id in range(bs):
            for c in range(1, self.num_classes):
                for m_id, m in enumerate(self.metric_list):
                    metric_array[c - 1, m_id] += self.compute_metric(preds[b_id] == c, targets[b_id] == c, m)
        return metric_array / bs


# following code modified from medpy(https://loli.github.io/medpy/)
# Note that the voxel spacing should be in numpy order (z, y, x)
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def assd(result, reference, voxelspacing=None, connectivity=1):
    assd = np.mean(
        (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)))
    return assd


def asd(result, reference, voxelspacing=None, connectivity=1):
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd


def dc(result, reference):
    r"""
    Dice coefficient
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def hd(result, reference, voxelspacing=None, connectivity=1):
    r"""
    Hausdorff Distance.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd
