from __future__ import print_function, division

import os
import logging
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

IS_ON_SERVER = False if os.getcwd().startswith('/home/SENSETIME/yuanjing1') else True

axis_name2np_dim = {
    "x": 2,
    "y": 1,
    "z": 0,
}


# --------
# data normalization
def mean_std_norm(array):
    return (array - array.mean()) / (array.std() + 1e-10)


def rescale01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def window_rescale(arr, a_min=None, a_max=None):
    arr = np.clip(arr, a_min=a_min, a_max=a_max)
    return (arr - a_min) / (a_max - a_min)


def auto_window(arr):
    return window_rescale(arr, a_min=np.percentile(arr, 1), a_max=np.percentile(arr, 99))


# --------
# pad & crop
def get_pad_border(origin_shape, target_shape):
    assert len(origin_shape) == len(target_shape), 'Dimension mismatch.'
    borders = []
    for i in range(len(origin_shape)):
        tmp = target_shape[i] - origin_shape[i]
        borders.extend((tmp // 2, tmp - tmp // 2))
    return tuple(zip(borders[::2], borders[1::2]))


def pad_zyx_constant(nda, target_shape, pad_value=0, strict=False):
    assert nda.ndim == len(target_shape), 'Dimension mismatch.'
    if strict:
        assert np.all(np.array(target_shape) >= np.array(nda.shape)), 'Target shape must be larger than input shape.'
    else:
        target_shape = np.maximum(nda.shape, target_shape)
    borders = get_pad_border(nda.shape, target_shape)
    nda = np.pad(nda, borders, mode='constant', constant_values=pad_value)
    return nda


def center_crop_zyx(nda, target_shape):
    starts = np.asarray((np.asarray(nda.shape) - np.asarray(target_shape)) // 2)
    slice_fn = tuple(map(slice, starts, np.asarray(starts) + np.asarray(target_shape)))
    return nda[slice_fn]


def constant_pad_crop(nda, target_shape, pad_value=0, strict=False):
    assert nda.ndim == len(target_shape), 'Dimension mismatch.'
    nda = pad_zyx_constant(nda, target_shape, pad_value, strict)
    return center_crop_zyx(nda, target_shape)


# --------
# logger
def log_init(log_dir):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    # handle for txt file
    f_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(formatter)
    # handle for screen
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tb'))
    return logger, writer


# --------
def get_bbox(np_lbl):
    lbl_indices = np.nonzero(np_lbl)
    bbox = np.array([[i.min(), i.max()] for i in lbl_indices])
    return bbox


def hist_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)


def make_one_hot(labels, num_classes):
    r"""
    Convert int labels to one-hot encoding labels
    :param labels: N x H x W or N x D x H x W shape torch.LongTensor
    :param num_classes: class number control output channel C
    :return: N x C x H x W or N x C x D x H x W
    """
    labels = torch.unsqueeze(labels, dim=1)
    one_hot_shape = list(labels.size())
    one_hot_shape[1] = num_classes
    one_hot = torch.zeros(one_hot_shape).long().to(labels.device)
    return one_hot.scatter_(dim=1, index=labels, value=1)


def save_csv(cfg, file_list, file_name):
    df = pd.DataFrame(np.array(file_list), columns=['file names'])
    df.to_csv(os.path.join(cfg.save_dir, file_name), index=False)


def train_val_test_split(data_list, train_control, val_control, data_stratify=None, random_seed=None):

    def _split_one_group(data_list, train_num, val_num, random_seed=None):
        data_length = len(data_list)
        if random_seed is not None:
            np.random.seed(random_seed)
        ids_seq = np.random.permutation(data_length)
        return data_list[ids_seq[0:train_num]], \
               data_list[ids_seq[train_num:train_num + val_num]], \
               data_list[ids_seq[train_num + val_num:]]

    data_length = len(data_list)
    if type(data_list) != np.ndarray:
        data_list = np.array(data_list)

    train_num = int(math.ceil(train_control * data_length)) if isinstance(train_control, float) else train_control
    val_num = int(math.floor(val_control * data_length)) if isinstance(val_control, float) else val_control

    if data_stratify is None:
        train_list, val_list, test_list = _split_one_group(data_list, train_num, val_num, random_seed)
    else:
        if type(data_stratify) != np.ndarray:
            data_stratify = np.array(data_stratify)
        classes, classes_counts = np.unique(data_stratify, return_counts=True)
        train_ratio = train_control if isinstance(train_control, float) else train_num / data_length
        val_ratio = val_control if isinstance(val_control, float) else val_num / data_length

        train_nums = []
        val_nums = []

        for i in range(len(classes)):
            if i != len(classes) - 1:
                train_nums.append(int(math.ceil(train_ratio * classes_counts[i])))
                val_nums.append(int(math.floor(val_ratio * classes_counts[i])))
            else:
                train_nums.append(train_num - np.asarray(train_nums).sum())
                val_nums.append(val_num - np.asarray(val_nums).sum())

        train_list = np.array([])
        val_list = np.array([])
        test_list = np.array([])
        for i, (t, v) in enumerate(zip(train_nums, val_nums)):

            tmp_train_list, tmp_val_list, tmp_test_list = \
                _split_one_group(data_list[data_stratify == classes[i]], t, v,
                                 random_seed + i * 10 if random_seed is not None else random_seed)
            train_list = np.concatenate((train_list, tmp_train_list))
            val_list = np.concatenate((val_list, tmp_val_list))
            test_list = np.concatenate((test_list, tmp_test_list))

    return train_list.tolist(), val_list.tolist(), test_list.tolist()
