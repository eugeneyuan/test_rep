import os
import random
import numpy as np
from scipy import ndimage

import torch
from torch.utils.data import Dataset

from src.utils.miscs import get_bbox, constant_pad_crop, axis_name2np_dim


class NPYSlice(Dataset):

    def __init__(self, cfg, df, augmentations=None, phase="train"):
        f_list = df["file name"].to_list()
        length_list = np.asarray(df["length"].to_list(), dtype=np.int)
        cumsum_length = np.cumsum(length_list, dtype=np.int)

        self.f_list = f_list
        self.cfg = cfg
        self.phase = phase
        self._cumsum_length = cumsum_length

        self._input_shape = cfg.DATA.INPUT_SHAPE
        self._side_num = int((self._input_shape[0] - 1) // 2)
        if self._side_num != 0:
            self._slice_indices_template = np.arange(-self._side_num, self._side_num + 1)
        else:
            self._slice_indices_template = 0
        self._length = int(cumsum_length[-1])

        self._augs = augmentations

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # choose file
        f_id = np.searchsorted(self._cumsum_length, index, 'right')
        np_src = np.load(os.path.join(self.cfg.DATA.SRC_DIR, self.f_list[f_id])).astype(np.float32)
        np_dst = np.load(os.path.join(self.cfg.DATA.DST_DIR, self.f_list[f_id])).astype(np.float32)
        np_mask = np.load(os.path.join(self.cfg.DATA.MASK_DIR, self.f_list[f_id])).astype(np.float32)

        if self.cfg.DATA.MODIFY_SRC and self.phase == "train":
            if random.random() < 0.3:
                np_src = self.modify_src(np_src,
                                         self.cfg.DATA.MODIFY_A_RANGE, self.cfg.DATA.MODIFY_B_RANGE)
                np_src = (np_src - np.amin(np_src)) / (np.amax(np_src) - np.amin(np_src))

        # 3d augmentation
        if self._augs:
            np_src, np_dst, dst_seg = self._augs(
                (np_src, np_dst, np_mask), ("image", "image", "mask"))

        # choose slices
        slice_id = index if f_id == 0 else index - self._cumsum_length[f_id - 1]
        slice_id += get_bbox(np_mask)[0, 0]
        slices_indices = (self._slice_indices_template + slice_id).astype(np.int)
        # select
        np_src = np.take(np_src, slices_indices, axis=0).astype(np.float32)
        np_dst = np.take(np_dst, slice_id, axis=0).astype(np.float32)
        np_mask = np.take(np_mask, slice_id, axis=0).astype(np.uint8)
        # ensure shape
        if self._side_num == 0:
            np_src = np_src[np.newaxis]
        # np_src = constant_pad_crop(np_src, self.cfg.DATA.INPUT_SHAPE, pad_value=np.amin(np_src))
        # np_dst = constant_pad_crop(np_dst, self.cfg.DATA.INPUT_SHAPE[1:], pad_value=np.amin(np_dst))
        # np_mask = constant_pad_crop(np_mask, self.cfg.DATA.INPUT_SHAPE[1:], pad_value=0)
        return torch.from_numpy(np_src).float(), torch.from_numpy(np_dst[np.newaxis]).float(), \
               torch.from_numpy(np_mask[np.newaxis]).float()

    @staticmethod
    def modify_src(np_src, a_range, b_range):
        def _get_tmp(out_shape, out_zoom, value_range):
            tmp = np.random.rand(*out_shape).astype(np.float32) * value_range - (value_range / 2)
            tmp_min = np.amin(tmp)
            tmp_max = np.amax(tmp)
            tmp = ndimage.gaussian_filter(tmp, (3, 3, 3))
            tmp = ndimage.zoom(tmp, out_zoom, order=1)
            tmp = (tmp - np.amin(tmp)) / (np.amax(tmp) - np.amin(tmp) + 1e-9)
            tmp = tmp * (tmp_max - tmp_min) + tmp_min
            return tmp

        tmp_shape = np.maximum(np.asarray(np_src.shape) // 20, 10)
        zoom_rate = (np.asarray(np_src.shape) + 1e-9) / tmp_shape
        tmp_a = _get_tmp(tmp_shape, zoom_rate, a_range) + 1
        tmp_b = _get_tmp(tmp_shape, zoom_rate, b_range)
        tmp_mask = np_src == np.amin(np_src)
        tmp_a[tmp_mask] = 1
        tmp_b[tmp_mask] = 0
        return np_src * tmp_a + tmp_b


class NPYSliceEval(Dataset):

    def __init__(self, cfg, file_name, length):
        self.cfg = cfg
        self.f_name = file_name
        self._length = length

        self._input_shape = cfg.DATA.INPUT_SHAPE
        self._side_num = int((self._input_shape[0] - 1) // 2)
        if self._side_num != 0:
            self._slice_indices_template = np.arange(-self._side_num, self._side_num + 1)
        else:
            self._slice_indices_template = 0

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        np_src = np.load(os.path.join(self.cfg.DATA.SRC_DIR, self.f_name)).astype(np.float32)
        np_dst = np.load(os.path.join(self.cfg.DATA.DST_DIR, self.f_name)).astype(np.float32)
        np_mask = np.load(os.path.join(self.cfg.DATA.MASK_DIR, self.f_name)).astype(np.float32)

        # choose slices
        slice_id = index + get_bbox(np_mask)[0, 0]
        slices_indices = (self._slice_indices_template + slice_id).astype(np.int)
        # select
        np_src = np.take(np_src, slices_indices, axis=0).astype(np.float32)
        np_dst = np.take(np_dst, slice_id, axis=0).astype(np.float32)
        np_mask = np.take(np_mask, slice_id, axis=0).astype(np.uint8)
        # ensure shape
        if self._side_num == 0:
            np_src = np_src[np.newaxis]
        np_src = constant_pad_crop(np_src, self.cfg.DATA.INPUT_SHAPE, pad_value=np.amin(np_src))
        np_dst = constant_pad_crop(np_dst, self.cfg.DATA.INPUT_SHAPE[1:], pad_value=np.amin(np_dst))
        np_mask = constant_pad_crop(np_mask, self.cfg.DATA.INPUT_SHAPE[1:], pad_value=0)
        return torch.from_numpy(np_src).float(), torch.from_numpy(np_dst[np.newaxis]).float(), \
               torch.from_numpy(np_mask[np.newaxis]).float()
