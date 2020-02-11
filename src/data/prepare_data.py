# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from src.utils.itk_tools import itk_change_spacing
from src.utils.miscs import auto_window, constant_pad_crop, get_bbox, axis_name2np_dim, train_val_test_split

exclude_list = [
    "b01_c02_p01.nii.gz",  # 矫形金属
    "b01_c04_p01.nii.gz",  # 矫形金属
    "b01_c05_p04.nii.gz",  # 图像模糊
    "b01_c06_p05.nii.gz",  # CBCT缺层
    "b04_c02_p04.nii.gz",  # CT下颌骨分割不完整
]

info_csv_path = "/data/databak/mouth/九院数据/info/cbct.csv"


def prepare_npy(cbct_dir, ct_dir, cbct_seg_dir, save_dir, choose_axis, dst_sp, dst_shape,
                mask_dilation=False):
    f_list = sorted(os.listdir(cbct_dir))
    for f in exclude_list:
        if f in f_list:
            f_list.remove(f)
    for d in ["cbct", "cbct_seg", "ct"]:
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)
    del f, d

    side_num = int((dst_shape[0] - 1) // 2)
    pad_width = [(0, 0) for _ in range(3)]
    pad_width[0] = (side_num, side_num)
    axis = axis_name2np_dim[choose_axis]

    def move_pad_crop(np_img, pad_value, mode="reflect"):
        np_img = np.moveaxis(np_img, axis, 0)
        if mode == "constant":
            np_img = np.pad(np_img, pad_width, mode, constant_values=pad_value)
        else:
            np_img = np.pad(np_img, pad_width, mode)
        np_img = constant_pad_crop(np_img, (np_img.shape[0], dst_shape[1], dst_shape[2]), pad_value)
        return np_img

    # info length csv
    src_info_csv = pd.read_csv(info_csv_path)
    info_csv = src_info_csv[src_info_csv["file name"].isin(f_list)]
    info_csv = pd.DataFrame.reset_index(info_csv, drop=True)
    info_csv["file name"] = info_csv["file name"].str.replace(".nii.gz", ".npy")
    info_csv = info_csv.drop(columns=["min", "max"])
    info_csv["length"] = 0

    for f in f_list:
        cbct_path = os.path.join(cbct_dir, f)
        ct_path = os.path.join(ct_dir, f)
        cbct_seg_path = os.path.join(cbct_seg_dir, f)
        # ct_seg_path = os.path.join(ct_seg_dir, f)
        if os.path.exists(cbct_seg_path) and os.path.exists(ct_path):  # and os.path.exists(ct_seg_path):

            # cbct and cbct seg
            itk_cbct = sitk.ReadImage(cbct_path)
            itk_cbct = itk_change_spacing(itk_cbct, dst_sp, "Linear")
            np_cbct = sitk.GetArrayFromImage(itk_cbct)

            itk_cbct_seg = sitk.ReadImage(cbct_seg_path)
            itk_cbct_seg = itk_change_spacing(itk_cbct_seg, dst_sp, "NearestNeighbor")
            np_cbct_seg = sitk.GetArrayFromImage(itk_cbct_seg)

            # ct
            itk_ct = sitk.ReadImage(ct_path)
            itk_ct = itk_change_spacing(itk_ct, dst_sp, "Linear")
            np_ct = sitk.GetArrayFromImage(itk_ct)

            # itk_ct_seg = sitk.ReadImage(ct_seg_path)
            # itk_ct_seg = itk_change_spacing(itk_ct_seg, dst_sp, "NearestNeighbor")
            # np_ct_seg = sitk.GetArrayFromImage(itk_ct_seg)

            # set nonsense position seg to 0
            np_cbct_seg[np.logical_and(np_cbct == np.amin(np_cbct), np_ct <= 0)] = 0
            # np_ct_seg[np.logical_and(np_cbct == np.amin(np_cbct), np_ct <= 0)] = 0

            # dilation seg
            if mask_dilation:
                struct1 = ndimage.generate_binary_structure(3, 1)
                tmp = np.array(np_cbct_seg == 1).astype(np.uint8)
                tmp = ndimage.binary_dilation(tmp, struct1, iterations=mask_dilation)
                np_cbct_seg[tmp != 0] = 1
                # tmp = np.array(np_ct_seg == 1).astype(np.uint8)
                # tmp = ndimage.binary_dilation(tmp, struct1, iterations=mask_dilation)
                # np_ct_seg[tmp != 0] = 1

            np_cbct_seg = np.asarray(np_cbct_seg > 0).astype(np.uint8)
            # np_ct_seg = np.asarray(np_ct_seg > 0).astype(np.uint8)

            # value normalize, the percentile clip need further add otsu to ignore air
            np_cbct = np.clip(np_cbct, np.percentile(np_cbct, 1), np.percentile(np_cbct, 99.9))
            np_cbct = (np_cbct - np.amin(np_cbct)) / (np.amax(np_cbct) - np.amin(np_cbct))
            # np_cbct = np_cbct / 1000
            np_ct[np_ct < -1000] = -1000
            np_ct += 1000
            np_ct = np_ct / 1000

            # move axis and pad crop
            np_cbct = move_pad_crop(np_cbct, 0)
            np_cbct_seg = move_pad_crop(np_cbct_seg, 0, "constant")
            np_ct = move_pad_crop(np_ct, 0)
            # np_ct_seg = move_pad_crop(np_ct_seg, 0)

            # get mask length in axis 0
            cbct_bbox = get_bbox(np_cbct_seg)
            info_csv.loc[info_csv["file name"] == f.replace(".nii.gz", ".npy"), "length"] = cbct_bbox[0, 1] - cbct_bbox[0, 0] + 1

            # save
            np.save(os.path.join(save_dir, "cbct", f.replace(".nii.gz", ".npy")), np_cbct.astype(np.float32))
            np.save(os.path.join(save_dir, "cbct_seg", f.replace(".nii.gz", ".npy")), np_cbct_seg.astype(np.uint8))
            np.save(os.path.join(save_dir, "ct", f.replace(".nii.gz", ".npy")), np_ct.astype(np.float32))
            # np.save(os.path.join(save_dir, "ct_seg", f.replace(".nii.gz", ".npy")), np_ct_seg.astype(np.uint8))

    info_csv.to_csv(os.path.join(save_dir, "info.csv"), index=False)


if __name__ == "__main__":
    def save_csv(src_csv, f_list, save_path):
        new_csv = src_csv[src_csv["file name"].isin(f_list)]
        new_csv = pd.DataFrame.reset_index(new_csv, drop=True)
        new_csv.to_csv(save_path, index=False)

    data_dir = ""  # data directory
    save_dir = os.path.join("",  # save directory
                            "relative_zyx_" + time.strftime(str("%y%m%d"), time.localtime()))
    os.makedirs(save_dir)
    prepare_npy(os.path.join(data_dir, "cbct"),
                os.path.join(data_dir, "ct"),
                os.path.join(data_dir, "cbct_seg_tuned"),
                save_dir,
                "z",
                (0.5, 0.5, 1.0), (11, 384, 384), False)
    info_csv = pd.read_csv(os.path.join(save_dir, "info.csv"))
    data_list = os.listdir(os.path.join(save_dir, "cbct"))
    data_stratify = np.zeros(len(data_list))
    for idx, f in enumerate(data_list):
        m = info_csv[info_csv["file name"] == f]["Manufacturer"].to_list()[0]
        if m.startswith("SOREDEX"):
            pass
        elif m.startswith("Carestream Health"):
            data_stratify[idx] = 1
        else:
            data_stratify[idx] = 2

    train_list, val_list, test_list = train_val_test_split(data_list, 0.8, 0.1, data_stratify, 30)
    save_csv(info_csv, train_list, os.path.join(save_dir, "train.csv"))
    save_csv(info_csv, val_list, os.path.join(save_dir, "val.csv"))
    save_csv(info_csv, test_list, os.path.join(save_dir, "test.csv"))