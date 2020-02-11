import os
import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.itk_tools import itk_change_spacing, map2origin
from src.utils.miscs import constant_pad_crop, axis_name2np_dim


class Slice(Dataset):

    def __init__(self, img, side_num):
        self.img = img
        self._side_num = side_num
        if self._side_num != 0:
            self._slice_indices_template = np.arange(-self._side_num, self._side_num + 1)
        else:
            self._slice_indices_template = 0
        self._length = img.shape[0] - 2*side_num

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        slices_indices = self._slice_indices_template + self._side_num + index
        return torch.from_numpy(np.take(self.img, slices_indices, axis=0).astype(np.float32)).float()


def inference(net, itk_img, axis, dst_sp, dst_shape, batch_size, device, num_workers=4):
    resample_itk = itk_change_spacing(itk_img, dst_sp, "Linear")
    np_img = sitk.GetArrayFromImage(resample_itk)
    side_num = int((dst_shape[0] - 1) // 2)
    pad_width = [(0, 0) for _ in range(3)]
    pad_width[0] = (side_num, side_num)
    axis = axis_name2np_dim[axis]
    np_img = np.clip(np_img, np.percentile(np_img, 1), np.percentile(np_img, 99.9))
    np_img = (np_img - np.amin(np_img)) / (np.amax(np_img) - np.amin(np_img))
    np_img = np.moveaxis(np_img, axis, 0)
    resample_shape = np_img.shape
    np_img = np.pad(np_img, pad_width, "edge")
    np_img = constant_pad_crop(np_img, (np_img.shape[0], dst_shape[1], dst_shape[2]), 0)

    data = Slice(np_img, side_num)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_out = []
    net.eval()
    with torch.no_grad():
        for test_iter, imgs in enumerate(data_loader):
            imgs = imgs.to(device)
            out = net(imgs)
            test_out.append(out.squeeze(1).cpu().numpy())

    test_out = np.concatenate(test_out, axis=0)

    test_out = np.moveaxis(test_out, 0, axis)
    test_out = constant_pad_crop(test_out, resample_shape, 0)
    test_out = test_out * 1000
    test_out -= 1000
    itk_out = sitk.GetImageFromArray(test_out, isVector=False)
    itk_out.CopyInformation(resample_itk)
    itk_out = sitk.Cast(itk_out, sitk.sitkFloat32)
    itk_out = map2origin(itk_out, itk_img, "Linear")
    return itk_out


if __name__ == "__main__":
    from src.archs.net2d.simple_unet import UNet

    net_settings = {
        'encoder': [[64, ], ["max", 128, 128], ["max", 256, 256], ["max", 512, 512], ["max", 512, 512]],
        'mid': ["max", 512, 512],
        'decoder': [[512, 512], [512, 512], [256, 256], [128, 128], [64, ], ]
    }

    input_shape = (3, 384, 384)
    input_sp = (0.5, 0.5, 1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    net = UNet(input_shape[0], 1, net_settings, use_norm=True,
               norm_type="bn", act_type="relu")
    net = net.to(device)
    net.load_state_dict(torch.load("./logs/unet25d_z_e2e_ms_191226-223118/best.pth"))
    net.eval()

    itk_img = sitk.ReadImage(
        "/data/databak/mouth/九院数据/registered/cbct/b01_c01_p02.nii.gz")
    itk_out = inference(net, itk_img, "z", input_sp, input_shape, 8, device)
    sitk.WriteImage(itk_out, "b01_c01_p02.nii.gz")
