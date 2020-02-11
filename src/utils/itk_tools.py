from __future__ import division

import numpy as np
import SimpleITK as sitk


def itk_resize(src_itk, output_size, interpolation_method='Linear'):
    """

    :param src_itk:
    :param output_size: W * H * D
    :param interpolation_method: Linear or NearestNeighbor
    :return:
    """
    src_size = src_itk.GetSize()
    src_spacing = src_itk.GetSpacing()
    output_scale = tuple(np.array(output_size).astype(np.float) / np.array(src_size))
    output_spacing = tuple(np.array(src_spacing).astype(np.float) / np.array(output_scale))

    re_sampler = sitk.ResampleImageFilter()
    re_sampler.SetReferenceImage(src_itk)
    re_sampler.SetSize(output_size)
    re_sampler.SetOutputSpacing(output_spacing)
    re_sampler.SetInterpolator(eval('sitk.sitk' + interpolation_method))
    return re_sampler.Execute(src_itk)


def resize2ref(src_itk, ref_itk, interpolate_method='NearestNeighbor'):
    re_sampler = sitk.ResampleImageFilter()
    re_sampler.SetReferenceImage(src_itk)
    re_sampler.SetSize(ref_itk.GetSize())
    re_sampler.SetOutputSpacing(ref_itk.GetSpacing())
    re_sampler.SetInterpolator(eval('sitk.sitk' + interpolate_method))
    return re_sampler.Execute(src_itk)


def map2origin(src_itk, ref_itk, interpolate_method='NearestNeighbor'):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_itk)
    resampler.SetInterpolator(eval('sitk.sitk' + interpolate_method))
    new_itk = resampler.Execute(src_itk)
    return sitk.Cast(new_itk, src_itk.GetPixelID())


def itk_change_spacing(src_itk, output_spacing, interpolate_method='Linear'):
    src_size = src_itk.GetSize()
    src_spacing = src_itk.GetSpacing()

    re_sample_scale = tuple(np.array(src_spacing) / np.array(output_spacing).astype(np.float))
    re_sample_size = tuple(np.array(src_size).astype(np.float) * np.array(re_sample_scale))

    re_sample_size = [int(round(x)) for x in re_sample_size]
    output_spacing = tuple((np.array(src_size) / np.array(re_sample_size)) * np.array(src_spacing))

    re_sampler = sitk.ResampleImageFilter()
    re_sampler.SetOutputPixelType(src_itk.GetPixelID())
    re_sampler.SetReferenceImage(src_itk)
    re_sampler.SetSize(re_sample_size)
    re_sampler.SetOutputSpacing(output_spacing)
    re_sampler.SetInterpolator(eval('sitk.sitk' + interpolate_method))  # sitk.sitkNearestNeighbor
    return re_sampler.Execute(src_itk)


def itk_flip_axis(src_itk, axis=2):
    # origin_direction = src_itk.GetDirection()
    re_sampler = sitk.FlipImageFilter()
    flip_axes = [False, False, False]
    if isinstance(axis, int):
        flip_axes[axis] = True
    elif isinstance(axis, (tuple, list)):
        for i in axis:
            flip_axes[i] = True
    re_sampler.SetFlipAxes(flip_axes)
    itk_img = re_sampler.Execute(src_itk)
    # itk_img.SetDirection(origin_direction)
    return itk_img


def eul2quat(ax, ay, az, atol=1e-8):
    """
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    """
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r = np.zeros((3, 3))
    r[0, 0] = cz*cy
    r[0, 1] = cz*sy*sx - sz*cx
    r[0, 2] = cz*sy*cx+sz*sx

    r[1, 0] = sz*cy
    r[1, 1] = sz*sy*sx + cz*cx
    r[1, 2] = sz*sy*cx - cz*sx

    r[2, 0] = -sy
    r[2, 1] = cy*sx
    r[2, 2] = cy*cx

    # Compute quaternion:
    qs = 0.5*np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs, 0.0, atol):
        i = np.argmax([r[0, 0], r[1, 1], r[2, 2]])
        j = (i+1) % 3
        k = (j+1) % 3
        w = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i, j] + r[j, i])/(2*w)
        qv[k] = (r[i, k] + r[k, i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2, 1] - r[1, 2])/denom
        qv[1] = (r[0, 2] - r[2, 0])/denom
        qv[2] = (r[1, 0] - r[0, 1])/denom
    return qv


# The parameters are scale (float), rotation angle (radians), x translation, y translation
def rotate_translate_scale2d(img, parameters, interpolator='Linear', default_intensity_value=0):
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    aug_transform = sitk.Similarity2DTransform()
    aug_transform.SetCenter(img_center)
    aug_transform.SetParameters(parameters)
    aug_transform = aug_transform.GetInverse()
    return sitk.Resample(img, aug_transform, eval('sitk.sitk' + interpolator), default_intensity_value)


# The parameters are thetaX, thetaY, thetaZ, tx, ty, tz, scale
def rotate_translate_scale3d(img, parameters, interpolator='Linear', default_intensity_value=0):
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    aug_transform = sitk.Similarity3DTransform()
    aug_transform.SetCenter(img_center)
    aug_transform.SetParameters(list(eul2quat(*parameters[:3])) + parameters[3:])
    aug_transform = aug_transform.GetInverse()
    return sitk.Resample(img, aug_transform, eval('sitk.sitk' + interpolator), default_intensity_value)
