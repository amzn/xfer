# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================
import os
import random
import shutil
import numpy as np
import errno
import glob
from urllib.request import urlretrieve
from PIL import Image


def download_url(url, out_file):
    try:
        urlretrieve(url, out_file)
    except IOError:
        raise IOError()

    return out_file


def list_files(path, extension=None):
    if extension is None:
        list_files = (
            file for file in os.listdir(path)
            if os.path.isfile(os.path.join(path, file))
        )
    else:
        list_files = (
            file for file in os.listdir(path)
            if (
                os.path.isfile(os.path.join(path, file)) and
                (file.split(os.path.extsep)[-1] == extension)
            )
        )
    return [os.path.join(path, f) for f in list_files]


def list_dirs(path):
    return (dir for dir in os.listdir(path)
            if ((not os.path.isfile(os.path.join(path, dir))) and (dir[0] != ".")))


def copy_dir(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def resize_imgs(root):
    image_path = os.path.join(root, '*/*/')
    all_images = glob.glob(image_path + '*')
    for image_file in all_images:
        im = Image.open(image_file)
        im = im.resize((28, 28), resample=Image.LANCZOS)
        im.save(image_file)


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):

    matrix_center = np.array([
        [1, 0, center[0]],
        [0, 1, center[1]],
        [0, 0, 1]
    ])

    inv_matrix_scale_rotate_shear = (1 / (scale * np.cos(shear))) * np.array([
        [np.cos(angle + shear), np.sin(angle + shear), 0],
        [-np.sin(angle), np.cos(shear), 0],
        [0, 0, scale * np.cos(shear)]
    ])

    inv_matrix_center_dot_inv_matrix_trans = np.array([
        [1, 0, -(center[0] + translate[0])],
        [0, 1, -(center[1] + translate[1])],
        [0, 0, 1]
    ])

    inv_affine = matrix_center.dot(
        inv_matrix_scale_rotate_shear).dot(
        inv_matrix_center_dot_inv_matrix_trans
    )

    return inv_affine


def affine_transform(img, angle, translate, scale, shear):
    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    return img.transform(output_size, Image.AFFINE, matrix)


def random_affine_matrix(img, scale=(0.8, 1.2), rotation=(0, 2*np.pi), translation=(0.2, 0.2)):
    angle = np.random.uniform(low=rotation[0], high=rotation[1])
    tx = translation[0] * img.size[0]
    ty = translation[1] * img.size[1]
    translation = (np.round(random.uniform(-tx, tx)), np.round(random.uniform(-ty, ty)))
    scale = random.uniform(scale[0], scale[1])
    matrix = affine_transform(img, angle=angle, translate=translation, scale=scale, shear=0.0)
    return matrix
