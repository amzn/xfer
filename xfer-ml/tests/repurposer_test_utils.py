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
import mxnet as mx
from unittest.mock import Mock
import os
import numpy as np
import zipfile
import random
import glob

MXNET_MODEL_ZOO_PATH = 'http://data.mxnet.io/models/imagenet/'


class RepurposerTestUtils:
    ERROR_INCORRECT_INPUT = 'Test case assumes incorrect input'
    VALIDATE_REPURPOSE_METHOD_NAME = '_validate_before_repurpose'
    VALIDATE_PREDICT_METHOD_NAME = '_validate_before_predict'

    LAYER_FC1 = 'fc1'
    LAYER_RELU = 'relu1'
    LAYER_FC2 = 'fc2'
    LAYER_SOFTMAX = 'softmax'
    ALL_LAYERS = [LAYER_FC1, LAYER_RELU, LAYER_FC2, LAYER_SOFTMAX]

    META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS = 'xfer.meta_model_repurposer.ModelHandler'
    MNIST_MODEL_PATH_PREFIX = 'tests/data/test_mnist_model'

    @staticmethod
    def create_mxnet_module():
        # Define an mxnet Module with 2 layers
        data = mx.sym.Variable('data')
        fc1 = mx.sym.FullyConnected(data, name=RepurposerTestUtils.LAYER_FC1, num_hidden=64)
        relu1 = mx.sym.Activation(fc1, name=RepurposerTestUtils.LAYER_RELU, act_type="relu")
        fc2 = mx.sym.FullyConnected(relu1, name=RepurposerTestUtils.LAYER_FC2, num_hidden=5)
        out = mx.sym.SoftmaxOutput(fc2, name=RepurposerTestUtils.LAYER_SOFTMAX)
        return mx.mod.Module(out)

    @staticmethod
    def get_mock_model_handler_object():
        mock_model_handler = Mock()
        mock_model_handler.layer_names = RepurposerTestUtils.ALL_LAYERS
        return mock_model_handler

    @staticmethod
    def get_image_iterator():
        image_list = [[0, 'accordion/image_0001.jpg'], [0, 'accordion/image_0002.jpg'], [1, 'ant/image_0001.jpg'],
                      [1, 'ant/image_0002.jpg'], [2, 'anchor/image_0001.jpg'], [2, 'anchor/image_0002.jpg']]
        return mx.image.ImageIter(2, (3, 224, 224), imglist=image_list, path_root='tests/data/test_images',
                                  label_name='softmax_label')

    @staticmethod
    def _assert_common_attributes_equal(repurposer1, repurposer2):
        assert repurposer1.__dict__.keys() == repurposer2.__dict__.keys()
        assert repurposer1._save_source_model_default == repurposer2._save_source_model_default
        RepurposerTestUtils.assert_provide_equal(repurposer1.provide_data, repurposer2.provide_data)
        RepurposerTestUtils.assert_provide_equal(repurposer1.provide_label, repurposer2.provide_label)
        assert repurposer1.get_params() == repurposer2.get_params()

    @staticmethod
    def assert_provide_equal(provide1, provide2):
        if provide1 is None:
            assert provide2 is None
            return
        assert len(provide1) == len(provide2)
        assert provide1[0][0] == provide2[0][0]
        assert len(provide1[0][1]) == len(provide2[0][1])

    @staticmethod
    def _remove_files_with_prefix(prefix):
        for filename in os.listdir('.'):
            if filename.startswith(prefix):
                os.remove(filename)

    @staticmethod
    def download_vgg19():
        # Download vgg19 (trained on imagenet)
        [mx.test_utils.download(MXNET_MODEL_ZOO_PATH+'vgg/vgg19-0000.params'),
         mx.test_utils.download(MXNET_MODEL_ZOO_PATH+'vgg/vgg19-symbol.json')]

    @staticmethod
    def download_squeezenet():
        # Download squeezenet (trained on imagenet)
        [mx.test_utils.download(MXNET_MODEL_ZOO_PATH+'squeezenet/squeezenet_v1.1-0000.params'),
         mx.test_utils.download(MXNET_MODEL_ZOO_PATH+'squeezenet/squeezenet_v1.1-symbol.json')]

    @staticmethod
    def download_resnet():
        # Download reset (trained on imagenet)
        [mx.test_utils.download(MXNET_MODEL_ZOO_PATH+'resnet/101-layers/resnet-101-0000.params'),
         mx.test_utils.download(MXNET_MODEL_ZOO_PATH+'resnet/101-layers/resnet-101-symbol.json')]

    @staticmethod
    def unzip_mnist_sample():
        zip_ref = zipfile.ZipFile('tests/data/mnist_sample.zip', 'r')
        zip_ref.extractall('.')
        zip_ref.close()

    @staticmethod
    def create_img_iter(data_dir, batch_size, label_name='softmax_label'):
        # assert dir exists
        if not os.path.isdir(data_dir):
            raise ValueError('Directory not found: {}'.format(data_dir))
        # get class names
        classes = [x.split('/')[-1] for x in glob.glob(data_dir+'/*')]
        classes.sort()
        fnames = []
        labels = []
        for c in classes:
            # get all the image filenames and labels
            images = glob.glob(data_dir+'/'+c+'/*')
            images.sort()
            fnames += images
            labels += [c]*len(images)
        # create imglist for ImageIter
        imglist = []
        for label, filename in zip(labels, fnames):
            imglist.append([int(label), filename])

        random.shuffle(imglist)
        # make iterators
        iterator = mx.image.ImageIter(batch_size, (3, 224, 224), imglist=imglist, label_name=label_name, path_root='')
        return iterator

    @staticmethod
    def get_labels(iterator):
        iterator.reset()
        labels = []
        while True:
            try:
                labels = labels + iterator.next().label[0].asnumpy().astype(int).tolist()
            except StopIteration:
                break
        return labels

    @staticmethod
    def assert_feature_indices_equal(expected_feature_indices, actual_feature_indices):
        if not type(expected_feature_indices) == type(actual_feature_indices):
            raise AssertionError("Incorrect feature_indices type: {}. Expected: {}"
                                 .format(type(actual_feature_indices), type(expected_feature_indices)))

        if not expected_feature_indices.keys() == actual_feature_indices.keys():
            raise AssertionError("Incorrect keys in feature_indices: {}. Expected: {}"
                                 .format(actual_feature_indices.keys(), expected_feature_indices.keys()))

        for key in expected_feature_indices:
            if not np.array_equal(expected_feature_indices[key], actual_feature_indices[key]):
                raise AssertionError("Incorrect values in feature_indices dictionary")

    @staticmethod
    def create_mnist_test_iterator():
        # Create data iterator for mnist test images
        return mx.io.MNISTIter(image='tests/data/t10k-images-idx3-ubyte', label='tests/data/t10k-labels-idx1-ubyte')
