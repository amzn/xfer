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
import numpy as np
import zipfile
import matplotlib.pyplot as plt

from PIL import Image
from mxnet.gluon.data.dataset import Dataset
import mxnet

from .data import MetaTaskDataContainer, TaskDataContainer
from .utils import list_dirs, list_files, random_affine_matrix, copy_dir, resize_imgs, download_url
from .config import DEFAULT_CONFIG_OMNIGLOT


class MetaTaskOmniglot(MetaTaskDataContainer):

    def __init__(self, root="./data", num_classes=None,
                 config=None, seed=1, context=None):

        """
        Onmiglot Data Container for Meta-Learning

        :param root: path containing the onmiglot dataset (if it does not exists, it
            download the data into this directory.
        :param num_classes: If not None, only those alphabet with at least num_classes
            are taken into consideration.
        :param config: If None, DEFAULT_CONFIG is loaded.
        :param seed: seed for random generator.
        """

        if config is None:
            self.config = DEFAULT_CONFIG_OMNIGLOT

        if context is None:
            context = mxnet.cpu()
        self.context = context

        self.url_dir = 'https://github.com/brendenlake/omniglot/raw/master/python/'
        self.urls = [
            os.path.join(self.url_dir, "images_background.zip"),
            os.path.join(self.url_dir, "images_evaluation.zip")
            ]

        self.root = root
        self.target_folder = "images_resized"

        self.download_dataset()

        self.seed = seed
        random.seed(self.seed)

        self.num_classes = num_classes

        self.alphabets = []
        self.alphabets_train = []
        self.alphabets_test = []
        self.alphabets_val = []

        self.target_path = os.path.join(self.root, self.target_folder)

        self._init_train_test_val_alphabets()

        hold = self.config["hold_out"]
        transform_image = self.config["transform_image"]
        transform_mxnet = self.config["transform_mxnet"]

        # Generate the training/test/val dataset.
        # Each dataset is a list of SubOmniglot objects (one per task)
        data_train = [TaskOmniglot(self.target_path, [a], num_classes, hold, transform_image=transform_image,
                                   transform_mxnet=transform_mxnet, context=context) for a in self.alphabets_train]
        data_test = [TaskOmniglot(self.target_path, [a], num_classes, hold, transform_image=transform_image,
                                  transform_mxnet=transform_mxnet, context=context) for a in self.alphabets_test]
        data_val = [TaskOmniglot(self.target_path, [a], num_classes, hold, transform_image=transform_image,
                                 transform_mxnet=transform_mxnet, context=context) for a in self.alphabets_val]

        super(MetaTaskOmniglot, self).__init__(data_train, data_test, data_val, context)

    def _init_train_test_val_alphabets(self):

        trs = self.config["num_tasks_train"]
        tes = self.config["num_tasks_test"]
        val = self.config["num_tasks_val"]

        # Filter out alphabet with less than num_classes, and shuffle
        self.alphabets = list(list_dirs(self.target_path))
        if self.num_classes:
            self.alphabets = []
            for a in self.alphabets:
                if len(list(list_dirs(os.path.join(self.target_path, a)))) >= self.num_classes:
                    self.alphabets.append(a)
            assert trs + tes < len(self.alphabets), 'cannot create test set'
        random.shuffle(self.alphabets)

        # init train/test/val alphabets
        self.alphabets_train = self.alphabets[:trs]
        self.alphabets_test = self.alphabets[trs:trs + tes]
        if val is not None:
            self.alphabets_val = self.alphabets[trs + tes:trs + tes + val]
        else:
            self.alphabets_val = self.alphabets[trs + tes:]

    def _copy_to_target(self, path_origin):
        out_path = os.path.abspath(os.path.join(self.root, self.target_folder))
        for alphabet in os.listdir(path_origin):
            copy_dir(os.path.join(path_origin, alphabet), os.path.join(out_path, alphabet))

    def _resize(self):
        resize_imgs(os.path.abspath(os.path.join(self.root, self.target_folder)))

    def download_dataset(self):
        """
        Helper function to download Onmiglot

        It download  both files, images_background.zip and images_evaluation, extract and extract
        them in self.target_path. It finally resizes all the image to 28X28 px.
        """
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        for url in self.urls:
            file_name = url.split(os.sep)[-1]
            out_file = os.path.join(self.root, file_name)

            if not os.path.exists(out_file):
                out_file = download_url(url, out_file)
                with zipfile.ZipFile(out_file, "r") as zip_file:
                    zip_file.extractall(os.path.dirname(out_file))

                dir_images = os.path.abspath(out_file).split(".")[-2]
                self._copy_to_target(dir_images)
                self._resize()

    def plot_sample(self, num_samples, root="./sample_onmiglot"):

        """Plot N images from each alphabet and store the images in root."""

        if not os.path.exists(root):
            os.makedirs(root)

        fig_train = self._plot(num_samples, [dd._train_dataset for dd in self.train_tasks],
                               "Training Samples for Training Tasks")
        fig_train.savefig(os.path.join(root, "sample_train_train_tasks.png"))
        del fig_train
        fig_test = self._plot(num_samples, [dd._train_dataset for dd in self.test_tasks],
                              "Training Samples for Test Tasks")
        fig_test.savefig(os.path.join(root, "sample_train_test_tasks.png"))
        del fig_test
        fig_val = self._plot(num_samples, [dd._train_dataset for dd in self.val_tasks],
                             "Training Samples for Validation Tasks")
        fig_val.savefig(os.path.join(root, "sample_train_val_tasks.png"))
        del fig_val
        fig_train = self._plot(num_samples, [dd._val_dataset for dd in self.train_tasks],
                               "Validation Samples for Training Tasks")
        fig_train.savefig(os.path.join(root, "sample_val_train_tasks.png"))
        del fig_train
        fig_test = self._plot(num_samples, [dd._val_dataset for dd in self.test_tasks],
                              "Validation Samples for Test Tasks")
        fig_test.savefig(os.path.join(root, "sample_val_test_tasks.png"))
        del fig_test
        fig_val = self._plot(num_samples, [dd._val_dataset for dd in self.val_tasks],
                             "Validation Samples for Validation Tasks")
        fig_val.savefig(os.path.join(root, "sample_val_val_tasks.png"))
        del fig_val

    def _plot(self, num_samples, data, title):

        """Helper function for plotting."""

        num_alphabets = len(data)
        fig, ax = plt.subplots(num_alphabets, num_samples, figsize=(2 * num_samples, 2 * num_alphabets))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        for mm in range(num_alphabets):
            ix_perm = np.random.permutation(len(data[mm]))
            for nn in range(num_samples):
                im, label = data[mm][ix_perm[nn]]
                ax[mm, nn].imshow(im.asnumpy())
                ax[mm, nn].set_title("Label: {} \n ({})".format(label, data[mm].items[nn][0].split("/")[-3][0:5]))
                ax[mm, nn].axis("off")
        fig.suptitle(title, size=18)
        return fig


class TaskOmniglot(TaskDataContainer):
    """
    Onmiglot Task Container

    Represent a single Onmiglot Task (can contain one or more alphabets)
    """

    def __init__(self, path, alphabets, num_classes=None, hold_out=None,
                 transform_image=None, transform_mxnet=None, seed=None, context=None):
        """

        :param path: Root path
        :param alphabets: Alphabet to be considered for the task
        :param num_classes: Number of classes/characters per alphabet
        :param hold_out: Number of images to hold out for validation
        :param transform_image: Applied to the PIL image (before loading it in an mx.ndarray)
        :param transform_mxnet: Applied to the mx.ndarray when loading a character
        :param seed: seed for the random generator
        """

        if context is None:
            context = mxnet.cpu()
        self.context = context

        self.path = path
        self.alphabets = alphabets
        self.num_classes = num_classes
        self.hold_out = hold_out
        self.transform_image = transform_image
        self.transform_mxnet = transform_mxnet
        self.target_transform = None
        self.seed = seed

        # Generate a list of alphabets and characters
        self._alphabets = [a for a in list_dirs(self.path) if a in self.alphabets]
        self._characters = sum([[os.path.join(a, c) for c in list_dirs(os.path.join(self.path, a))]
                                for a in self._alphabets], [])

        if seed:
            random.seed(seed)

        random.shuffle(self._characters)

        if self.num_classes:
            self._characters = self._characters[:num_classes]

        self._train_character_images = []
        self._val_character_images = []
        for idx, character in enumerate(self._characters):
            train_characters = []
            val_characters = []

            for img_count, image in enumerate(list_files(os.path.join(self.path, character), 'png')):
                if hold_out and img_count < hold_out:
                    val_characters.append((image, idx))
                else:
                    train_characters.append((image, idx))
            self._train_character_images.append(train_characters)
            self._val_character_images.append(val_characters)

        self._flat_train_character_images = sum(self._train_character_images, [])
        self._flat_val_character_images = sum(self._val_character_images, [])

        train_dataset = ImageListDataset(self._flat_train_character_images,
                                         flag=0, transform_image=transform_image,
                                         transform_mxnet=transform_mxnet, context=context)

        if len(self._flat_val_character_images) > 0:
            val_dataset = ImageListDataset(self._flat_val_character_images, flag=0,
                                           transform_image=transform_image,
                                           transform_mxnet=transform_mxnet, context=context)
        else:
            val_dataset = None

        super(TaskOmniglot, self).__init__(train_dataset, val_dataset)


class ImageListDataset(Dataset):

    def __init__(self, items, flag=1, transform_image=None, transform_mxnet=None, context=None):
        self.items = items
        self._flag = flag
        self._transform_image = transform_image
        self._transform_mxnet = transform_mxnet

        if context is None:
            context = mxnet.cpu()
        self.context = context

    def __getitem__(self, idx):
        img = Image.open(self.items[idx][0], mode='r').convert('L')
        if self._transform_image is not None:
            img = random_affine_matrix(img)
        img = mxnet.nd.array(img, ctx=self.context)
        label = self.items[idx][1]
        if self._transform_mxnet is not None:
            return self._transform_mxnet(img, label)
        return img, label

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':

    o = MetaTaskOmniglot()
    o.plot_sample(5)

    meta_batch_size = 2
    batch_size = 20
    train_tasks = o.train_tasks

    assert len(train_tasks) == 3

    for task in train_tasks:
        tr_iterator = task.get_train_iterator(batch_size)
        for data in tr_iterator:
            assert (data[0].shape == (batch_size, 28, 28))
            assert (data[1].shape == (batch_size, ))
            assert (data[1].asnumpy().dtype == np.int)
            break

        val_iterator = task.get_val_iterator(batch_size)
        for data in val_iterator:
            assert (data[0].shape == (batch_size, 28, 28))
            assert (data[1].shape == (batch_size, ))
            assert (data[1].asnumpy().dtype == np.int)
            break
