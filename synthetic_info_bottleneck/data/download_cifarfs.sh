# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

wget https://www.dropbox.com/s/wuxb1wlahado3nq/cifar-fs-splits.zip?dl=0
mv cifar-fs-splits.zip?dl=0 cifar-fs-splits.zip
unzip cifar-fs-splits.zip
rm cifar-fs-splits.zip

python get_cifarfs.py
mv cifar-fs-splits/val1000* cifar-fs/

wget https://www.dropbox.com/s/g9ru5ac5tpupvg6/netFeatBest62.561.pth?dl=0
mv netFeatBest62.561.pth?dl=0 netFeatBest62.561.pth
mkdir ../ckpts
mkdir ../ckpts/CIFAR-FS
mv netFeatBest62.561.pth ../ckpts/CIFAR-FS/
