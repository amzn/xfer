## Similarity of Neural Networks with Gradients 
--------------------------------------------------------------------------------

## Introduction  

This folder contains code for comparing trained neural networks using both feature and gradient information. The implementation relies on the following three files:

*sketched_kernels.py* computes the sketched kernel matrices of individual residual blocks based on a pretrained ImageNet model and a given dataset.

*sim_indices.py* computes the similarity scores between two residual blocks.

*utils.py* provides two helper functions, including *load_model* for loading an ImageNet model and *load_dataset* for creating a dataloader object.

## Requirements
```
python >= 3.5
torch >= 1.0
torchvision
numpy
```

## Example
Generate our proposed kernel matrices for individual residual blocks
given a pretrained ImageNet model and a dataset (cifar10 below)
```
CUDA_VISIBLE_DEVICES=0 python -u cwt_kernel_mat.py \
        --datapath data/ \
        --modelname resnet18 \
        --pretrained \
        --seed 1111 \
        --task cifar10 \
        --split test \
        --bsize 256 \
        --num-buckets-sketching 128 \
        --num-buckets-per-sample 1
```

Given sketched kernel matrices calculated on one dataset (cifar10 below),
compute a heatmap in which each entry is the similarity score between two residual blocks
```
python -u compute_similarity.py \
        --loadpath sketched_kernel_mat/ \
        --filename1 resnet18_test_cifar10_1111.npy \
        --simindex cka
```

Given sketched kernel matrices calculated on two datasets (cifar10 and cifar100 below),
compute a heatmap in which each entry is the similarity score between two residual blocks
```
python -u compute_similarity.py \
        --loadpath sketched_kernel_mat/ \
        --filename1 resnet18_test_cifar10_1111.npy \
        --filename2 resnet18_test_cifar100_1111.npy \
        --simindex cka
```

## Authors  
Shuai Tang
