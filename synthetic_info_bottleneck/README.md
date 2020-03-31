# \[ICLR 2020\] Synthetic information bottleneck for transductive meta-learning
This repo contains the implementation of the *synthetic information bottleneck* algorithm for few-shot classification on Mini-ImageNet,
which is used in our ICLR 2020 paper 
[Empirical Bayes Transductive Meta-Learning with Synthetic Gradients](https://openreview.net/forum?id=Hkg-xgrYvH).

If our code is helpful for your research, please consider citing: 
``` Bash
@inproceedings{
    Hu2020Empirical,
    title={Empirical Bayes Transductive Meta-Learning with Synthetic Gradients},
    author={Shell Xu Hu and Pablo Garcia Moreno and Yang Xiao and Xi Shen and Guillaume Obozinski and Neil Lawrence and Andreas Damianou},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2020},
    url={https://openreview.net/forum?id=Hkg-xgrYvH}
}
```

## Authors of the code
[Shell Xu Hu](http://hushell.github.io/), [Xi Shen](https://xishen0220.github.io/) and [Yang Xiao](https://youngxiao13.github.io/)


## Dependencies
The code is tested under **Pytorch > 1.0 + Python 3.6** environment with extra packages:
``` Bash
pip install -r requirements.txt
```


## How to use the code on Mini-ImageNet?
### **Step 0**: Download Mini-ImageNet dataset

``` Bash
cd data
bash download_miniimagenet.sh 
cd ..
```

### **Step 1** (optional): train a WRN-28-10 feature network (aka backbone)
The weights of the feature network are downloaded in step 0, but you may also train from scratch by running

``` Bash
python main_feat.py --outDir miniImageNet_WRN_60Epoch --cuda --dataset miniImageNet --nbEpoch 60
```

### **Step 2**: Meta-training on Mini-ImageNet, e.g., 5-way-1-shot:

``` Bash
python main.py --config config/miniImageNet_1shot.yaml --seed 100 --gpu 0
```

### **Step 3**: Meta-testing on Mini-ImageNet with a checkpoint:

``` Bash
python main.py --config config/miniImageNet_1shot.yaml --seed 100 --gpu 0 --ckpt cache/miniImageNet_1shot_K3_seed100/outputs_xx.xxx/netSIBBestxx.xxx.pth
```

## Mini-ImageNet Results (LAST ckpt)

| Setup         | 5-way-1-shot  | 5-way-5-shot |
| ------------- | -------------:| ------------:|
| SIB (K=3)     | 70.700% ± 0.585% | 80.045% ± 0.363%|
| SIB (K=5)     | 70.494 ± 0.619% | 80.192% ± 0.372%|

## CIFAR-FS Results (LAST ckpt)

| Setup         | 5-way-1-shot  | 5-way-5-shot |
| ------------- | -------------:| ------------:|
| SIB (K=3)     | 79.763% ± 0.577% | 85.721% ± 0.369%|
| SIB (K=5)     | 79.627 ± 0.593% | 85.590% ± 0.375%|
