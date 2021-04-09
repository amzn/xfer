
![Xfer](xfer-ml/docs/image/logo_330x200.png)

## Transfer and meta-learning in Python
--------------------------------------------------------------------------------


Each folder in this repository corresponds to a method or tool for transfer/meta-learning. `xfer-ml` is a standalone MXNet library (installable with pip) which largely automates deep transfer learning. The rest of the folders contain research code for a novel method in transfer or meta-learning, implemented in a variety of frameworks (not necessarily in MXNet). 

In more detail:
- [xfer-ml](xfer-ml): A library that allows quick and easy transfer of knowledge stored in deep neural networks implemented in MXNet. xfer-ml can be used with data of arbitrary numeric format, and can be applied to the common cases of image or text data. It can be used as a pipeline that spans from extracting features to training a repurposer. The repurposer is then an object that carries out predictions in the target task. You can also use individual components of the library as part of your own pipeline. For example, you can leverage the feature extractor to extract features from deep neural networks or ModelHandler, which allows for quick building of neural networks, even if you are not an MXNet expert.   
- [leap](leap): MXNet implementation of  "leap", the meta-gradient path learner [(link)](https://arxiv.org/abs/1812.01054) by S. Flennerhag, P. G. Moreno, N. Lawrence, A. Damianou, which appeared at ICLR 2019. 
- [nn_similarity_index](nn_similarity_index): PyTorch code for comparing trained neural networks using both feature and gradient information. The method is used in the arXiv paper [(link)](https://arxiv.org/abs/2003.11498) by S. Tang, W. Maddox, C. Dickens, T. Diethe and A. Damianou.   
- [finite_ntk](finite_ntk): PyTorch implementation of finite width neural tangent kernels from the paper *Fast Adaptation with Linearized Neural Networks* [(link)](https://arxiv.org/abs/2103.01439), by W. Maddox, S. Tang, P. G. Moreno, A. G. Wilson, and A. Damianou, which appeared at AISTATS 2021.    
- [synthetic_info_bottleneck](synthetic_info_bottleneck) PyTorch implementation of the *Synthetic Information Bottleneck* algorithm for few-shot classification on Mini-ImageNet, which is used in paper *Empirical Bayes Transductive Meta-Learning with Synthetic Gradients*  [(link)](https://openreview.net/forum?id=Hkg-xgrYvH) by S. X. Hu, P. G. Moreno, Y. Xiao, X. Shen, G. Obozinski, N. Lawrence and A. Damianou, which appeared at ICLR 2020.
- [var_info_distil](var_info_distil) PyTorch implementation of the paper *Variational Information Distillation for Knowledge Transfer* [(link)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf) by S. Ahn, S. X. Hu, A. Damianou, N. Lawrence, Z. Dai, which appeared at CVPR 2019.  


Navigate to the corresponding folder for more details.


### Contributing 

You may contribute to the existing projects by reading the individual contribution guidelines in each corresponding folder. 

## License

The code under this repository is licensed under the [Apache 2.0 License](LICENSE).
