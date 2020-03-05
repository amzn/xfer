
![Xfer](xfer-ml/docs/image/logo_330x200.png)

## Transfer and meta-learning in Python
--------------------------------------------------------------------------------


Each folder in this repository corresponds to a method or tool for transfer/meta-learning. `xfer-ml` is a standalone MXNet library (installable with pip) which largely automates deep transfer learning. The rest of the folders contain research code for a novel method in transfer or meta-learning, implemented in a variety of frameworks (not necessarily in MXNet). 

In more detail:
- [xfer-ml](xfer-ml): A library that allows quick and easy transfer of knowledge stored in deep neural networks implemented in MXNet. xfer-ml can be used with data of arbitrary numeric format, and can be applied to the common cases of image or text data. It can be used as a pipeline that spans from extracting features to training a repurposer. The repurposer is then an object that carries out predictions in the target task. You can also use individual components of the library as part of your own pipeline. For example, you can leverage the feature extractor to extract features from deep neural networks or ModelHandler, which allows for quick building of neural networks, even if you are not an MXNet expert.   
- [leap](leap): MXNet implementation of  "leap", the meta-gradient path learner published in ICLR 2019: [(link)](https://arxiv.org/abs/1812.01054) by S. Flennerhag, P. G. Moreno, N. Lawrence, A. Damianou.  


Navigate to the corresponding folder for more details.


### Contributing 

You may contribute to the existing projects by reading the individual contribution guidelines in each corresponding folder. 

## License

The code under this repository is licensed under the [Apache 2.0 License](LICENSE).
