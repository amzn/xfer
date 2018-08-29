![Xfer](docs/image/logo_330x200.png)

## Deep Transfer Learning for MXNet

--------------------------------------------------------------------------------


[Website](https://github.com/amzn/xfer) |
[Documentation](https://github.com/amzn/xfer/docs) |
[Contribution Guide](CONTRIBUTING.md)

#### What is Xfer?
Xfer is a library that allows quick and easy transfer of knowledge<sup>[1](ftp://ftp.cs.wisc.edu/machine-learning/shavlik-group/torrey.handbook09.pdf),[2](http://cs231n.github.io/transfer-learning/),[3](https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)</sup> stored in deep neural networks implemented in [MXNet](https://mxnet.incubator.apache.org/). Xfer can be used with data of arbitrary numeric format, and can be applied to the common cases of image or text data.

Xfer can be used as a pipeline that spans from extracting features to training a repurposer. The repurposer is then an object that carries out predictions in the target task.

You can also use individual components of Xfer as part of your own pipeline. For example, you can leverage the feature extractor to extract features from deep neural networks or ModelHanlder, which allows for quick building of neural networks, even if you are not an MXNet expert.

#### Why should I use Xfer?
* _Resource efficiency_: you don't have to train big neural networks from scratch.
* _Data efficiency_: by transferring knowledge, you can classify complex data even if you have very few labels.
* _Easy access to neural networks_: you don't need to be an ML ninja in order to leverage the power of neural networks. With Xfer you can easily re-use them or even modify existing architectures and create your own solution.
* _Utilities for feature extraction from neural networks_.
* _Rapid prototyping_: ModelHandler allows you to easily modify a neural network architecture, e.g. by providing one-liners for adding / removing / freezing layers.
* _Uncertainty modeling_: With the Bayesian neural network (BNN) or the Gaussian process (GP) repurposers, you can obtain uncertainty in the predictions of the rerpurposer.

#### Minimal demo

After defining an MXNet _source_ model and data iterators for your _target_ task, you can perform transfer learning with just 3 lines of code:
```
repurposer = xfer.LrRepurposer(source_model, feature_layer_names=['fc7'])
repurposer.repurpose(train_iterator)
predictions = repurposer.predict_label(test_iterator)
```

## Getting Started
* [Documentation](TODO)
* [Tutorial: Introduction and transfer learning for image data](TODO)
* [Tutorial: Transfer learning with automatic hyperparameter tuning](TODO)
* [Tutorial: Transfer learning for text data](TODO)
* [Tutorial: Creating your own custom repurposer](TODO)
* [Tutorial: xfer.ModelHandler for easy manipulation and inspection of MXNet models](TODO)

## Installation
* __Dependencies:__
Primary dependencies are MXNet >=1.2 and GPy >= 1.9.3. See [requirements](requirements.txt).
* __Supported architectures / versions:__
Tested on Python 3.4+ on MacOS and Amazon Linux.

- __Install with pip:__
```
pip install xfer-ml
```
-  __Install from source:__
To install Xfer from source, after cloning the repository run the following from the top-level directory:
```
pip install .
```

To confirm installation, run:
```python
>>> import xfer
>>> xfer.__version__
```
And confirm that version returned matches the expected package version number.

## Contributing
Have a look at our [contributing guide](CONTRIBUTING.md), thanks for the interest!


## License

Xfer is licensed under the [Apache 2.0 License](LICENSE).
