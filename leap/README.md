## leap: meta-gradient path learner in MXNet

--------------------------------------------------------------------------------

MXNet implementation of  "leap", the meta-gradient path learner published in ICLR 2019: [(link)](https://arxiv.org/abs/1812.01054) by S. Flennerhag, P. G. Moreno, N. Lawrence, A. Damianou.


## Getting Started
Check the `demos` and `test` folders. 


## Installation
* __Dependencies:__
Primary dependency is MXNet >=1.2. See all requirements in [setup.py](setup.py).
* __Supported architectures / versions:__
Python 3.6+ on MacOS and Amazon Linux. 


-  __Install from source:__
To install leap from source, after cloning the repository run the following from the xfer/leap directory:
```
pip install .
```

Alternatively you can install with:
```
python setup.py install 
```

To confirm installation, run:
```python
>>> import leap
>>> leap.__version__
```
And confirm that version returned matches the expected package version number.


## License

leap is licensed under the [Apache 2.0 License](LICENSE).
