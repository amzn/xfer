## leap: meta-gradient path learner in MXNet  
--------------------------------------------------------------------------------

## Introduction  

This is the MXNet implementation of  "leap", the meta-gradient path learner published in ICLR 2019: [(link)](https://arxiv.org/abs/1812.01054) by S. Flennerhag, P. G. Moreno, N. Lawrence, A. Damianou.

Please cite our work if you find it useful:
```
@inproceedings{
   flennerhag2018transferring,
   title={Transferring Knowledge across Learning Processes},
   author={Sebastian Flennerhag and Pablo Garcia Moreno and Neil Lawrence and Andreas Damianou},
   booktitle={International Conference on Learning Representations},
   year={2019},
   url={https://openreview.net/forum?id=HygBZnRctX}
}
```

## Getting Started
Check the `demos` folder. The `.ipynb` file can be run in Jupyter [lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) or [notebook](https://jupyter.readthedocs.io/en/latest/install.html).


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
Alternatively, you can install with:
```
python setup.py install
```
For installing in editable/development mode, the above commands become respectively:
`pip install -e .` and `python setup.py develop` .

To confirm installation, run:
```python
>>> import leap
>>> leap.__version__
```
And confirm that version returned matches the expected package version number.


## License

leap is licensed under the [Apache 2.0 License](LICENSE).
