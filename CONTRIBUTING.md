## Overview

Hello and thanks for considering contributing to Xfer! It's people like you who make the project great!

Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Following these guidelines helps smooth out the process of contributing for both you as a contributor and those who maintain the project, making better use of everyone's time and energy.

### Reporting a bug
If you find a bug in Xfer, please file an issue using our [template](issue template), detailing as much about your setup as possible to help future developers in reproducing the issue. Tag it as a bug, with any additional tags as relevant.

Information should contain the following:
1. What version of Xfer and MXNet are you using?
2. What operating system are you using?
3. What python version are you using?
4. What did you do?
5. What did you expect to happen?
6. What actually happened?
7. Have you made any modifications relevant to the bug?

### Submitting a feature request
If you're wishing for a feature that doesn't exist yet in Xfer, there are probably others who need it too. Open an issue on GitHub describing what it is you'd like to see added, why it's useful, and how it might work. Add an "Enhancement" tag for bonus points! Better yet, submit a pull request providing the feature you need!

### Contributing code

If you're thinking about adding code to Xfer, here are some guidelines to get you started.

* If the change is a major feature, create a [CIP](link to CIP definition) in the docs/CIPs folder and post it as a PR, optionally with a prototype implementation of your proposed changes. This is to get community feedback on the changes and document the design reasoning of Xfer for future reference.

* Keep pull requests small, preferably one feature per pull request. This lowers the bar to entry for a reviewer, and keeps feedback focused for each feature.

Some major areas where we appreciate contributions:
* [Creating new Repurposers](repurposer_howto_link)
* Expanding on the capabilities of Model Handler.
* Example notebooks showing how to repurpose models in new data domains. [Example](example)

If you're still not sure where to begin, have a look at our [issues](issues) page for open work.


## Contributing via Pull Requests

So you've written up a new Repurposer and you want to give it back to the community. Now it's time to submit a pull request! Before sending us a pull request, please ensure that:

1. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
2. You open an issue to discuss any significant work - we would hate for your time to be wasted.
3. You are working against the latest source on the *master* branch.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, making sure you've answered the questions in the pull request checklist (see below.)
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

Feel free to ask for help if you're unsure what to do at any point in the process, or on how best to integrate your code with the existing codebase!

After submitting a pull request, you can expect a response from a maintainer within 7 days. If you still haven't heard back by then, feel free to ping the thread as a reminder.

#### Coding style
We follow to PEP8 standards as a general rule.

We prefer explicit variable names like ```mean``` and ```variance``` to ones like ```location``` and ```scale``` or greek letters like ```mu``` and ```sigma```.


#### Pull Request Checklist
Before submitting the pull request, please go through this checklist to make the process smoother for both yourself and the reviewers.
* Are there unit tests with good code coverage?  Please include numerical stability checks against edge cases if applicable.
* Do all public functions have docstrings including examples? If you added a new module, did you add it to the Sphinx ```api.rst``` file in the ```doc``` folder?
* Is the code style correct (PEP8)? You can verify this by running `flake8` from the top level directory.
* Is the commit message formatted correctly?
* If this is a large addition, is there a tutorial or more extensive module-level description? Did you discuss the addition in a [CIP](CIP)? Is there an issue related to the change? If so, please link the issue or CIP.


## Setting up a development environment

### Building the code
Once you have the repository cloned from git, installing it should be as simple as running:
```
pip install -e .
```

### Running tests
Install test requirements by running:
```
pip install -r test_requirements.txt
```
Run the full suite of tests by running the following command from the top level directory:
```
pytest
```

To run unit tests and integration tests separately, use:
```
pytest tests/unit
pytest tests/integration
```

#### Troubleshooting
When running the tests on OSX, you may get this error when running the GpRepurposer.
```
from matplotlib.backends import _macosx
RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
```
This can be fixed by running the following command from the root directory of the repo.
```
echo 'backend: Agg' > matplotlibrc
```

### Generating docs
Documentation contributions are much appreciated! If you see something incorrect or poorly explained, please fix it and send the update!

If you'd like to generate the docs locally, first install the required Xfer dependencies from ```requirements.txt``` and the necessary documentation packages from the ```docs/requirements.txt``` file. Due a dependency on [nbconvert](https://nbconvert.readthedocs.io/en/latest/index.html), you will need to install Pandoc separately from [here](http://pandoc.org/installing.html). Then, from inside the doc directory, run:

```
make html
```

## Licensing

See the [LICENSE](https://github.com/${GITHUB_ORG}/${GITHUB_REPO}/blob/master/LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.

## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact opensource-codeofconduct@amazon.com with any additional questions or comments.

## Acknowledgements
CONTRIBUTING contents partially inspired from [scipy's](https://github.com/scipy/scipy/blob/master/HACKING.rst.txt) and [Rust's](https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.md) contribution guides!
