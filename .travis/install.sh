#!/bin/bash

# Workaround for OSX based on https://github.com/travis-ci/travis-ci/issues/2312#issuecomment-195620855
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
  # Get latest version of brew
  brew update
  # Recommended installs https://github.com/pyenv/pyenv/wiki#suggested-build-environment
  brew outdated openssl || brew upgrade openssl
  brew outdated readline || brew upgrade readline
  # Install pyenv
  brew outdated pyenv || brew upgrade pyenv
  # Install pyenv wrapper
  brew install pyenv-virtualenv
  pyenv install $PYTHON
  # Set pyenv to use chosen Python version
  export PYENV_VERSION=$PYTHON
  export PATH="/Users/travis/.pyenv/shims:${PATH}"
  # Create and activate virtual environment
  pyenv-virtualenv venv
  source venv/bin/activate
  # Check Python version
  python --version
fi