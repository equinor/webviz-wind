[![Build Status](https://github.com/equinor/webviz-plugin-boilerplate/workflows/webviz-plugin-boilerplate/badge.svg)](https://github.com/equinor/webviz-plugin-boilerplate/actions?query=branch%3Amaster)
[![Python 3.6 | 3.7 | 3.8 | 3.9 ](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9-blue.svg)](https://www.python.org/)


## Install your new Python plugin package

Make sure that you have the right public keys available on github

See some hints here.
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

One way of generating a new public key:
```bash
cd                      (go to your private folder on Linux)
ssh-keygen -t rsa       (a new key is generated and put on .ssh/id_rsa  and .ssh/id_rsa.pub. The latter is the public key)
```

Add  the public key to your github account:
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

Then you are ready to clone:

```bash
git clone git@github.com:equinor/webviz-wind.git
```

To install your plugin package in _development mode_, run

```bash
komodoenv -r stable kevn            (to generate a virtual environment so you can use pip install later)
source kenv/enable.csh              (now you are ready to install your new webviz plugins)

cd YOUR_GIT_CLONE_OF_THIS_PROJECT
pip install -e .
```

This will (first time) install all dependencies, but the `-e` flag will also make sure your plugin project is installed in edit/development mode. This means that when you update the Python files in your package, this will automatically be available in your current Python environment without having to reinstall the package.

## Test your new Python plugin package

After installation you can test the custom plugins from your package using the provided example configuration file:

```bash
webviz build ./examples/wind_example.yml
```

If you want to use the Equinor theme, you can do this once to configure Equinor theme as default:
```bash
pip install webviz-config-equinor
webviz preferences --theme equinor
```

If you want to install test and linting dependencies, you can in addition run

```bash
pip install .[tests]
```

### Linting

You can do automatic linting of your code changes by running

```bash
black --check webviz_wind # Check code style
pylint webviz_wind # Check code quality
bandit -r webviz_wind  # Check Python security best practice
```

### Usage and documentation

For general usage, see the documentation on
[webviz-config](https://github.com/equinor/webviz-config).

## Make awesome stuff :eyeglasses:

You are now ready to modify the package with your own plugins. Have fun! :cake: