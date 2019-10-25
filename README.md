# Signal Processing Algorithms

A suite of algorithms based upon a python version of [e.divisive](https://www.rdocumentation.org/packages/ecp/versions/3.1.0/topics/e.divisive) which is itself based on [this document](https://arxiv.org/pdf/1306.4933.pdf) for change point detection, and [Generalized ESD Test for Outliers
(gesd)](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm) for outlier detection.

## Common Configuration

First off, configure your pip: https://github.com/10gen/example-dag-python-package

## Getting Started - Developers

Getting the code:

```
$ git clone git@github.com:10gen/signal-processing-algorithms.git
$ cd signal-processing-algorithms
```

Making a virtual environment and installing the stuff you need into it:
```
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ pip install -e .
$ pip install -r requirements.txt
```
Testing stuff:
```
$ pytest --flake8
```
Formatting things:
```
$ black setup.py src tests
```

Doing that stuff automatically before you push to the repo:
```
cp scripts/pre-push .git/hooks
```

## Getting Started - Users
```
pip install signal-processing-algorithms
```