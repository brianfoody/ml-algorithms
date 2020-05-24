# Building out my intuition for ML Algorithms

Trying to ease the burden for getting start with ML learning using Python.

This project uses PyEnv to lock in the Python version and Virtual Env for easily managing dependencies.

All python code will be typed to hopefully aid understanding.

## Setup

- [Install PyEnv](https://github.com/pyenv/pyenv-installer)
- Install virtualenv
  `python -m pip install --user virtualenv`
- Initialise your virtual environment
  `python -m virtualenv .venv`
- Source into it. Open up a Python terminal in VS Code to do automaticall or
  `source .venv/bin/activate`
- Install dependencies
  `python -m pip install -r requirements.txt`

### K-Means

Basic flow of k-means on a 2D and 3D dataset
`python src/k-means/twoD.py`
`python src/k-means/threeD.py`

Recommend uncommenting each line individually under \_\_main\_\_ to get a feel for the process.

https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
