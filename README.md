<p align="center"><img src="https://github.com/nathanbronson/AmorphousNeuralNetwork/blob/main/logo.jpg?raw=true" alt="logo" width="200"/></p>

_____
# AmorphousNeuralNetwork
a neural framework that knows no bounds

## About
AmorphousNeuralNetwork (`ANN`) is a basic attempt at a neural framework that transcends the layers and operations common to conventional neural networks. It uses a non-hierarchical node-based architecture. Each node uses only simple logic and an output action potential to communicate with other nodes. In their composition, a network, they are meant to approximate a given function based, which they learn from training data.

This codebase includes the code for a `Node`, `Network`, a few simple `Datasets`, and some training protocols. It also includes utilities to facilitate and visualize implementations using the framework.

In tests, `ANN` showed limited but intriguing signs of optimization. Further exploration of training protocols more fit for the framework could improve `ANN`'s performance. As well, `Node`s' operations can likely be improved to involve operations more fundamental than those currently implemented to facilitate more complete generalization.

Maintainence of this codebase has not been active since 2021.

## Usage
To replicate any predefined experiments, run `ANN.py`. Predefined experiments are:
```
ext
digital
digitalgof
extdigitalgof
extdigitalbpgof
analoggof
extanaloggof
extanalogbpgof
eoanaloggof
exteoanaloggof
```
Correct usage is as follows:
```
$ python3 ANN.py <experiment name>
```

All elements of the code are written to facilitate easy import in other code. They may be imported from either their source file or `ANN.py`.

## License
See `LICENSE`.
