# catastrophicForgetting project
The following project is a group of research tools (mostly Pythonic ones) used to study high-dimensional dynamics of neural networks.
The idea for the project follows the study of the dynamics of generalization error in ([Advani et al. 2020](https://www.sciencedirect.com/science/article/pii/S0893608020303117)). 

At this moment all neural networks in this project are build using [PyTorch](https://pytorch.org/) library.

Main simulation is currently in [Main](research/main.py).

Example figure presents three error plots: training error, validation error and cross generalization (how well network recognizes other data, not presented to its during training),
all computed simultaneously during training phase. Training phase consists of two subphases, when the main network (called Student) is fed with data first from the Teacher 1 and then from the Teacher 2 networks.
![alt text](https://github.com/Michaeldz36/catastrophicForgetting/blob/master/docs/figs/numerical+analytical_plots.png?raw=true)


The jupyter notebook version of the simulation (with possibility of small setup-tweaking) can be runned in cloud using following link:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Michaeldz36/catastrophicForgetting/blob/master/research/research_simulation.ipynb)