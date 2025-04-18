# Quantum state tomography with gradient descent 

This repository contains the code required to reproduce the plots from https://arxiv.org/abs/2503.04526  

## Installation and use

Install the myenvironment.yml or you can install the libraries by yourself, but be careful with the version of QuTip, in this project qutip==5.0.0 is used.

```python
conda env create -f environment.yml
conda activate applesilicon_gd_qst_env
```
The notebooks in the `examples` folder are tutorials on how to do the quantum state tomography with the different methods of gradient descent. 

The `data_and_paper-plots` folder contains a Jupyter Notebook (`.ipynb`) that runs the methods used to generate the plots presented in the paper, along with a ZIP file containing the corresponding data.
