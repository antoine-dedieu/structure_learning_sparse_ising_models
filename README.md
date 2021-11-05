# Sample-Efficient L0-L2 Constrained Structure Learning of Sparse Ising Models

This repo contains code for reproducing the results in the paper [Sample-Efficient L0-L2 Constrained Structure Learning of Sparse Ising Models](https://arxiv.org/abs/2012.01744) published at the 35th AAAI Conference on Artificial Intelligence (AAAI 2021).



## Getting started

Dependencies can be simply installed via

```
pip install -r requirements.txt
python setup.py develop
```

  

## Ising model

This is the main folder. It contains the following files:
- `data.py` generates data for Ising models with the topologies described in the paper, without or with the use of a Gibbs-sampler
- `test_data.py` test the data generation code
- `l1_logreg.py` implements the L1 logistic regression estimator
- `l1_constrained_logreg.py` implements the L1-constrained logistic regression estimator
- `l1_ise.py` implements the L1 interaction screening estimator
- `l0_l2constrained_logreg.py` implements the proposed L0-L2 logistic regression estimator
- `l0_l2constrained_ise.py` implements the proposed L0-L2 interaction screening estimator

We refer to the main paper for a presentation of the different algorithms.


## Scripts

This folder contains the bash scripts used to generate the results in the main paper for Examples 1 and 2.

Each bash script calls the `compare_ising_models.py` python script.

`test_compare_ising_models_P16-ex1_small.sh` tests the implementation for a smaller number of independent problems (4 instead of 30).


## Notebooks

This folder contains a notebook which displays the simulation results as graphs which are similar to the ones in the main paper.

The default code assumes that the script `compare_ising_models_P16-ex1.sh` has been run.