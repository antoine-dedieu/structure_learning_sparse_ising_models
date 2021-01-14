# Sample-Efficient L0-L2 Constrained Structure Learning of Sparse Ising Models

  

  

## Getting started

We use the following dependencies

```

Python 3.7.4 with standard distribution (anaconda)

spgl1 package which can be downloaded via "pip install spgl1"

```

Dependencies can be simply installed via

```

python setup.py develop

```

  

## Ising model

This is the main folder. It contains the following files:

- data.py generates data for Ising models of specified topologies, with or without Gibbs-sampler

- test_data.py test the data generation code

- l1_logreg.py implements the L1 logistic regression estimator

- l1_constrained_logreg.py implements the L1-constrained logistic regression estimator

- l1_ise.py implements the L1 interaction screening estimator

- l0_l2constrained_logreg.py implements the proposed L0-L2 logistic regression estimator

- l0_l2constrained_ise.py implements the proposed L0-L2 interaction screening estimator

We refer to the main paper for a presentation of the different algorithms.

  

## Scripts

This folder contains the bash scripts used to generate the results in the main paper for Examples 1 and 2.

In particular each bash script calls the `compare_ising_models.py` python script.

  

  

## Notebooks

This folder contains a notebook which displays the simulation results as graphs which are similar to the ones in the main paper.