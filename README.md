# Model-Based Domain Generalization

This repository contains the coded needed to reporduce the results of [Model-Based Domain Generalization](https://arxiv.org/abs/2102.11436).  In particular, we include two repositorys:

1. A fork of [DomainBed](https://github.com/facebookresearch/DomainBed) which can be used to reproduce our results on ColoredMNIST, PACS, and VLCS.
2. A separate implementation that can be used to reproduce our results on Camelyon17-WILDS and FMoW-WILDS.

## DomainBed implementation

In the DomainBed implementation of our code, we implement our primal-dual style MBDG algorithm in `./domainbed/algorithms.py` as well as three algorithmic variants as described in Appendix C: MBDA, MBDG-DA, and MBDG-Reg.  These algorithms can be run using the same commands as the original DomainBed repository.

## WILDS implementation

The WILDS datasets provide out-of-distribution validation sets to perform model selection.  Our code uses these validation sets in the `./mbdg-for-wilds` sub-repository.  The launcher script in `./dist_launch` can be used to train classifiers on both Camelyon17-WILDS and on FMoW-WILDS.

