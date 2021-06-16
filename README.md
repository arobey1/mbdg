# Model-Based Domain Generalization

This repository contains the coded needed to reporduce the results of [Model-Based Domain Generalization](https://arxiv.org/abs/2102.11436).  In particular, we include two repositorys:

1. A fork of [DomainBed](https://github.com/facebookresearch/DomainBed) which can be used to reproduce our results on ColoredMNIST, PACS, and VLCS.
2. A separate implementation that can be used to reproduce our results on Camelyon17-WILDS and FMoW-WILDS.

We also include a library of trained domain transformation models for ColoredMNIST, PACS, Camelyon17-WILDS, and FMoW-WILDS.  This library will be updated in the future with more models.  All models can be downloaded from the following Google Drive link:

```
https://drive.google.com/drive/folders/1vDlZXk_Jow3bkPTlJLlloYCxOZAwnGBv?usp=sharing
```

In this README, we provide an overview describing how this code can be run.  If you find this repository useful in your research, please consider citing:


```latex
@article{robey2021model,
  title={Model-Based Domain Generalization},
  author={Robey, Alexander and Pappas, George J and Hassani, Hamed},
  journal={arXiv preprint arXiv:2102.11436},
  year={2021}
}
```

## DomainBed implementation

In the DomainBed implementation of our code, we implement our primal-dual style MBDG algorithm in `./domainbed/algorithms.py` as well as three algorithmic variants as described in Appendix C: MBDA, MBDG-DA, and MBDG-Reg.  These algorithms can be run using the same commands as the original DomainBed repository.

Our method is based on a primal-dual scheme for solving the Model-Based Domain Generalization constrained optimization problem.  This procedure is described in Algorithm 1 in our paper.  In particular, the core of our algorithm is an alternation between updating the primal variable θ (e.g., the parameter of a neural network based classifier) and updating the dual variable λ.  Below, we highlight a short code snippet that outlines our method:

```python
class MBDG(MBDG_Base):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MBDG, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.dual_var = torch.tensor(1.0).cuda().requires_grad_(False)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])

        # calculate classification loss (loss(θ) in Algorithm 1)
        clean_output = self.predict(all_x)
        clean_loss = F.cross_entropy(clean_output, all_y)

        # calculate regularization term (distReg(θ) in Algorithm 1)
        dist_reg = self.calc_dist_reg(all_x, clean_output)

        # formulate the (empirical) Lagrangian Λ = loss(θ) + λ distReg(θ)
        loss = clean_loss + self.dual_var * dist_reg

        # perform primal step in θ
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # calculate constaint unsatisfaction term (distReg(θ) - γ)
        const_unsat = dist_reg.detach() - self.hparams['mbdg_gamma']

        # perform dual step in λ
        self.dual_var = self.relu(self.dual_var + self.hparams['mbdg_dual_step_size'] * const_unsat)

        return {'loss': loss.item(), 'dist_reg': dist_reg.item(), 'dual_var': self.dual_var.item()}

```

## WILDS implementation

The WILDS datasets provide out-of-distribution validation sets to perform model selection.  Our code uses these validation sets in the `./mbdg-for-wilds` sub-repository.  The launcher script in `./dist_launch` can be used to train classifiers on both Camelyon17-WILDS and on FMoW-WILDS.

