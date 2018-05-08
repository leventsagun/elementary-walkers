# elementary-walkers
one or many walkers can interact and take various hyper-parameter values to explore a simple landscape  

## Intro 

This repository is designed to collect data on various ways of training simple neural networks. It can run SGD on multiple workers where they may interact (à la EASGD or Robust Ensembles), it can run GD and SGD for a comparison, it can interpolate between all workers at each iteration, it can calculate the full spectrum of the Hessian and/or top/bottom few eigenvalues, it can interactively change hyperparameters after predetermined timesteps and it can do random walk on the loss surface. Training can be done on true labels or on soft labels. The dataset can be corrupted in several ways. 

## Dependencies  
Required packages: 
- [Torch7](http://torch.ch)
- [fb.python](https://github.com/facebookarchive/fblualib/blob/master/fblualib/python/README.md)
- pandas and numpy for python 2.7
  
*NOTE:* Beware that this code has tricky dependencies, I encountered some issues setting `fb.python` up on MacOs.   

## References
  
This code is used in the following works:

[1] Levent Sagun, Leon Bottou, Yann LeCun, [*Eigenvalues of the Hessian in Deep Learning: Singularity and Beyond*](https://arxiv.org/abs/1611.07476)    

[2] Levent Sagun, Utku Evci, V. Ugur Guney, Yann Dauphin, Leon Bottou, [*Empirical Analysis of the Hessian of Over-Parametrized Neural Networks*](https://arxiv.org/abs/1706.04454)    

Relevant related works (albeit in an orthogonal direction) are:

[3] Sixin Zhang, Anna Choromanska, Yann LeCun, [*Deep learning with Elastic Averaging SGD*](https://arxiv.org/abs/1412.6651)  

[4] Carlo Baldassi, Christian Borgs, Jennifer Chayes, Alessandro Ingrosso, Carlo Lucibello, Luca Saglietti, Riccardo Zecchina, [*Unreasonable Effectiveness of Learning Neural Networks: From Accessible States and Robust Ensembles to Basic Algorithmic Schemes*](https://arxiv.org/abs/1605.06444)  

[5] Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, Riccardo Zecchina, [*Entropy-SGD: Biasing Gradient Descent Into Wide Valleys*](https://arxiv.org/abs/1611.01838)  
