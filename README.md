# Deep Multi-Objective Reinforcement Learning

[A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation](https://arxiv.org/abs/1908.08342),  to appear in NeurIPS'19.   

## Abstract

We introduce a new algorithm for **multi-objective reinforcement learning (MORL)** with **linear preferences**, with the goal of enabling **few-shot adaptation** to new tasks. In MORL, the aim is to learn policies over multiple competing objectives whose relative importance (preferences) is **unknown** to the agent. While this alleviates dependence on scalar reward design, the expected return of a policy can change significantly with varying preferences, making it challenging to learn a single model to produce optimal policies under different preference conditions. We propose a generalized version of the Bellman equation to learn a single parametric representation for **optimal policies over the space of all possible preferences**. After this initial learning phase, our agent  can execute the optimal policy under any given preference, or automatically infer an underlying preference with very few samples. Experiments across four different domains demonstrate the effectiveness of our approach.

## Instructions

The experiments on two synthetic domains, **Deep Sea Treasure (DST)** and **Fruit Tree Navigation (FTN)**, as well as two complex real domains, **Task-Oriented Dialog Policy Learning (Dialog)** and **SuperMario Game (SuperMario)**.

### `synthetic`

PyTorch version for the code in `synthetic` was torch 0.4.0 (sorry for the 2 years old code) with Python 3.5,
and the visdom version is 0.1.6.3

* Example - train envelope MOQ-learning on FTN domain:  
`python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --name 0`

The code for our envelope MOQ-learning algorithm is in `synthetic/crl/envelope/meta.py`, neural network architecture is configurable in `synthetic/crl/envelope/models`. Two synthetic environments are under the file `synthetic/envs`.

### `pydial`

Code for Task-Oriented Dialog Policy Leanring. The environment is modified from [PyDial](http://www.camdial.org/pydial/).

PyTorch version for the code in `pydial` was torch 0.4.1 with Python 2.7 (since the PyDial requires Python 2)

* Example - train envelope MOQ-learning on Dialog domain:  
`pydial train config/MORL/Train/envelope.cfg`

The code for our envelope MOQ-learning algorithm is in `pydial/policy/envelope.py`.

### `multimario`

The multi-objective version SuperMario Game. The environment is modified from [Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros). 

PyTorch version for the code in `multimario` was torch 1.1.0 with Python 3.5.

* Example - train envelope MOQ-learning on SuperMario domain:  
`python run_e3c_double.py --env-id SuperMarioBros-v2 --use-cuda --use-gae --life-done --single-stage --training --standardization --num-worker 16 --sample-size 8 --beta 0.05 --name e3c_b05`

The code for our envelope MOQ-learning algorithm is in `multimario/agent.py`. Two multi-objective version environment is in `multimario/env.py`.

## Citation
```
@incollection{yang2019morl,
  title = {A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation},
  author = {Yang, Runzhe and Sun, Xingyuan and Narasimhan, Karthik},
  booktitle = {Advances in Neural Information Processing Systems 32},
  editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
  pages = {14610--14621},
  year = {2019},
  publisher = {Curran Associates, Inc.},
  url = {http://papers.nips.cc/paper/9605-a-generalized-algorithm-for-multi-objective-reinforcement-learning-and-policy-adaptation.pdf}
}
```
