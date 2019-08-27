# Deep Multi-Objective Reinforcement Learning

[A Generalized Algorithm for Multi-Objective RLand Policy Adaptation](https://arxiv.org/abs/1908.08342), arXiv:1908.08342.   

## Abstract

We introduce a new algorithm for multi-objective reinforcement learning (MORL) with linear preferences, with the goal of enabling few-shot adaptation to new tasks.In MORL, the aim is to learn policies over multiple competing objectives whoserelative importance (preferences) is unknown to the agent. While this alleviatesdependence on scalar reward design, the expected return of a policy can changesignificantly with varying preferences, making it challenging to learn a singlemodel to produce optimal policies under different preference conditions. We propose a generalized version of the Bellman equation to learn a single parametricrepresentation for optimal policies over the space of all possible preferences. Afterthis initial learning phase, our agent can quickly adapt to any given preference, orautomatically infer an underlying preference with very few samples. Experimentsacross four different domains demonstrate the effectiveness of our approach.a

## Instructions

The experiments on two synthetic domains, Deep Sea Treasure (DST) and Fruit Tree Navigation (FTN), as well as two complex real domains, Task-Oriented Dialog Policy Learning (Dialog) and SuperMario Game (SuperMario).

### `synthetic`

* Example - train envelope MOQ-learning on FTN domain:  
`python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --name 0`

The code for our envelope MOQ-learning algorithm is in `synthetic/crl/envelope/meta.py`, neural network architecture is configurable in `synthetic/crl/envelope/models`. Two synthetic environments are under the file `synthetic/envs`.

### `pydial`

Code for Task-Oriented Dialog Policy Leanring. The environment is modified from [PyDial](http://www.camdial.org/pydial/).

* Example - train envelope MOQ-learning on Dialog domain:  
`pydial train config/MORL/Train/envelope.cfg`

The code for our envelope MOQ-learning algorithm is in `pydial/policy/envelope.py`.

### `multimario`

The multi-objective version SuperMario Game. The environment is modified from [Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros). 

* Example - train envelope MOQ-learning on SuperMario domain:  
`python run_e3c_double.py --env-id SuperMarioBros-v2 --use-cuda --use-gae --life-done --single-stage --training --standardization --num-worker 16 --sample-size 8 --beta 0.05 --name e3c_b05`

The code for our envelope MOQ-learning algorithm is in `multimario/agent.py`. Two multi-objective version environment is in `multimario/env.py`.

## Cite
```
@article{MORL:arxiv-1908-08342,
  author    = {Runzhe Yang and
               Xingyuan Sun and
               Karthik Narasimhan},
  title     = {A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation},
  journal   = {CoRR},
  volume    = {abs/1908.08342},
  year      = {2019},
  url       = {http://arxiv.org/abs/1908.08342},
  archivePrefix = {arXiv},
  eprint    = {1908.08342},
}

```
