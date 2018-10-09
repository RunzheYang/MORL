# Prioritized Experience Replay

### Usage
1. in rank_base.py Experience.stroe give a simple description of store replay memory, or you can also refer rank_base_test.py
2. It's more convenient to store replay as format (state_1, action_1, reward, state_2, terminal). If we use this method, all replay memory in Experience are legal and can be sampled as we like.

### Rank-based
use binary heap tree as priority queue, and build an Experience class to store and retrieve the sample
  
    Interface:
    * All interfaces are in rank_based.py
    * init conf, please read Experience.__init__ for more detail, all parameters can be set by input conf
    * replay sample store: Experience.store
    	params: [in] experience, sample to store
    	returns: bools, True for success, False for failed
    * replay sample sample: Experience.sample
    	params: [in] global_step, used for cal beta
    	returns: 
    		experience, list of samples
    		w, list of weight
    		rank_e_id, list of experience's id, used for update priority value
    * update priority value: Experience.update
    	params: 
    		[in] indices, rank_e_ids
    		[in] delta, new TD-error

### Proportional
not implement yet

### Reference
1. "Prioritized Experience Replay" http://arxiv.org/abs/1511.05952
2. [Atari](https://github.com/Kaixhin/Atari) by @Kaixhin, Atari uses torch to implement rank-based algorithm.

### Application
1. TEST1 PASSED: These code has been applied to my own NLP DQN experiment, it significantly improves performance.

### TO DO LIST
1. proportional method, if anyone would like to do this part, please let me know.
