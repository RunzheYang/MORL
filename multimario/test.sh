python run_a3c.py \
	--env-id SuperMarioBros-v2 \
	--load-model \
	--prev-model SuperMarioBros-v2_a3c_Dec09_00-52-37.model \
	--render \
	--use-gae \
	--life-done \
	--standardization \
	--num-worker 1
