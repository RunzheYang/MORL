python run_a3c.py \
	--env-id SuperMarioBros-v2 \
	--load-model \
	--prev-model SuperMarioBros-v2_a3c_Dec09_23-53-57.model \
	--render \
	--use-gae \
	--life-done \
	--single-stage \
	--standardization \
	--num-worker 1
