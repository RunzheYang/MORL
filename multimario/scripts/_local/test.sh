python run_n3c.py \
	--env-id SuperMarioBros-v2 \
	--load-model \
	--prev-model SuperMarioBros-v2_n3c_Dec11_21-07-16.model\
	--use-gae \
	--life-done \
	--single-stage \
	--standardization \
	--num-worker 1 \
	--sample-size 1 \
	# --render \

python run_e3c.py \
	--env-id SuperMarioBros-v2 \
	--load-model \
	--prev-model SuperMarioBros-v2_e3c_b95_Jan23_11-04-34.model \
	--use-gae \
	--life-done \
	--single-stage \
	--standardization \
	--num-worker 1 \
	--sample-size 1 \
	# --render \

# SuperMarioBros-v2_e3c_b95_Jan23_11-04-34.model
# SuperMarioBros-v2_e3c_b95_Jan21_08-38-27.model
# SuperMarioBros-v2_e3c_b95_Jan19_11-41-37.model
# SuperMarioBros-v2_e3c_b95_Jan19_01-34-29.model
# SuperMarioBros-v2_n3c_Dec15_03-40-48.model \
# SuperMarioBros-v2_a3c_Dec11_15-56-58.model
# SuperMarioBros-v2_n3c_Dec11_21-07-16.model
# SuperMarioBros-v2_n3c_Dec12_09-23-28.model 
# SuperMarioBros-v2_a3c_Dec11_15-56-58.model