CUDA_VISIBLE_DEVICES=1 python train.py \
--env-name SuperMarioBros-v1 \
--method naive \
--model cnn \
--gamma 0.9999 \
--mem-size 100000 \
--batch-size 32 \
--lr  1e-4 \
--epsilon 0.8 \
--epsilon-decay \
--weight-num 4 \
--episode-num 10000 \
--update-freq 50 \
--optimizer Adam \
--save saved/ \
--log logs/ \
--name naive_v1_cnn_100000_32_1e-4_4_50

