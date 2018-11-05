python train.py --env-name SuperMarioBros-v2 \
--method naive \
--model cnn \
--gamma 0.99 \
--mem-size 10000 \
--batch-size 16 \
--lr  1e-3 \
--epsilon 0.5 \
--epsilon-decay \
--weight-num 1 \
--episode-num 10000 \
--update-freq 100 \
--priority \
--optimizer Adam \
--save saved/ \
--log logs/ \
--single \
--name naive_v2_cnn_10000_16_1e-3_1_pri_100_single