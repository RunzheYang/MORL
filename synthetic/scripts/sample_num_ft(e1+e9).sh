# No. | mem_size | batch_size | weight_num | update_freq | lr | beta
# ENVELOPE:
# b0.1_s0:   4000 | 256 | 1   | 100 | 1e-3 | 0.1
# b0.1_s0.5: 4000 | 256 | 4   | 100 | 1e-3 | 0.1
# b0.1_s1:   4000 | 256 | 8   | 100 | 1e-3 | 0.1
# b0.1_s2:   4000 | 256 | 16  | 100 | 1e-3 | 0.1
# b0.1_s3:   4000 | 256 | 32  | 100 | 1e-3 | 0.1
# b0.1_s4:   4000 | 256 | 64  | 100 | 1e-3 | 0.1
# b0.1_s5:   4000 | 256 | 128 | 100 | 1e-3 | 0.1

# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0.5_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0.5_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0.5_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0.5_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s0.5_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s1_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s1_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s1_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s1_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s1_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s2_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s2_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s2_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s2_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s2_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s3_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s3_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s3_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s3_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s3_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s4_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s4_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s4_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s4_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s4_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s5_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s5_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s5_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s5_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.10 --name b0.1_s5_r4




# ENEVELOPE:
# b0.9_s0:   4000 | 256 | 1   | 100 | 1e-3 | 0.9
# b0.9_s0.5: 4000 | 256 | 8   | 100 | 1e-3 | 0.9
# b0.9_s1:   4000 | 256 | 8   | 100 | 1e-3 | 0.9
# b0.9_s2:   4000 | 256 | 16  | 100 | 1e-3 | 0.9
# b0.9_s3:   4000 | 256 | 32  | 100 | 1e-3 | 0.9
# b0.9_s4:   4000 | 256 | 64  | 100 | 1e-3 | 0.9
# b0.9_s5:   4000 | 256 | 128 | 100 | 1e-3 | 0.9
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0.5_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0.5_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0.5_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0.5_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s0.5_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s1_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s1_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s1_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s1_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s1_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s2_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s2_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s2_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s2_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s2_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s3_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s3_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s3_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s3_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s3_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s4_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s4_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s4_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s4_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s4_r4
#
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s5_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s5_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s5_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s5_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --name b0.9_s5_r4


# beta = 0.1
# test Pareto
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0.5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0.5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0.5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0.5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s0.5_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s1_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s1_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s1_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s1_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s1_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s2_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s2_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s2_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s2_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s2_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s3_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s3_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s3_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s3_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s3_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s4_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s4_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s4_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s4_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s4_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.1_s5_r4

# test control
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0.5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0.5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0.5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0.5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s0.5_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s1_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s1_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s1_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s1_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s1_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s2_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s2_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s2_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s2_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s2_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s3_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s3_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s3_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s3_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s3_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s4_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s4_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s4_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s4_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s4_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.1_s5_r4


# beta = 0.9
# test Pareto
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0.5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0.5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0.5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0.5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s0.5_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s1_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s1_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s1_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s1_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s1_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s2_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s2_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s2_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s2_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s2_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s3_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s3_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s3_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s3_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s3_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s4_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s4_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s4_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s4_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s4_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name b0.9_s5_r4
# test control
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0.5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0.5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0.5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0.5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s0.5_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s1_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s1_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s1_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s1_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s1_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s2_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s2_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s2_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s2_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s2_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s3_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s3_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s3_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s3_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s3_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s4_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s4_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s4_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s4_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s4_r4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name b0.9_s5_r4


##### results for sync ########
# F1: policy-0.905982905982906|prediction-0.10277406417112299
# F1: policy-0.9421487603305785|prediction-0.10402219140083217
# F1: policy-0.9333333333333333|prediction-0.09199967245332459
# F1: policy-0.905982905982906|prediction-0.10682789951897381
# F1: policy-0.967741935483871|prediction-0.06544419854568448
# F1: policy-0.976|prediction-0.15603566529492457
# F1: policy-0.9508196721311475|prediction-0.21296865396435158
# F1: policy-0.959349593495935|prediction-0.16108149276466108
# F1: policy-0.959349593495935|prediction-0.18523352952344788
# F1: policy-0.9421487603305785|prediction-0.13644214162348878
# F1: policy-0.9921259842519685|prediction-0.23482952768220203
# F1: policy-0.9421487603305785|prediction-0.21665232358003442
# F1: policy-0.9508196721311475|prediction-0.23211370046014548
# F1: policy-0.9841269841269841|prediction-0.18490046214006395
# F1: policy-0.976|prediction-0.1784057082822623
# F1: policy-0.967741935483871|prediction-0.23550328227571113
# F1: policy-0.9841269841269841|prediction-0.2465730831256414
# F1: policy-0.9841269841269841|prediction-0.1583874961312287
# F1: policy-0.959349593495935|prediction-0.32142857142857145
# F1: policy-0.9333333333333333|prediction-0.2606802497535327
# F1: policy-0.976|prediction-0.2555309734513274
# F1: policy-0.9421487603305785|prediction-0.26556457849961335
# F1: policy-0.9508196721311475|prediction-0.19354293441514536
# F1: policy-0.9841269841269841|prediction-0.29653876069484053
# F1: policy-0.9152542372881356|prediction-0.25057394298832986
# F1: policy-0.9841269841269841|prediction-0.1940824233885171
# F1: policy-0.9421487603305785|prediction-0.3625389408099688
# F1: policy-0.959349593495935|prediction-0.25762446422683816
# F1: policy-0.9841269841269841|prediction-0.2187273649138056
# F1: policy-0.967741935483871|prediction-0.32915508966910834
# F1: policy-0.9841269841269841|prediction-0.3137005995350544
# F1: policy-0.9841269841269841|prediction-0.3256322302932472
# F1: policy-0.9841269841269841|prediction-0.2684729064039409
# F1: policy-0.959349593495935|prediction-0.32954545454545453
# F1: policy-0.9921259842519685|prediction-0.32465277777777773
# discrepancies: policy-2.198395682146693|predict-3.438999207133274
# discrepancies: policy-1.9751193603567245|predict-3.4887265316804483
# discrepancies: policy-2.1928946233226525|predict-4.170364411213295
# discrepancies: policy-1.9441335584154715|predict-3.9056926704435595
# discrepancies: policy-1.7926680241527242|predict-3.7673452798980738
# discrepancies: policy-1.2002508975636907|predict-3.6545711074016443
# discrepancies: policy-1.2470173701426996|predict-3.6827457585101477
# discrepancies: policy-1.3167576703015351|predict-3.880580785733193
# discrepancies: policy-1.643894262273946|predict-3.2361068495007435
# discrepancies: policy-2.024550188478489|predict-3.9032445301455967
# discrepancies: policy-1.4440423574403962|predict-2.7448862212620493
# discrepancies: policy-1.3067755816088273|predict-3.4941441193704947
# discrepancies: policy-1.202556078195259|predict-2.8960557699706353
# discrepancies: policy-1.3293173093829802|predict-3.772301674584499
# discrepancies: policy-1.5022601016790902|predict-3.720229715904541
# discrepancies: policy-1.295359821769448|predict-3.0473571692128734
# discrepancies: policy-1.1868005491570173|predict-3.3165107716309126
# discrepancies: policy-1.1357729814038082|predict-4.4078258232355925
# discrepancies: policy-1.510249704611665|predict-2.727059051194436
# discrepancies: policy-1.3571781063156967|predict-3.424336146354455
# discrepancies: policy-0.9992495382607898|predict-3.657861087788539
# discrepancies: policy-1.6146634556164177|predict-3.412870384039451
# discrepancies: policy-1.1928043388116596|predict-3.5879815294908495
# discrepancies: policy-1.1466383102504094|predict-2.749182329460701
# discrepancies: policy-1.805004032991933|predict-3.094803650171312
# discrepancies: policy-1.1292221513617913|predict-3.5562799673053127
# discrepancies: policy-1.10580591073999|predict-2.422785678722222
# discrepancies: policy-1.9255791049056286|predict-3.099701266240322
# discrepancies: policy-1.1517503945996896|predict-2.566605596848537
# discrepancies: policy-1.2866444000591222|predict-3.041843354145178
# discrepancies: policy-1.0632552416142027|predict-3.1544205216835115
# discrepancies: policy-1.0594650425947063|predict-3.19249020747751
# discrepancies: policy-1.140808479007116|predict-2.9539881994012442
# discrepancies: policy-1.4254320231453843|predict-2.9179426984856143
# discrepancies: policy-1.1838284105529289|predict-2.604142503959996
# F1: policy-0.8363636363636363|prediction-0.11401488306165838
# F1: policy-0.8869565217391304|prediction-0.06293706293706294
# F1: policy-0.8256880733944955|prediction-0.06613858835651726
# F1: policy-0.8869565217391304|prediction-0.033880171184022825
# F1: policy-0.8869565217391304|prediction-0.046883505715841045
# F1: policy-0.967741935483871|prediction-0.12300683371298404
# F1: policy-0.959349593495935|prediction-0.09364060676779463
# F1: policy-0.9152542372881356|prediction-0.06954916745208922
# F1: policy-0.959349593495935|prediction-0.12135010118531366
# F1: policy-0.9421487603305785|prediction-0.12351084107695971
# F1: policy-0.967741935483871|prediction-0.13171389300063438
# F1: policy-0.959349593495935|prediction-0.1544286809815951
# F1: policy-0.976|prediction-0.12202493388741971
# F1: policy-0.9841269841269841|prediction-0.17401529804818006
# F1: policy-0.9152542372881356|prediction-0.1417162370566204
# F1: policy-0.967741935483871|prediction-0.1890715667311412
# F1: policy-0.9508196721311475|prediction-0.2254134029590949
# F1: policy-0.9841269841269841|prediction-0.14409914909359972
# F1: policy-0.976|prediction-0.16618697895990472
# F1: policy-0.9841269841269841|prediction-0.1251892505677517
# F1: policy-0.967741935483871|prediction-0.15639898076606937
# F1: policy-0.9921259842519685|prediction-0.17254358161648178
# F1: policy-0.9841269841269841|prediction-0.20567158067158064
# F1: policy-1.0|prediction-0.18779281624154084
# F1: policy-0.9841269841269841|prediction-0.19427660983280914
# F1: policy-0.9921259842519685|prediction-0.12568278373457414
# F1: policy-0.9921259842519685|prediction-0.1602900213009995
# F1: policy-0.9921259842519685|prediction-0.2142174051641696
# F1: policy-1.0|prediction-0.1905237106817819
# F1: policy-0.967741935483871|prediction-0.15531591193842847
# F1: policy-1.0|prediction-0.24814939434724093
# F1: policy-0.9921259842519685|prediction-0.22638213851761851
# F1: policy-0.9333333333333333|prediction-0.31564525024127943
# F1: policy-0.9508196721311475|prediction-0.2520775623268698
# F1: policy-0.976|prediction-0.239603088593877
# discrepancies: policy-2.18803191185167|predict-3.083992250902856
# discrepancies: policy-2.4123982394632537|predict-2.6307602935571572
# discrepancies: policy-2.9381877029398544|predict-3.925840281943288
# discrepancies: policy-2.7592523182136226|predict-4.393480619064561
# discrepancies: policy-2.8980442828221413|predict-3.8719202408398945
# discrepancies: policy-3.932906628110658|predict-2.5920473247335307
# discrepancies: policy-1.475150963804622|predict-3.0281621021308114
# discrepancies: policy-1.2368947116997593|predict-2.8226764983708335
# discrepancies: policy-1.3668583711675668|predict-2.3398711034770643
# discrepancies: policy-1.4858346753412668|predict-2.1937255531929454
# discrepancies: policy-1.1224008663894234|predict-3.031541940491308
# discrepancies: policy-0.995913667633431|predict-1.8646888576458107
# discrepancies: policy-0.7772053076607118|predict-2.9780060393353667
# discrepancies: policy-0.8397712380644397|predict-2.995424740356595
# discrepancies: policy-1.3753095775653434|predict-2.1832331916688643
# discrepancies: policy-1.074413287942133|predict-1.9336615174278415
# discrepancies: policy-1.2116089573744437|predict-1.981584124841084
# discrepancies: policy-0.69768915173029|predict-2.0214324618327395
# discrepancies: policy-0.8959711222800522|predict-1.7669511321956315
# discrepancies: policy-1.1027090870434413|predict-1.9000820942794663
# discrepancies: policy-0.7093533754432028|predict-2.120317580153357
# discrepancies: policy-0.8660128569970627|predict-1.8030165165687055
# discrepancies: policy-0.7242279985779025|predict-1.9980835427470616
# discrepancies: policy-0.7682203586122265|predict-1.8369951651122782
# discrepancies: policy-0.8865405867216846|predict-2.0889381918865024
# discrepancies: policy-0.8769970330462377|predict-1.9189648018485699
# discrepancies: policy-0.7379810096407717|predict-2.338220047954303
# discrepancies: policy-0.6679006103404409|predict-1.8142904207153965
# discrepancies: policy-0.6741293429807003|predict-1.8330340950065236
# discrepancies: policy-0.9743548193187075|predict-1.9842388584936606
# discrepancies: policy-0.7278972750608578|predict-1.7506578610702348
# discrepancies: policy-0.5781577539902087|predict-1.6556504380782044
# discrepancies: policy-1.0616289638608405|predict-1.5814853263890072
# discrepancies: policy-1.1266509008472119|predict-1.8831749845606638
# discrepancies: policy-0.8231389199994242|predict-1.619781369259766
