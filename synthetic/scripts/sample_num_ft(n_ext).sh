# mem_size | batch_size | weight_num | update_freq | lr | beta
# NAIVE Linear Ext:
# s0:  4000 | 256 | 1   | 100 | 1e-3
# s0.5 4000 | 256 | 4   | 100 | 1e-3
# s1:  4000 | 256 | 8   | 100 | 1e-3
# s2:  4000 | 256 | 16  | 100 | 1e-3
# # s3:  4000 | 256 | 32  | 100 | 1e-3
# # s4:  4000 | 256 | 64  | 100 | 1e-3
# # s5:  4000 | 256 | 128 | 100 | 1e-3
#
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r0
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r1
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r2
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r3
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r4
#
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r0
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r1
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r2
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r3
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r4
#
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r0
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r1
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r2
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r3
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r4
#
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r0
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r1
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r2
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r3
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r4
#
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r0
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r1
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r2
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r3
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r4
#
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r0
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r1
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r2
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r3
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r4
#
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r0
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r1
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r2
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r3
# python train.py --env-name ft --method crl-naive --model linear_ext --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0.5_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0.5_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0.5_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0.5_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s0.5_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s1_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s1_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s1_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s1_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s1_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s2_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s2_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s2_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s2_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s2_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s3_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s3_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s3_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s3_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s3_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s4_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s4_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s4_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s4_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s4_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s5_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s5_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s5_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s5_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltpareto --name s5_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0.5_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0.5_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0.5_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0.5_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0.5_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r4

python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear_ext --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r4
