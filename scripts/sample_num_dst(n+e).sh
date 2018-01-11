# mem_size | batch_size | weight_num | update_freq | lr | beta
# NAIVE:

python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r5
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r6
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r7
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r8
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r9
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r10
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r11
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r12
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r13
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0_r14
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 0
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 0

python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r5
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r6
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r7
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r8
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r9
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r10
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r11
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r12
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r13
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s1_r14
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 1
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 12
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r5
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r6
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r7
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r8
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r9
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r10
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r11
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r12
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r13
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s2_r14
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 2
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 2

python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r5
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r6
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r7
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r8
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r9
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r10
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r11
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r12
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r13
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s3_r14
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 3
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 3

python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r5
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r6
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r7
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r8
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r9
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r10
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r11
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r12
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r13
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s4_r14
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 4
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 4

python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r5
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r6
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r7
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r8
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r9
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r10
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r11
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r12
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r13
python train.py --env-name dst --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s5_r14
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 5
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 5



# #eval
# # #naive
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0_r4
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s0_r4
#
#
#
#
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s1_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s1_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s1_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s1_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s1_r4
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s1_r4
#
# #
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s2_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s2_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s2_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s2_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s2_r4
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s2_r4
#
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s3_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s3_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s3_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s3_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s3_r4
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s3_r4
#
#
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s4_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s4_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s4_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s4_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s4_r4
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s4_r4
#
#
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s5_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s5_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s5_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s5_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s5_r4
#
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r0
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r1
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r2
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r3
# python test/eval_dst.py --env-name dst --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name s5_r4
#

# ENEVELOPE:beta 0.98
# s0: 8000 | 256 | 1   | 100 | 1e-3 | homotopy
# s1: 8000 | 256 | 8   | 100 | 1e-3 | homotopy
# s2: 8000 | 256 | 16  | 100 | 1e-3 | homotopy
# s3: 8000 | 256 | 32  | 100 | 1e-3 | homotopy
# s4: 8000 | 256 | 64  | 100 | 1e-3 | homotopy
# s5: 8000 | 256 | 128 | 100 | 1e-3 | homotopy
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r5
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r6
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r7
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r8
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r9
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r10
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r11
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r12
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r13
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s0_r14
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r5
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r6
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r7
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r8
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r9
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r10
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r11
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r12
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r13
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s1_r14
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r5
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r6
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r7
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r8
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r9
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r10
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r11
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r12
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r13
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s2_r14
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r5
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r6
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r7
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r8
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r9
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r10
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r11
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r12
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r13
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s3_r14
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r5
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r6
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r7
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r8
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r9
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r10
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r11
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r12
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r13
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s4_r14
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r5
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r6
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r7
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r8
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r9
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r10
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r11
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r12
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r13
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name s5_r14
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r14

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r14



python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r14

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r14



python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r14

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r14



python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r14

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r14



python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r14

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r14



python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r14

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r5
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r6
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r7
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r8
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r9
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r10
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r11
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r12
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r13
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r14
