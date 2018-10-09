# mem_size | batch_size | weight_num | update_freq | lr | beta
# NAIVE:
# s0.5:  4000 | 256 | 4   | 100 | 1e-3 | 0.01

# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r0
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r1
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r2
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r3
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name s0.5_r4
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 0
#python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 0


# ENEVELOPE:
# s0.5: 4000 | 256 | 4   | 100 | 1e-3 | homotopy
# s1: 4000 | 256 | 8   | 100 | 1e-3 | homotopy
# s2: 4000 | 256 | 16  | 100 | 1e-3 | homotopy
# s3: 4000 | 256 | 32  | 100 | 1e-3 | homotopy
# s4: 4000 | 256 | 64  | 100 | 1e-3 | homotopy
# s5: 4000 | 256 | 128 | 100 | 1e-3 | homotopy
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name s0.5_r0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name s0.5_r1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name s0.5_r2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name s0.5_r3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 4 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name s0.5_r4
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0

# test naive
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0.5_r0
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0.5_r1
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0.5_r2
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0.5_r3
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name s0.5_r4
#
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol  --name s0.5_r0
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol  --name s0.5_r1
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol  --name s0.5_r2
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol  --name s0.5_r3
# python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol  --name s0.5_r4


# test envelope homotopy Pareto
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0.5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0.5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0.5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0.5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0.5_r4

python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name s0.5_r0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name s0.5_r1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name s0.5_r2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name s0.5_r3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol  --name s0.5_r4
