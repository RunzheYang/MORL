# mem_size | batch_size | weight_num | update_freq | lr | beta
# ENEVELOPE:
# 0: 4000 | 256 | 32 | 100 | 1e-3 | 0.65
# 1: 4000 | 256 | 32 | 100 | 1e-3 | 0.70
# 2: 4000 | 256 | 32 | 100 | 1e-3 | 0.75
# 3: 4000 | 256 | 32 | 100 | 1e-3 | 0.80
# 4: 4000 | 256 | 32 | 100 | 1e-3 | 0.85
# 5: 4000 | 256 | 32 | 100 | 1e-3 | 0.90
# 6: 4000 | 256 | 32 | 100 | 1e-3 | 0.95
# 7: 4000 | 256 | 32 | 100 | 1e-3 | 1.00
#
# python train.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.65 --homotopy --name 0+1
# python train.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.70 --homotopy --name 0+2
# python train.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.75 --homotopy --name 0+3
# python train.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.80 --homotopy --name 0+4
# python train.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.85 --homotopy --name 0+5
# python train.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.90 --homotopy --name 0+6
# python train.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.95 --homotopy --name 0+7
# python train.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 1.00 --homotopy --name 0+8
#

#Control eval
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+0 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+1 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+2 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+3 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+4 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+5 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+6 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+7 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+8 --pltcontrol

#Pareto eval
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+0 --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+1 --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+2 --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+3 --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+4 --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+5 --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+6 --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+7 --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0+8 --pltpareto
