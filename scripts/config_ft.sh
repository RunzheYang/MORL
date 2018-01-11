# mem_size | batch_size | weight_num | update_freq | lr |
# NAIVE:
# 0: 4000 | 256 | 32 | 100 | 1e-3 |
# 1: 8000 | 256 | 32 | 100 | 1e-3 |
# 2: 4000 | 512 | 16 | 100 | 1e-3 |
# 3: 4000 | 128 | 64 | 100 | 1e-3 |
# 4: 4000 | 256 | 32 | 50  | 1e-3 |
# 5: 4000 | 256 | 32 | 100 | 5e-4 |
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name 0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 0
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 0
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name 1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 1
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 1
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 512 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name 2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 2
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 2
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name 3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 3
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 3
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 50  --name 4
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 4
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 4
# python train.py --env-name ft --method crl-naive --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  5e-4 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/naive/saved/ --log crl/naive/logs/ --update-freq 100 --name 5
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltpareto  --name 5
python test/eval_ft.py --env-name ft --method crl-naive --model  linear --gamma  0.99 --save crl/naive/saved/ --pltcontrol --name 5

# ENEVELOPE:
# 0: 4000 | 256 | 32 | 100 | 1e-3 | 0.01
# 1: 8000 | 256 | 32 | 100 | 1e-3 | 0.01
# 2: 4000 | 512 | 16 | 100 | 1e-3 | 0.01
# 3: 4000 | 128 | 64 | 100 | 1e-3 | 0.01
# 4: 4000 | 256 | 32 | 50  | 1e-3 | 0.01
# 5: 4000 | 256 | 32 | 100 | 5e-4 | 0.01
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --name 0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --name 1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 512 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --name 2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --name 3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 3
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.01 --name 4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  5e-4 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --name 5
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5
