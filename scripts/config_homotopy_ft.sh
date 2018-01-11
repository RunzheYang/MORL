# # ENEVELOPE:
# # 0+h: 4000 | 256 | 32 | 100 | 1e-3 | 0.01
# # 1+h: 8000 | 256 | 32 | 100 | 1e-3 | 0.01
# # 2+h: 4000 | 512 | 16 | 100 | 1e-3 | 0.01
# # 3+h: 4000 | 128 | 64 | 100 | 1e-3 | 0.01
# # 4+h: 4000 | 256 | 32 | 50  | 1e-3 | 0.01
# # 5+h: 4000 | 256 | 32 | 100 | 5e-4 | 0.01
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --homotopy --name 0+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0+h
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --homotopy --name 1+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1+h
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 512 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --homotopy --name 2+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2+h
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --homotopy --name 3+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 3+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3+h
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.01 --homotopy --name 4+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4+h
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  5e-4 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --homotopy --name 5+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5+h

# ENEVELOPE:
# 0+he: 4000 | 256 | 32 | 100 | 1e-3 | 0.00
# 1+he: 8000 | 256 | 32 | 100 | 1e-3 | 0.00
# 2+he: 4000 | 512 | 16 | 100 | 1e-3 | 0.00
# 3+he: 4000 | 128 | 64 | 100 | 1e-3 | 0.00
# 4+he: 4000 | 256 | 32 | 50  | 1e-3 | 0.00
# 5+he: 4000 | 256 | 32 | 100 | 5e-4 | 0.00
# s+he: 8000 | 128 | 64 | 50  | 5e-4 | 0.00
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 0+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0+he
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 1+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1+he
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 512 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 2+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2+he
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 3+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 3+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3+he
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.00 --homotopy --name 4+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4+he
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  5e-4 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 5+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5+he
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.00 --homotopy --name m+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name m+he
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name m+he

# ENEVELOPE:
# 0+_e: 4000 | 256 | 32 | 100 | 1e-3 | 0.00
# 1+_e: 8000 | 256 | 32 | 100 | 1e-3 | 0.00
# 2+_e: 4000 | 512 | 16 | 100 | 1e-3 | 0.00
# 3+_e: 4000 | 128 | 64 | 100 | 1e-3 | 0.00
# 4+_e: 4000 | 256 | 32 | 50  | 1e-3 | 0.00
# 5+_e: 4000 | 256 | 32 | 100 | 5e-4 | 0.00
# s+_e: 8000 | 128 | 64 | 50  | 5e-4 | 0.00
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 0+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0+_e
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 1+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1+_e
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 512 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 2+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2+_e
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 3+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 3+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3+_e
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.00 --homotopy --name 4+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4+_e
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  5e-4 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.00 --homotopy --name 5+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5+_e
# python train.py --env-name ft --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.00 --homotopy --name m+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name m+_e
python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name m+_e