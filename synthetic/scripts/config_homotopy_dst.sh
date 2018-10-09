# mem_size | batch_size | weight_num | update_freq | lr | beta
# # ENEVELOPE:
# # 0+h: 4000 | 256 | 32 | 100 | 1e-3 | 0.95 | homotopy
# # 1+h: 8000 | 256 | 32 | 100 | 1e-3 | 0.95 | homotopy
# # 2+h: 4000 | 512 | 16 | 100 | 1e-3 | 0.95 | homotopy
# # 3+h: 4000 | 128 | 64 | 100 | 1e-3 | 0.95 | homotopy
# # 4+h: 4000 | 256 | 32 | 50  | 1e-3 | 0.95 | homotopy
# # 5+h: 4000 | 256 | 32 | 100 | 5e-4 | 0.95 | homotopy
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.95 --homotopy --name b0.95_0+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.95 --homotopy --name b0.95_1+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 512 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.95 --homotopy --name b0.95_2+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.95 --homotopy --name b0.95_3+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 3+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.95 --homotopy --name b0.95_4+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  5e-4 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.95 --homotopy --name b0.95_5+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5+h



#Control eval
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 1 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 2 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 3 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 4 --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 5 --pltcontrol

# #Pareto eval
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 0 --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 1 --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 2 --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 3 --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 4 --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name 5 --pltpareto



# # # ENEVELOPE:
# # # 0+h: 4000 | 256 | 32 | 100 | 1e-3 | 0.85 | homotopy
# # # 1+h: 8000 | 256 | 32 | 100 | 1e-3 | 0.85 | homotopy
# # # 2+h: 4000 | 512 | 16 | 100 | 1e-3 | 0.85 | homotopy
# # # 3+h: 4000 | 128 | 64 | 100 | 1e-3 | 0.85 | homotopy
# # # 4+h: 4000 | 256 | 32 | 50  | 1e-3 | 0.85 | homotopy
# # # 5+h: 4000 | 256 | 32 | 100 | 5e-4 | 0.85 | homotopy
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.85 --homotopy --name b0.85_0+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.85 --homotopy --name b0.85_1+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 512 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.85 --homotopy --name b0.85_2+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.85 --homotopy --name b0.85_3+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 3+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.85 --homotopy --name b0.85_5+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4+h
#  python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  5e-4 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.85 --homotopy --name b0.85_6+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5+h
# # python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5+h
#
#
# #Pareto eval
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_0+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_1+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_2+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_3+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_4+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_5+h --pltpareto
#
# #Control eval
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_0+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_1+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_2+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_3+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_4+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.85_5+h --pltcontrol






# # ENEVELOPE:
# # 0+h: 4000 | 256 | 32 | 100 | 1e-3 | 0.98 | homotopy
# # 1+h: 8000 | 256 | 32 | 100 | 1e-3 | 0.98 | homotopy
# # 2+h: 4000 | 512 | 16 | 100 | 1e-3 | 0.98 | homotopy
# # 3+h: 4000 | 128 | 64 | 100 | 1e-3 | 0.98 | homotopy
# # 4+h: 4000 | 256 | 32 | 50  | 1e-3 | 0.98 | homotopy
# # 5+h: 4000 | 256 | 32 | 100 | 5e-4 | 0.98 | homotopy
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name b0.98_0+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name b0.98_1+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 512 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name b0.98_2+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name b0.98_3+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 3+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.98 --homotopy --name b0.98_4+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  5e-4 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.98 --homotopy --name b0.98_5+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5+h

# # ENEVELOPE:
# # 0+h: 4000 | 256 | 32 | 100 | 1e-3 | 0.99 | homotopy
# # 1+h: 8000 | 256 | 32 | 100 | 1e-3 | 0.99 | homotopy
# # 2+h: 4000 | 512 | 16 | 100 | 1e-3 | 0.99 | homotopy
# # 3+h: 4000 | 128 | 64 | 100 | 1e-3 | 0.99 | homotopy
# # 4+h: 4000 | 256 | 32 | 50  | 1e-3 | 0.99 | homotopy
# # 5+h: 4000 | 256 | 32 | 100 | 5e-4 | 0.99 | homotopy
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.99 --homotopy --name b0.99_0+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 8000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.99 --homotopy --name b0.99_1+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 512 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.99 --homotopy --name b0.99_2+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 128 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.99 --homotopy --name b0.99_3+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 3+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 50  --beta 0.99 --homotopy --name b0.99_4+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4+h
 python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  5e-4 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.99 --homotopy --name b0.99_5+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5+h
# python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5+h98





# #Pareto eval
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_0+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_1+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_2+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_3+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_4+h --pltpareto
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_5+h --pltpareto
#
# #Control eval
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_0+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_1+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_2+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_3+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_4+h --pltcontrol
# python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.98_5+h --pltcontrol


#Pareto eval
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_0+h --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_1+h --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_2+h --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_3+h --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_4+h --pltpareto
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_5+h --pltpareto

#Control eval
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_0+h --pltcontrol
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_1+h --pltcontrol
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_2+h --pltcontrol
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_3+h --pltcontrol
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_4+h --pltcontrol
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --name b0.99_5+h --pltcontrol



# ENEVELOPE:beta 0.96
# s0: 8000 | 256 | 1   | 100 | 1e-3 |
# s1: 8000 | 256 | 8   | 100 | 1e-3 |
# s2: 8000 | 256 | 16  | 100 | 1e-3 |
# s3: 8000 | 256 | 32  | 100 | 1e-3 |
# s4: 8000 | 256 | 64  | 100 | 1e-3 |
# s5: 8000 | 256 | 128 | 100 | 1e-3 |
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r5_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r6_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r7_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r8_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r9_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r10_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r11_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r12_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r13_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 1 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s0_r14_nh
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 0
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 0
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r5_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r6_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r7_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r8_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r9_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r10_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r11_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r12_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r13_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s1_r14_nh
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 1
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 1
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r5_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r6_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r7_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r8_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r9_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r10_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r11_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r12_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r13_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s2_r14_nh
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 2
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r5_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r6_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r7_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r8_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r9_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r10_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r11_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r12_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r13_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s3_r14_nh
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 2
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 3
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r5_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r6_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r7_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r8_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r9_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r10_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r11_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r12_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r13_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 64 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s4_r14_nh
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 4
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 4
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r5_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r6_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r7_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r8_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r9_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r10_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r11_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r12_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r13_nh
python train.py --env-name dst --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 128 --episode-num 2000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.96 --name s5_r14_nh
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name 5
#python test/eval_ft.py --env-name ft --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name 5

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s0_r14_nh

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s0_r14_nh


python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s1_r14_nh

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s1_r14_nh



python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s2_r14_nh

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s2_r14_nh



python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s3_r14_nh

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s3_r14_nh



python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s4_r14_nh

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s4_r14_nh



python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltpareto  --name s5_r14_nh

python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r5_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r6_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r7_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r8_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r9_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r10_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r11_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r12_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r13_nh
python test/eval_dst.py --env-name dst --method crl-envelope --model  linear --gamma  0.99 --save crl/envelope/saved/ --pltcontrol --name s5_r14_nh
