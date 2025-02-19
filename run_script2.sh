#!/bin/bash

# 做回归拟合的时候注意将regress.py中的函数改为random_score_regression
# nohup python regress.py --task dude-smi --name aofb  --dim 2048 --out_path ./models/latest-abl/aofb.pth --num 4000 --seed 1 >>log2/aofb.log &
# nohup python regress.py --task dude-smi --name cp3a4  --dim 2048 --out_path ./models/latest-abl/cp3a4.pth --num 4000 --seed 1 >>log2/cp3a4.log &
# nohup python regress.py --task dude-smi --name cxcr4  --dim 2048 --out_path ./models/latest-abl/cxcr4.pth --num 4000  --seed 1  >>log2/cxcr4.log &

# nohup python regress.py --task lit-smi --name ALDH1  --dim 2048 --out_path ./models/latest-abl/ALDH1.pth --num 40000 >>log2/ALDH1.log &
# nohup python regress.py --task lit-smi --name VDR  --dim 2048 --out_path ./models/latest-abl/VDR.pth --num 4000 >>log2/VDR.log &
# nohup python regress.py --task lit-smi --name PKM2  --dim 2048 --out_path ./models/latest-abl/PKM2.pth --num 4000 >>log2/PKM2.log &

# wait


# nohup python aso.py --task dude-smi --name aofb  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest-abl/aofb.pth --strategy ranking >>log2/aofb.log &
# nohup python aso.py --task dude-smi --name cp3a4  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest-abl/cp3a4.pth --strategy ranking >>log2/cp3a4.log &
# nohup python aso.py --task dude-smi --name cxcr4  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest-abl/cxcr4.pth --strategy ranking >>log2/cxcr4.log &

# nohup python aso_abl.py --task lit-smi --name ALDH1  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest-abl/ALDH1.pth --strategy ranking --seed 2 >>log2/ALDH1.log &
nohup python aso_abl.py --task lit-smi --name VDR  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest-abl/VDR.pth --strategy greedy --seed 2 >>log2/VDR.log &
nohup python aso_abl.py --task lit-smi --name PKM2  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest-abl/PKM2.pth --strategy greedy --seed 2 >>log2/PKM2.log &