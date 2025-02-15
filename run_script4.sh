#!/bin/bash

# 在做回归拟合的时候，注意取消dataloader中对PKM2，VDR的采样，取全集训练打分函数
# nohup python regress.py --task lit-pkl --clip --name ALDH1  --dim 128 --out_path ./models/latest-pkl-clip/ALDH1.pth --num 40000 >>log4/ALDH1.log &
# nohup python regress.py --task lit-pkl --clip --name VDR  --dim 128 --out_path ./models/latest-pkl-clip/VDR.pth --num 40000 >>log4/VDR.log &
# nohup python regress.py --task lit-pkl --clip --name PKM2  --dim 128 --out_path ./models/latest-pkl-clip/PKM2.pth --num 40000 >>log4/PKM2.log &

# nohup python regress.py --task lit-pkl --clip --name ESR_antago  --dim 128 --out_path ./models/latest-pkl-clip/ESR_antago.pth --num 4000 >>log4/ESR_antago.log &
# nohup python regress.py --task lit-pkl --clip --name MAPK1  --dim 128 --out_path ./models/latest-pkl-clip/MAPK1.pth --num 4000 >>log4/MAPK1.log &
# nohup python regress.py --task lit-pkl --clip --name MTORC1  --dim 128 --out_path ./models/latest-pkl-clip/MTORC1.pth --num 4000 >>log4/MTORC1.log &
# nohup python regress.py --task lit-pkl --clip --name TP53  --dim 128 --out_path ./models/latest-pkl-clip/TP53.pth --num 4000 >>log4/TP53.log &

# wait


# nohup python aso_abl.py --task lit-pkl --clip --name ALDH1 --dim 128  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest-pkl-clip/ALDH1.pth --strategy ranking --seed 1 >>log4/ALDH1.log &
nohup python aso_abl.py --task lit-pkl --clip --name VDR --dim 128  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest-pkl-clip/VDR.pth --strategy ranking --seed 1 >>log4/VDR.log &
nohup python aso_abl.py --task lit-pkl --clip --name PKM2 --dim 128  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest-pkl-clip/PKM2.pth --strategy ranking --seed 1 >>log4/PKM2.log &
 

nohup python aso_abl.py --task lit-pkl --clip --name ESR_antago --model_path ./models/latest-pkl-clip/ESR_antago.pth --dim 128  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 1 >>log4/ESR_antago.log &
nohup python aso_abl.py --task lit-pkl --clip --name MAPK1 --model_path ./models/latest-pkl-clip/MAPK1.pth --dim 128  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 1 >>log4/MAPK1.log &
nohup python aso_abl.py --task lit-pkl --clip --name MTORC1 --model_path ./models/latest-pkl-clip/MTORC1.pth --dim 128  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 1 >>log4/MTORC1.log &
nohup python aso_abl.py --task lit-pkl --clip --name TP53 --model_path ./models/latest-pkl-clip/TP53.pth --dim 128 --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 1 >>log4/TP53.log &

# wait