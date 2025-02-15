#!/bin/bash

# 在做回归拟合的时候，注意取消dataloader中对PKM2，VDR的采样，取全集训练打分函数
# nohup python regress.py --task lit-smi --name ALDH1  --dim 2048 --out_path ./models/latest/ALDH1.pth --num 40000 >>log1/ALDH1.log &
# nohup python regress.py --task lit-smi --name VDR  --dim 2048 --out_path ./models/latest/VDR.pth --num 40000 >>log1/VDR.log &
# nohup python regress.py --task lit-smi --name PKM2  --dim 2048 --out_path ./models/latest/PKM2.pth --num 40000 >>log1/PKM2.log &

# wait

nohup python aso_abl.py --task lit-smi --name ALDH1  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest/ALDH1.pth --strategy ranking --seed 1 >>log1/ALDH1.log &
nohup python aso_abl.py --task lit-smi --name VDR  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest/VDR.pth --strategy ranking --seed 1 >>log1/VDR.log &
nohup python aso_abl.py --task lit-smi --name PKM2  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest/PKM2.pth --strategy ranking --seed 1 >>log1/PKM2.log &
 