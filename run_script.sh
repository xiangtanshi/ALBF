#!/bin/bash
# akt1, aofb, cp3a4, cxcr4, gria2, hivpr, hivrt, kif11

# m1, m2
# nohup python ac.py --task dude-smi --name akt1 --model_path ./models/C-model-2048/akt1.pth --lr 0.0003 --rounds 10 --bs 5 --epoch 20 --dim 2048 --out_dim 2 --strategy greedy --seed 2042 >>log1/akt1.log &
# nohup python ac.py --task dude-smi --name aofb --model_path ./models/C-model-2048/aofb.pth --lr 0.0003 --rounds 10 --bs 5 --epoch 20 --dim 2048 --out_dim 2 --strategy greedy --seed 2042 >>log1/aofb.log &
# nohup python ac.py --task dude-smi --name cp3a4 --model_path ./models/C-model-2048/cp3a4.pth --lr 0.0003 --rounds 10 --bs 5 --epoch 20 --dim 2048 --out_dim 2 --strategy greedy --seed 2042  >>log1/cp3a4.log &
# nohup python ac.py --task dude-smi --name gria2 --model_path ./models/C-model-2048/gria2.pth --lr 0.0003 --rounds 10 --bs 5 --epoch 20 --dim 2048 --out_dim 2 --strategy greedy --seed 2042 >>log1/gria2.log &
# nohup python ac.py --task dude-smi --name cxcr4 --model_path ./models/C-model-2048/cxcr4.pth --lr 0.0003 --rounds 10 --bs 5 --epoch 20 --dim 2048 --out_dim 2 --strategy greedy --seed 2042 >>log1/cxcr4.log &
# nohup python ac.py --task dude-smi --name hivpr --model_path ./models/C-model-2048/hivpr.pth --lr 0.0003 --rounds 10 --bs 5 --epoch 20 --dim 2048 --out_dim 2 --strategy greedy --seed 2042 >>log1/hivpr.log &
# nohup python ac.py --task dude-smi --name hivrt --model_path ./models/C-model-2048/hivrt.pth --lr 0.0003 --rounds 10 --bs 5 --epoch 20 --dim 2048 --out_dim 2 --strategy greedy --seed 2042 >>log1/hivrt.log &
# nohup python ac.py --task dude-smi --name kif11 --model_path ./models/C-model-2048/kif11.pth --lr 0.0003 --rounds 10 --bs 5 --epoch 20 --dim 2048 --out_dim 2 --strategy greedy --seed 2042 >>log1/kif11.log &

# wait
# # m3

# nohup python regress.py --task dude-smi --name akt1  --dim 2048 --out_path ./models/latest/akt1.pth --num 7000 >>log1/akt1.log &
# nohup python regress.py --task dude-smi --name aofb  --dim 2048 --out_path ./models/latest/aofb.pth --num 4000 >>log1/aofb.log &
# nohup python regress.py --task dude-smi --name cp3a4  --dim 2048 --out_path ./models/latest/cp3a4.pth --num 4000 >>log1/cp3a4.log &
# nohup python regress.py --task dude-smi --name gria2  --dim 2048 --out_path ./models/latest/gria2.pth --num 3000 >>log1/gria2.log &
# nohup python regress.py --task dude-smi --name cxcr4  --dim 2048 --out_path ./models/latest/cxcr4.pth --num 4000 >>log1/cxcr4.log &
# nohup python regress.py --task dude-smi --name hivpr  --dim 2048 --out_path ./models/latest/hivpr.pth --num 4000 >>log1/hivpr.log &
# nohup python regress.py --task dude-smi --name hivrt  --dim 2048 --out_path ./models/latest/hivrt.pth --num 4000 >>log1/hivrt.log &
# nohup python regress.py --task dude-smi --name kif11  --dim 2048 --out_path ./models/latest/kif11.pth --num 4000 >>log1/kif11.log &


# wait

# nohup python aso.py --task dude-smi --name akt1  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/akt1.pth --strategy ranking --seed 0 >>log1/akt1.log &
# nohup python aso.py --task dude-smi --name aofb  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/aofb.pth --strategy ranking --seed 0 >>log1/aofb.log &
# nohup python aso.py --task dude-smi --name cp3a4  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/cp3a4.pth --strategy ranking --seed 0 >>log1/cp3a4.log &
# nohup python aso.py --task dude-smi --name gria2  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/gria2.pth --strategy ranking --seed 0 >>log1/gria2.log &
# nohup python aso.py --task dude-smi --name cxcr4  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/cxcr4.pth --strategy ranking --seed 0 >>log1/cxcr4.log &
# nohup python aso.py --task dude-smi --name hivpr  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/hivpr.pth --strategy ranking --seed 0 >>log1/hivpr.log &
# nohup python aso.py --task dude-smi --name hivrt  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/hivrt.pth --strategy ranking --seed 0 >>log1/hivrt.log &
# nohup python aso.py --task dude-smi --name kif11  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/kif11.pth --strategy ranking --seed 0 >>log1/kif11.log &

nohup python aso.py --task lit-smi --name VDR  --lr 0.01 --rounds 10 --bs 100 --model_path ./models/latest/VDR.pth --strategy ranking --seed 0 >>log1/VDR.log &
nohup python aso.py --task lit-smi --name PKM2  --lr 0.01 --rounds 10 --bs 100 --model_path ./models/latest/PKM2.pth --strategy ranking --seed 0 >>log1/PKM2.log &

# nohup python aso.py --task dude-smi --name akt1  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/akt1.pth --strategy random --seed 2 >>log1/akt1.log &
# nohup python aso.py --task dude-smi --name aofb  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/aofb.pth --strategy random --seed 2 >>log1/aofb.log &
# nohup python aso.py --task dude-smi --name cp3a4  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/cp3a4.pth --strategy random --seed 2 >>log1/cp3a4.log &
# nohup python aso.py --task dude-smi --name gria2  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/gria2.pth --strategy random --seed 2 >>log1/gria2.log &
# nohup python aso.py --task dude-smi --name cxcr4  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/cxcr4.pth --strategy random --seed 2 >>log1/cxcr4.log &
# nohup python aso.py --task dude-smi --name hivpr  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/hivpr.pth --strategy random --seed 2 >>log1/hivpr.log &
# nohup python aso.py --task dude-smi --name hivrt  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/hivrt.pth --strategy random --seed 2 >>log1/hivrt.log &
# nohup python aso.py --task dude-smi --name kif11  --lr 0.01 --rounds 10 --bs 5 --model_path ./models/latest/kif11.pth --strategy random --seed 2 >>log1/kif11.log &