#!/bin/bash

# 在做回归拟合的时候，注意取消dataloader中对PKM2，VDR的采样，取全集训练打分函数
# nohup python regress.py --task lit-pkl --name ALDH1  --dim 2048 --out_path ./models/latest-pkl/ALDH1.pth --num 40000 >>log3/ALDH1.log &
# nohup python regress.py --task lit-pkl --name VDR  --dim 2048 --out_path ./models/latest-pkl/VDR.pth --num 40000 >>log3/VDR.log &
# nohup python regress.py --task lit-pkl --name PKM2  --dim 2048 --out_path ./models/latest-pkl/PKM2.pth --num 40000 >>log3/PKM2.log &

# nohup python regress.py --task lit-pkl --name ESR_antago  --dim 2048 --out_path ./models/latest-pkl/ESR_antago.pth --num 2000 >>log3/ESR_antago-.log &
# nohup python regress.py --task lit-pkl --name MAPK1  --dim 2048 --out_path ./models/latest-pkl/MAPK1.pth --num 10000 >>log3/MAPK1-.log &
# nohup python regress.py --task lit-pkl --name MTORC1  --dim 2048 --out_path ./models/latest-pkl/MTORC1.pth --num 10000 >>log3/MTORC1-.log &
# nohup python regress.py --task lit-pkl --name TP53  --dim 2048 --out_path ./models/latest-pkl/TP53.pth --num 2000 >>log3/TP53-.log &

# uniform score
# nohup python regress.py --task lit-pkl --name ESR_antago  --dim 2048 --out_path ./models/latest-pkl-abl/ESR_antago.pth --num 2000 >>log3/ESR_antago-.log &
# nohup python regress.py --task lit-pkl --name MAPK1  --dim 2048 --out_path ./models/latest-pkl-abl/MAPK1.pth --num 10000 >>log3/MAPK1-.log &
# nohup python regress.py --task lit-pkl --name MTORC1  --dim 2048 --out_path ./models/latest-pkl-abl/MTORC1.pth --num 10000 >>log3/MTORC1-.log &
# nohup python regress.py --task lit-pkl --name TP53  --dim 2048 --out_path ./models/latest-pkl-abl/TP53.pth --num 2000 >>log3/TP53-.log &

nohup python regress.py --task dude-z --name ital  --dim 2048 --out_path ./models/latest-pkl-abl/z-ital.pth --num 1000 >>log5/ital.log &

# wait

# 注意data_loader里面是用sample_selection选子集
# nohup python aso_abl.py --task lit-pkl --name ALDH1  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest-pkl/ALDH1.pth --strategy ranking --seed 2 >>log3/ALDH1.log &
# nohup python aso_abl.py --task lit-pkl --name VDR  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest-pkl/VDR.pth --strategy ranking --seed 1 >>log3/VDR.log &
# nohup python aso_abl.py --task lit-pkl --name PKM2  --lr 0.01 --rounds 10 --bs 20 --model_path ./models/latest-pkl/PKM2.pth --strategy ranking --seed 1 >>log3/PKM2.log &
 

# nohup python aso_abl.py --task lit-pkl --name ESR_antago --model_path ./models/latest-pkl/ESR_antago.pth  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 1 >>log3/ESR_antago-.log &
# nohup python aso_abl.py --task lit-pkl --name MAPK1 --model_path ./models/latest-pkl/MAPK1.pth  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 1 >>log3/MAPK1-.log &
# nohup python aso_abl.py --task lit-pkl --name MTORC1 --model_path ./models/latest-pkl/MTORC1.pth  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 1 >>log3/MTORC1-.log &
# nohup python aso_abl.py --task lit-pkl --name TP53 --model_path ./models/latest-pkl/TP53.pth  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 1 >>log3/TP53-.log &

# uniform score

# nohup python aso_abl.py --task lit-pkl --name ESR_antago --model_path ./models/latest-pkl-abl/ESR_antago.pth  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 2 >>log3/ESR_antago-.log &
# nohup python aso_abl.py --task lit-pkl --name MAPK1 --model_path ./models/latest-pkl-abl/MAPK1.pth  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 2 >>log3/MAPK1-.log &
# nohup python aso_abl.py --task lit-pkl --name MTORC1 --model_path ./models/latest-pkl-abl/MTORC1.pth  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 2 >>log3/MTORC1-.log &
# nohup python aso_abl.py --task lit-pkl --name TP53 --model_path ./models/latest-pkl-abl/TP53.pth  --lr 0.01 --rounds 10 --bs 20 --strategy ranking --seed 2 >>log3/TP53-.log &

nohup python -u aso_abl.py --task dude-z --name ital --model_path ./models/latest-pkl-abl/z-ital.pth  --lr 0.01 --rounds 10 --bs 5 --strategy ranking --seed 0 >>log5/ital.log &

# wait
