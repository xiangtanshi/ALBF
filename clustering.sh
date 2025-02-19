#!/bin/bash
# akt1, aofb, cp3a4, cxcr4, gria2, hivpr, hivrt, kif11
# nohup python  clustering.py --task dude-smi --name aofb --dim 128 --types bit --neighbor 50 >> log/aofb.log 2>&1 &
# nohup python  clustering.py --task dude-smi --name akt1 --dim 128 --types bit --neighbor 50 >> log/akt1.log 2>&1 &
# nohup python  clustering.py --task dude-smi --name cp3a4 --dim 128 --types bit --neighbor 50 >> log/cp3a4.log 2>&1 &
# nohup python  clustering.py --task dude-smi --name cxcr4 --dim 128 --types bit --neighbor 50 >> log/cxcr4.log 2>&1 &
# nohup python  clustering.py --task dude-smi --name hivpr --dim 128 --types bit --neighbor 50 >> log/hivpr.log 2>&1 &
# nohup python  clustering.py --task dude-smi --name gria2 --dim 128 --types bit --neighbor 50 >> log/gria2.log 2>&1 &
# nohup python  clustering.py --task dude-smi --name hivrt --dim 128 --types bit --neighbor 50 >> log/hivrt.log 2>&1 &
# nohup python  clustering.py --task dude-smi --name kif11 --dim 128 --types bit --neighbor 50 >> log/kif11.log 2>&1 &

python clustering.py --task dude-z --name aa2ar --dim 2048 --types bit --neighbor 50 --method bitbirch
# exit

# nohup python  clustering.py --task lit-smi --name ALDH1 --dim 128 --types bit --neighbor 50 >> log/ALDH1.log 2>&1 &
# nohup python  clustering.py --task lit-smi --name ALDH1 --dim 128 --types bit --neighbor 50 --scale 0.9 >> log/ALDH1.log 2>&1 &
# nohup python  clustering.py --task lit-smi --name ALDH1 --dim 128 --types bit --neighbor 50 --scale 0.85 >> log/ALDH1.log 2>&1 &


# nohup python  clustering.py --task lit-smi --name PKM2 --dim 128 --types bit --neighbor 50 >> log/PKM2.log 2>&1 &
# nohup python  clustering.py --task lit-smi --name PKM2 --dim 128 --types bit --neighbor 50 --scale 0.9 >> log/PKM2.log 2>&1 &
# nohup python  clustering.py --task lit-smi --name PKM2 --dim 128 --types bit --neighbor 50 --scale 0.85 >> log/PKM2.log 2>&1 &


# nohup python  clustering.py --task lit-smi --name VDR --dim 128 --types bit --neighbor 50 >> log/VDR.log 2>&1 &
# nohup python  clustering.py --task lit-smi --name VDR --dim 128 --types bit --neighbor 50 --scale 0.9 >> log/VDR.log 2>&1 &
# nohup python  clustering.py --task lit-smi --name VDR --dim 128 --types bit --neighbor 50 --scale 0.85 >> log/VDR.log 2>&1 &

# exit

# nohup python  clustering.py --task lit-pkl --name ALDH1 --dim 128 --types bit --neighbor 50 >> log/ALDH1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name ALDH1 --dim 128 --types bit --neighbor 50 --clip >> log/ALDH1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name PKM2 --dim 128 --types bit --neighbor 50 >> log/PKM2.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name PKM2 --dim 128 --types bit --neighbor 50 --clip >> log/PKM2.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name VDR --dim 128 --types bit --neighbor 50 >> log/VDR.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name VDR --dim 128 --types bit --neighbor 50 --clip >> log/VDR.log 2>&1 &


# nohup python  clustering.py --task lit-pkl --name ESR_ago --dim 128 --types bit --neighbor 50 >> log/ESR_ago.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name ESR_antago --dim 128 --types bit --neighbor 50 >> log/ESR_antago.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name MAPK1 --dim 128 --types bit --neighbor 50 >> log/MAPK1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name MTORC1 --dim 128 --types bit --neighbor 50 >> log/MTORC1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name PPARG --dim 128 --types bit --neighbor 50 >> log/PPARG.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name TP53 --dim 128 --types bit --neighbor 50 >> log/TP53.log 2>&1 &


# nohup python  clustering.py --task lit-pkl --name ESR_ago --dim 128 --types bit --neighbor 50 --clip >> log/ESR_ago.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name ESR_antago --dim 128 --types bit --neighbor 50 --clip >> log/ESR_antago.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name MAPK1 --dim 128 --types bit --neighbor 50 --clip >> log/MAPK1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name MTORC1 --dim 128 --types bit --neighbor 50 --clip >> log/MTORC1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name PPARG --dim 128 --types bit --neighbor 50 --clip >> log/PPARG.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name TP53 --dim 128 --types bit --neighbor 50 --clip >> log/TP53.log 2>&1 &
# exit

# nohup python  clustering.py --task lit-pkl --name ALDH1 --dim 128 --types bit --neighbor 50 >> log/ALDH1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name ALDH1 --dim 128 --types bit --neighbor 50 --clip >> log/ALDH1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name ALDH1 --dim 128 --types bit --neighbor 50 --scale 0.9 >> log/ALDH1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name ALDH1 --dim 128 --types bit --neighbor 50 --clip --scale 0.9 >> log/ALDH1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name ALDH1 --dim 128 --types bit --neighbor 50 --scale 0.85 >> log/ALDH1.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name ALDH1 --dim 128 --types bit --neighbor 50 --clip --scale 0.85 >> log/ALDH1.log 2>&1 &

# nohup python  clustering.py --task lit-pkl --name PKM2 --dim 128 --types bit --neighbor 50 >> log/PKM2.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name PKM2 --dim 128 --types bit --neighbor 50 --clip >> log/PKM2.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name PKM2 --dim 128 --types bit --neighbor 50 --scale 0.9 >> log/PKM2.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name PKM2 --dim 128 --types bit --neighbor 50 --clip --scale 0.9 >> log/PKM2.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name PKM2 --dim 128 --types bit --neighbor 50 --scale 0.85 >> log/PKM2.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name PKM2 --dim 128 --types bit --neighbor 50 --clip --scale 0.85 >> log/PKM2.log 2>&1 &

# nohup python  clustering.py --task lit-pkl --name VDR --dim 128 --types bit --neighbor 50 >> log/VDR.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name VDR --dim 128 --types bit --neighbor 50 --clip >> log/VDR.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name VDR --dim 128 --types bit --neighbor 50 --scale 0.9 >> log/VDR.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name VDR --dim 128 --types bit --neighbor 50 --clip --scale 0.9 >> log/VDR.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name VDR --dim 128 --types bit --neighbor 50 --scale 0.85 >> log/VDR.log 2>&1 &
# nohup python  clustering.py --task lit-pkl --name VDR --dim 128 --types bit --neighbor 50 --clip --scale 0.85 >> log/VDR.log 2>&1 &

