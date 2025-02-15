#!/bin/bash

# nohup python  clustering.py --task lit-smi --name ALDH1 --dim 128 --types bit --method birch >> log/ALDH1_.log 2>&1 &
nohup python  clustering.py --task lit-smi --name ALDH1 --dim 128 --types bit --method minik >> log/ALDH1_.log 2>&1 &


# nohup python  clustering.py --task lit-smi --name PKM2 --dim 128 --types bit --method birch >> log/PKM2_.log 2>&1 &
nohup python  clustering.py --task lit-smi --name PKM2 --dim 128 --types bit --method minik >> log/PKM2_.log 2>&1 &


# nohup python  clustering.py --task lit-smi --name VDR --dim 128 --types bit --method birch >> log/VDR_.log 2>&1 &
nohup python  clustering.py --task lit-smi --name VDR --dim 128 --types bit --method minik >> log/VDR_.log 2>&1 &