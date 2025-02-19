import os
import pandas as pd
import random

# 设置路径
path = "./vs/raw_data/dud-e_vina_result/"

# 要排除的文件列表, akt1, aofb, cp3a4, cxcr4, gcr, hivpr, hivrt, kif11
exclude_files = ['akt1.csv', 'aofb.csv', 'cp3a4.csv', 'cxcr4.csv', 'gria2.csv', 'hivpr.csv', 'hivrt.csv', 'kif11.csv']

# 存储所有符合条件的SMILES
all_active_smiles = []

# 遍历目录中的所有CSV文件
for filename in os.listdir(path):
    if filename.endswith('.csv') and filename not in exclude_files:
        file_path = os.path.join(path, filename)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 提取label为'actives'的行对应的SMILES
        active_smiles = df[df['label'] == 'actives']['smiles'].tolist()
        
        # 将提取的SMILES添加到总列表中
        all_active_smiles.extend(active_smiles)
    else:
        print(filename)

# 如果提取的SMILES总数超过2000，随机选择2000个
if len(all_active_smiles) > 2000:
    selected_smiles = random.sample(all_active_smiles, 2000)
else:
    selected_smiles = all_active_smiles

# 将选中的SMILES写入debias.txt文件
with open('./vs/raw_data/debias.txt', 'w') as f:
    for smiles in selected_smiles:
        f.write(f"{smiles}\n")

print(f"已成功提取并保存 {len(selected_smiles)} 个SMILES到debias.txt文件中。")