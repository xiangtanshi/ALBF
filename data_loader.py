from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
import pandas as pd
import pickle
from typing import Literal, Union, Optional
import joblib

def vectorize(smile_list: list, radius=2, dim=256, types='bit'):
    N = len(smile_list)
    feature = np.zeros((N,dim),dtype='float32')
    ecfp = [] 
    for idx,smile in enumerate(smile_list):
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprint(mol, radius)
        if types == 'norm':
            # 将smiles串转化为morgan dict，再提取出维度为dim的向量
            fp_dict = fp.GetNonzeroElements()
            feat = np.zeros(dim,dtype=float)
            for key,count in fp_dict.items():
                feat[key%dim] += count
            feat /= np.linalg.norm(feat)
        elif types == 'bit':
            # 直接提取bit向量
            fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, dim)
            feat = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp_vec, feat)
        feature[idx] = feat
        ecfp.append(fp)

    return feature, ecfp

def similarity(fp1, fp2, default=True):
    
    if default:
        # fp1, fp2是morgan fingerprint
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        # fp1 and fp2 are nonzeros elements of the morgan fingerprint, in the form of dict
        interset, union = 0, 0
        for key in fp1.keys():
            if key in fp2.keys():
                interset += min(fp1[key],fp2[key])
                union += max(fp1[key],fp2[key])
            else:
                union += fp1[key]

        for key in fp2.keys():
            if key not in fp1.keys():
                union += fp2[key]

        return interset/union

def sample_selection(label):
    N = len(label)
    selected = set()
    
    label_1_samples = set(np.where(label == 1)[0])
    label_1_selected = set(np.random.choice(list(label_1_samples), len(label_1_samples), replace=False))
    label_0_samples = set(np.where(label == 0)[0])
    label_0_selected = set(np.random.choice(list(label_0_samples), min(len(label_1_selected)*49, len(label_0_samples)), replace=False))
    
    selected.update(label_0_selected)
    selected.update(label_1_selected)
    
    # Step 3: Return all selected samples
    return np.array(list(selected))


def get_dataset(data : Literal['dude-smi','lit-smi','lit-pkl'], types: Literal['bit','norm'], name: str = 'aa2ar', dim: int = 128, clip_f: bool=False):
    '''
    
    '''
    if data == 'dude-smi':
        df = pd.read_csv(f'./vs/raw_data/dud-e_vina_result/{name}.csv')
        smiles_list = df['smiles'].tolist()
        label_list = df['label'].tolist()
        score_list = df['score_1'].tolist()
        label_list = [0 if item == 'decoys' else 1 for item in label_list]
        # add the debias decoys
        with open('./vs/raw_data/debias.txt','r') as f:
            de_smiles_list = [item for item in f.readlines()]
            assert len(de_smiles_list) == 2000
            smiles_list.extend(de_smiles_list)
            score_list.extend([0 for i in range(2000)])
            label_list.extend([0 for i in range(2000)])
        label = np.array(label_list)
        score = -np.array(score_list)
        feature, ecfp = vectorize(smiles_list,dim=dim,types=types)
    elif data == 'dude-z':
        with open(f'./vs/raw_data/dude-z/{name}/ligands.smi','r') as f:
            active_list = [line.split()[0] for line in f.readlines()]
        with open(f'./vs/raw_data/dude-z/{name}/decoys.smi','r') as f:
            decoy_list = [line.split()[0] for line in f.readlines()]
        smiles_list = active_list + decoy_list
        label_list = [1 for i in range(len(active_list))] + [0 for i in range(len(decoy_list))]
        label = np.array(label_list)
        score = np.zeros(len(smiles_list))
        feature, ecfp = vectorize(smiles_list,dim=dim,types=types)      
    elif data == 'lit-smi':
        with open(f'./vs/raw_data/receptor-litpcba/{name}/smiles.txt','r') as file:
            smiles_list = [line.split()[0] for line in file.readlines()]
            score = -np.loadtxt(f'./vs/raw_data/receptor-litpcba/{name}/{name}_score.txt')
            label = np.loadtxt(f'./vs/raw_data/receptor-litpcba/{name}/{name}_label.txt').astype(np.int32)
            if name in ['VDR', 'PKM2']:
                # choose a dense subset
                subset_indexs = sample_selection(label)
                label = label[subset_indexs]
                score = score[subset_indexs]
                smiles_list = [smiles_list[i] for i in subset_indexs]
            feature, ecfp = vectorize(smiles_list,dim=dim,types=types)
    elif data == 'lit-pkl':
        with open(f'./DrugCLIP/result/{name}.pkl','rb') as f:
            data = pickle.load(f)

        smiles_list = [line.split()[0] for line in data['mol_names']]
        label = data['labels']
        score = data['res_single']
        if name in ['VDR', 'PKM2', 'MAPK1', 'MTORC1', 'ESR_antago', 'TP53']:
                # choose a dense subset
                subset_indexs = sample_selection(label)
                label = label[subset_indexs]
                score = score[subset_indexs]
                smiles_list = [smiles_list[i] for i in subset_indexs]
        feature, ecfp = vectorize(smiles_list,dim=dim,types=types)
        if clip_f:
            feature = joblib.load(f'./DrugCLIP/features/{name}.joblib').astype('float32') # dim=128
            feature = feature[subset_indexs]
    return feature, label, score, ecfp, smiles_list

