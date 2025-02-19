import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import random
import os
from typing import Literal, Union, Optional, List
from collections import defaultdict
from sklearn.metrics import roc_auc_score, pairwise_distances_argmin_min
from sklearn.cluster import KMeans

def init_seed(seed=1):
    # make the result reproducible
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Modified dataset class for numpy vectors
class Mol_Dataset(Dataset):
    def __init__(self, X, Y):
        # X is the numpy array of feature vectors, Y is the label array
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        feat = self.X[idx]
        y = self.Y[idx]
        return torch.tensor(feat), torch.tensor(y)

def create_dataset(entire_set: np.array = None, selected_indices: Union[list, np.array] = None, score: np.array = None, 
                   batch_size: int = None, shuffle: bool = True):
    selected_set = entire_set[selected_indices]
    selected_score = score[selected_indices]
    dataset = Mol_Dataset(selected_set, selected_score)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# MLP network
class MLP1(nn.Module):
    def __init__(self, input_size: int = None, output_size: int = 1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2*input_size),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(2*input_size, input_size//2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(input_size//2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class MLP(nn.Module):
    def __init__(self, input_size: int = None, hidden_size=500, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLP2(nn.Module):
    def __init__(self, input_size: int = None,hidden_size: int = 500, output_size: int = 2):
        super(MLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
        
class TwoLayerNN(nn.Module):
    def __init__(self, input_size=1024, hidden_size=100, output_size=1):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        diff = torch.abs(inputs - targets)
        # loss = self.alpha * torch.pow((1 - torch.exp(-diff)), self.gamma) * diff
        loss = self.alpha * torch.pow(diff/20, self.gamma) * diff
        return loss.mean()

    
def percentile(score, label):
    # 对score和label按照score从大到小的顺序进行排序
    sorted_indices = np.argsort(score)[::-1]
    sorted_score = score[sorted_indices]
    sorted_label = label[sorted_indices]

    percentile_dict = defaultdict(int)
    # 初始化变量
    count = 0
    current_sum = 0

    # 遍历排序后的score和label
    for i in range(len(sorted_score)):
        current_sum += sorted_label[i]
        count += 1

        # 如果标签累加和达到目标值，则返回当前的个数
        if sorted_label[i]:
            percentile_dict[current_sum] = count

    return percentile_dict

class Strategy:
    
    def Random(self, unlabeled_index: np.array = None, batch_size: int = None, eval_score: np.array = None):
        indexs = np.where(unlabeled_index==1)[0]
        selected_index = np.random.choice(indexs, batch_size, replace=False)
        return selected_index
    

    def Greedy(self, unlabeled_index, batch_size, eval_score):
        eval_score = unlabeled_index * eval_score
        sorted_indexs = np.argsort(-eval_score)
        top_indices = sorted_indexs[:batch_size]     # small libarary
        return top_indices
    
    def Greedy_diverse(self, unlabeled_index: np.array = None, batch_size: int = None, eval_score: np.array = None, feature: np.array = None):
        # cluster the top-k into several clusters and choose their centroid sample to form the batch
        eval_score = unlabeled_index * eval_score
        sorted_indexs = np.argsort(-eval_score)
        top_indices = sorted_indexs[:1000]
        # dataloader = create_dataset(feature, top_indices, eval_score, len(top_indices), shuffle=False)
        # for features,_ in dataloader:
        #     break
        features = feature[top_indices]
        # features = features.numpy()
        kmeans = KMeans(n_clusters=batch_size)
        kmeans.fit(features)
        centroids_index,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
        selected_indexs = top_indices[centroids_index]
        return selected_indexs
    
    def Cluster_and_rank(self, unlabel_index, batch_size, eval_score, clus_result, clus_dict):
        # choose the largest cluster from top-1k molecules, then choose the representative samples for wet lab
        eval_score = unlabel_index * eval_score
        sorted_indexs = np.argsort(-eval_score)
        top_indices = sorted_indexs[:500]
        candidate_cluster = []
        rank_list = []
        for idx in top_indices:
            if clus_result[idx] in candidate_cluster:
                continue
            else:
                candidate_cluster.append(clus_result[idx])
                cur_clusters = clus_dict[clus_result[idx]]
                metric_rank = 0
                for mem in cur_clusters:
                    r = np.where(sorted_indexs==mem)[0][0]
                    metric_rank += 1/np.sqrt(r+10)
                metric_rank *= np.sqrt(len(cur_clusters))
                rank_list.append([idx,metric_rank])

        sorted_rank_list = sorted(rank_list, key=lambda x:x[1], reverse=True)
        top_samples = [item[0] for item in sorted_rank_list[:batch_size]]
        return top_samples