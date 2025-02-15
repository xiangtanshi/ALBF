import os
import numpy as np
import argparse
import faiss
from data_loader import *
import time
from sklearn.cluster import Birch, MiniBatchKMeans
from bitbirch import *
from sklearn.neighbors import NearestNeighbors

def faiss_search_approx_knn(query, target, k, num_gpu, norm=True):
    if norm:
        # for datasets with normalized features in a sphere space
        cpu_index = faiss.IndexFlatIP(target.shape[1])
    else:
        # for datasets with bit feature that are not normalized
        cpu_index = faiss.IndexFlatL2(target.shape[1])

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    co.usePrecomputed = False
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=num_gpu)
    try:
        gpu_index.add(target)
    except:
        raise ValueError('cannot load feature to GPU')
    _, knn = gpu_index.search(query, k=k)
    del gpu_index

    return knn

def fast_prob_clustering(knn, fp, epsilon=0.0001, scale=1.0):

    # get the log_prob_dict
    N = knn.shape[0]
    row_col, prob, contained_row_col_set = [], [], set()

    for i in range(N):
        for index, j in enumerate(knn[i]):
            key = (min(i,j),max(i,j))
            if key in contained_row_col_set:
                continue
            row_col.append(key)
            contained_row_col_set.add(key)
            prob.append(scale * similarity(fp[i],fp[j]))
    prob = np.clip(np.asarray(prob,dtype=float),epsilon,1-epsilon)
    log_prob = np.log(prob/(1-prob))

    log_prob_dict = dict(zip(row_col, log_prob))
    partition, cluster_dict = np.arange(N), dict([(i,{i}) for i in range(N)])

    def get_point_set_likelihood(i, cluster_id):
        cluster = cluster_dict[cluster_id]
        likelihood = 0.0
        for j in cluster:
            if i == j:
                continue
            key = (min(i,j),max(i,j))
            if key not in log_prob_dict:
                p_ij = scale * similarity(fp[i],fp[j])
                p_ij = min(1-epsilon, max(epsilon, p_ij))
                log_prob_dict[key] = np.log(p_ij/(1-p_ij))
            likelihood += log_prob_dict[key]
        return likelihood

    # perform the greedy cluster adjustment
    update = True
    while update:
        update = False
        for i in range(N):
            candidate_clusters = set(partition[knn[i]])
            cur_cluster_likelihood = get_point_set_likelihood(i, partition[i])
            max_cluster_id, max_cluster_likelihood = partition[i], 0.0
            for cluster_id in candidate_clusters:
                if cluster_id == partition[i]:
                    continue
                likelihood = get_point_set_likelihood(i, cluster_id) - cur_cluster_likelihood
                if likelihood > max_cluster_likelihood:
                    max_cluster_id = cluster_id
                    max_cluster_likelihood = likelihood
            if partition[i] != max_cluster_id:
                update = True
                cluster_dict[max_cluster_id].add(i)
                cluster_dict[partition[i]].remove(i)
                partition[i] = max_cluster_id

    # reorganize the labels and cluster_dict
    _, par = np.unique(partition, False, True)
    classes = len(np.unique(par))
    cluster_dict = dict()
    for i in range(classes):
        cluster_dict[i] = set()
    for i in range(N):
        cluster_dict[par[i]].add(i)
    return par, cluster_dict

def dict_info(cluster_dict):
    record = dict()
    for key in cluster_dict.keys():
        length = len(cluster_dict[key])
        if length not in record:
            record[length] = 1
        else:
            record[length] += 1
    print('Cluster size distribution:',end=',')
    record_list = [[item, record[item]] for item in record if item !=0]
    record_list.sort(key=lambda x: x[0])
    print(record_list)
    return 1


def recall_resource_ratio(label, par, cluster_dict):
    #  召回率=\sum{活性分子所在类的纯度 * 类内活性分子数 } / 活性分子总数

    num_cm = 0
    num_total = np.sum(label)
    active_cluster_list = []
    for i in range(label.shape[0]):
        if label[i] and par[i] not in active_cluster_list:
            active_cluster_list.append(par[i])
    for key in active_cluster_list:
        active_num = 0
        for idx in cluster_dict[key]:
            active_num += label[idx]
        mol_num = len(cluster_dict[key])
        num_cm += active_num / mol_num * active_num

    recall_rate = num_cm / num_total
    # 单位计算成本的活性分子召回率
    return recall_rate
    
def classic_clustering(method: str = 'birch', feature: np.array = None):
    N = feature.shape[0]
    n = N//5
    # n = (n//256)*256
    if method == "birch":
        clusterer = Birch(n_clusters=n, branching_factor=100, threshold=0.35)
    elif method == "minik":
        clusterer = MiniBatchKMeans(n_clusters=n, init="k-means++", batch_size=n)
    clusterer.fit(feature)
    par = clusterer.labels_
    x, par = np.unique(par, False, True)
    cluster_dict = dict()
    cls_num = len(x)
    for i in range(cls_num):
        cluster_dict[i] = set()
    for i in range(N):
        cluster_dict[par[i]].add(i)
    return par, cluster_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='set the file for processing.')
    parser.add_argument('--task', type=str, default='dude-z',choices=['dude-smi','lit-smi','lit-pkl','dude-z'])
    parser.add_argument('--name', type=str, default='aa2ar')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--types', type=str, default='bit',choices=['norm','bit'])
    parser.add_argument('--clip', action='store_true', default=False)
    parser.add_argument('--neighbor', type=int, default=50)
    parser.add_argument('--scale', type=float,default=1.0)
    parser.add_argument('--method', default='pc', choices=['pc','birch','minik','bitbirch'])
    args = parser.parse_args()
    start_t = time.time()

    feature, label, score, ecfp, smiles_list = get_dataset(args.task,args.types,args.name,args.dim,args.clip)
    
    if args.method == 'pc':
        # knn = faiss_search_approx_knn(feature, feature, args.neighbor, num_gpu=2, norm=(args.types=='norm' or args.clip))
        neigh = NearestNeighbors(n_neighbors=args.neighbor).fit(feature)
        _, knn = neigh.kneighbors(feature)
        # np.savetxt(f'./vs/raw_data/knn/{args.task}-{args.name}-{args.clip}.txt', knn, fmt='%d')
        # import sys
        # sys.exit()
        par, cluster_dict = fast_prob_clustering(knn, ecfp, scale=args.scale)
        end_t = time.time()
        duration = end_t - start_t
        dict_info(cluster_dict)
        rate = recall_resource_ratio(label, par, cluster_dict)
        print(f'setting: task-name-dim-types-clip-scale-time: {args.task}-{args.name}-{args.dim}-{args.types}-{args.clip}-{args.scale}-{duration:.2f}\tactive molecules purity:{rate}')
        # np.savetxt(f'./vs/raw_data/cluster/{args.task}-{args.name}-{args.dim}-{args.types}-{args.scale}-{args.clip}.txt', par, fmt='%d')
    elif args.method == 'bitbirch':
        clusterer = BitBirch(threshold=0.6, branching_factor=100)
        clusterer.fit(feature)
        cluster_list = clusterer.get_cluster_mol_ids()
        n_molecules = len(feature)
        cluster_labels = [0] * n_molecules
        cluster_dict = dict()
        for cluster_id, indices in enumerate(cluster_list):
            for idx in indices:
                cluster_labels[idx] = cluster_id
            cluster_dict[cluster_id] = set(indices)
        par = np.array(cluster_labels)
        end_t = time.time()
        duration = end_t - start_t
        dict_info(cluster_dict)
        rate = recall_resource_ratio(label, par, cluster_dict)
        print(f'setting: task-name-dim-types-method-time: {args.task}-{args.name}-{args.dim}-{args.types}-{args.method}-{duration:.2f}\tactive molecules purity:{rate}')
        # np.savetxt(f'./vs/raw_data/bitbirch_cluster/{args.task}-{args.name}-{args.dim}-{args.types}-{args.method}-5.txt', par, fmt='%d')
    else:
        par, cluster_dict = classic_clustering(method=args.method, feature=feature)
        end_t = time.time()
        duration = end_t - start_t
        dict_info(cluster_dict)
        rate = recall_resource_ratio(label, par, cluster_dict)
        print(f'setting: task-name-dim-types-method-time: {args.task}-{args.name}-{args.dim}-{args.types}-{args.method}-{duration:.2f}\tactive molecules purity:{rate}')
        np.savetxt(f'./vs/raw_data/classic_cluster/{args.task}-{args.name}-{args.dim}-{args.types}-{args.method}-5.txt', par, fmt='%d')