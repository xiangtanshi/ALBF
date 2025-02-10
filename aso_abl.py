'''
use active learning to simulate the disbutrition of docking score for a target protein and a molecule library
the feedback information for active learning is the real docking score for a selected molecule.
'''
from utils import *
from clustering import *
import torch.optim as optim
from data_loader import *
import argparse
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

def ft(model, dataloader, args, device):
    
    criterion = nn.MSELoss()
    # criterion = FocalLoss(alpha=1, gamma=3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.2)
    model.train()
    for epoch in range(args.epoch):
        losses = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device).float())
            loss = criterion(outputs, labels.to(device).float().unsqueeze(1))
            # loss = criterion(outputs, labels.to(device).float())   # wrong for mseloss, will lead to broadcasting and loss diverge
            loss.backward()
            optimizer.step()
            losses += loss.cpu().item() * inputs.size(0)
        avg_loss = losses / len(dataloader.dataset)
        scheduler.step()
        # if epoch%5 == 1:
        #     print('Epoch:{},average loss:{}.'.format(epoch,losses))

    return model

def evaluate(dataloader, model, label, device):
    
    eval_score = np.zeros(len(dataloader.dataset))
    model.eval()

    all_outputs = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs.to(device))
            all_outputs.extend(outputs)  # Convert to numpy array and accumulate
    gathered_outputs = torch.cat(all_outputs)
    eval_score = gathered_outputs.cpu().numpy()

    auc = roc_auc_score(label,eval_score)

    sorted_indices_pred = np.argsort(-eval_score)
    top_100_indices_pred = sorted_indices_pred[:100]
    top_05_indices_pred = sorted_indices_pred[:len(label)//200]
    recall_num_1 = np.sum(label[top_100_indices_pred])  # active molecules in the top-100 pred molecules
    recall_num_2 = np.sum(label[top_05_indices_pred])
    return eval_score, auc, recall_num_1, recall_num_2

def evaluate_direct(eval_score, label):
    auc = roc_auc_score(label,eval_score)
    sorted_indices_pred = np.argsort(-eval_score)
    top_100_indices_pred = sorted_indices_pred[:100]
    top_05_indices_pred = sorted_indices_pred[:len(label)//200]
    recall_num_1 = np.sum(label[top_100_indices_pred])  # active molecules in the top-1k pred molecules
    recall_num_2 = np.sum(label[top_05_indices_pred])
    return eval_score, auc, recall_num_1, recall_num_2


def active_regression(args, feature, label, ecfp, par, device):
    # bio-activity as the feedback for finetuning the score function net

    test_dataloader = create_dataset(feature, np.arange(len(label)), label, 4096, False)
    model = MLP(input_size=args.dim, output_size=args.out_dim)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    # evaluate the model
    eval_score, auc, re_n1, re_n2 = evaluate(test_dataloader, model, label, device)
    print(f"Molecule numbers:{feature.shape[0]}, active number:{np.sum(label)}, active query batch size:{args.bs}, initial model, AUC: {auc}, active numbers top-100:{re_n1}, top-0.5%:{re_n2}")

    N = feature.shape[0]
    if par is not None:
        classes = len(np.unique(par))
        cluster_dict = dict()
        for i in range(classes):
            cluster_dict[i] = set()
        for i in range(N):
            cluster_dict[par[i]].add(i)
    else:
        cluster_dict = None

    unlabel_index = np.ones(N)

    l1,l2,l3= [re_n1],[re_n2],[0]
    for i in range(args.rounds):
        # print("Round:", i)
        # acquire new batch of molecules
        if args.strategy == 'random':
            selected_indices = Strategy().Random(unlabel_index, args.bs, eval_score)
        elif args.strategy == 'greedy':
            selected_indices = Strategy().Greedy(unlabel_index, args.bs, eval_score)
        elif args.strategy == 'kmeans':
            selected_indices = Strategy().Greedy_diverse(unlabel_index, args.bs, eval_score, feature)
        elif args.strategy == 'ranking':
            selected_indices = Strategy().Cluster_and_rank(unlabel_index, args.bs, eval_score, par, cluster_dict)
        unlabel_index[selected_indices] = 0
        label_index = np.where(unlabel_index==0)[0]

        cur_dataloader = create_dataset(feature, label_index, label * 20 , args.bs, shuffle=True)
        # cur_dataloader = create_dataset(feature, selected_indices, label * 15, len(selected_indices), shuffle=True)
        model = ft(model, cur_dataloader, args, device)

        # evaluate the model
        eval_score, auc, re_n1, re_n2 = evaluate(test_dataloader, model, label, device)
        selected_score = eval_score[label_index]
        real_label = label[label_index]
        test_active = np.sum(label[label_index])
        # print(f"epoch: {i}, selected active by the strategy: {test_active}; after training, the results are AUC: {auc}, active molecules top-100:{re_n1}, top-0.5%:{re_n2}")
        l1.append(re_n1); l2.append(re_n2); l3.append(test_active)

    # save the final model
    model.cpu()
    # torch.save(model, './datas/model/wet_{}.pt'.format(name))
    print(f'seed-task-name-strategy:{args.seed}-{args.task}-{args.name}-{args.strategy},{l1},{l2},{l3}')

    return 1

def active_probabilistic_regression(args, feature, label, ecfp, clus_result, knn, device):
    # small library
    # feedback is activity
    N = feature.shape[0]
    unlabel_index = np.ones(N)
    print('Active query based on clusters and expanded neighoring area')

    test_dataloader = create_dataset(feature, np.arange(len(label)), label, 4096, False)
    model = MLP(input_size=args.dim, output_size=args.out_dim)
    model.load_state_dict(torch.load(args.model_path))
    # model = torch.load('./s-model-1024/lit-smi-{}.pt'.format(args.name))
    model = model.to(device)

    # print the clustering purity
    clus_dict = dict()
    for idx, cls_id in enumerate(clus_result):
        if cls_id not in clus_dict:
            clus_dict[cls_id] = [idx]
        else:
            clus_dict[cls_id].append(idx)
    purity = recall_resource_ratio(label, clus_result, clus_dict)
    print(f'Clustering purity:{purity}')

    # evaluate the model
    eval_score, auc, re_n1, re_n2 = evaluate(test_dataloader, model, label, device)
    print(f"initial model, AUC: {auc}, active hit rate among top-100:{re_n1}, top-0.5%:{re_n2}")
    # transform the score to docking probability
    eval_prob = eval_score / max(eval_score) * 0.3
    # eval_prob = np.clip(eval_prob, 0, None)

    pseudo_score = np.zeros(N)
    l1,l2,l3 = [re_n1],[re_n2],[0]
    test_active = 0


    alpha, beta = 1, 0.4

    for i in range(args.rounds):
        print("Round:", i)
        # acquire new batch of molecules
        if args.strategy == 'random':
            selected_indices = Strategy().Random(unlabel_index, args.bs, eval_prob)
        elif args.strategy == 'greedy':
            selected_indices = Strategy().Greedy(unlabel_index, args.bs, eval_prob)
        elif args.strategy == 'greedy-diverse':
            selected_indices = Strategy().Greedy_diverse(unlabel_index, args.bs, eval_prob, feature)
        elif args.strategy == 'ranking':
            selected_indices = Strategy().Cluster_and_rank(unlabel_index, args.bs, eval_prob, par, clus_dict)
        unlabel_index[selected_indices] = 0

        # adjust the score based on the web lab 
        for index in selected_indices:
            cur_cluster = clus_dict[clus_result[index]]
            if label[index]:
                # positive feedback for the cluster members
                for idx in cur_cluster:
                    if idx != index:
                        eval_prob[idx] += alpha * (1 - eval_prob[index]) * similarity(ecfp[index],ecfp[idx])
                eval_prob[index] = 1.0
                unlabel_index[cur_cluster] = 0
            else:
                # find the neighbor area
                neighbors = []
                for idx in cur_cluster:
                    for nei in knn[idx][1:]:
                        if nei not in neighbors and unlabel_index[nei] and similarity(ecfp[idx],ecfp[nei]) > beta:
                            neighbors.append(nei)
                    if idx not in neighbors:
                        neighbors.append(idx)
                # negative feedback for the neighbor members
                if index in neighbors:
                    neighbors.remove(index)
                for idx in neighbors:
                    eval_prob[idx] -= alpha * eval_prob[index] * similarity(ecfp[index],ecfp[idx])
                eval_prob[index] = 0
            eval_prob = np.clip(eval_prob,0,1)

        # evaluate the model
        eval_score, auc, re_n1, re_n2 = evaluate_direct(eval_prob, label)
        test_active += np.sum(label[selected_indices])
        print(f"initial model, AUC: {auc}, active hit rate among top-100:{re_n1}, top-0.5%:{re_n2}, found active: {test_active}")
        l1.append(re_n1); l2.append(re_n2);l3.append(test_active)

    # save the final model
    model.to(torch.device('cpu'))
    # torch.save(model, './datas/model/cls_wet_{}.pt'.format(name))
    print(f'seed-task-name-strategy:{args.seed}-{args.task}-{args.name}-{args.strategy},{l1}-{l2}-{l3}')

    return 1



# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='set the file for processing.')
    parser.add_argument('--rounds', default=10, type=int, help='number of active learning cycles')
    parser.add_argument('--epoch', default=30, type=int, help='number of training epochs')
    parser.add_argument('--bs', default=5, type=int, help='the number of molecules selected each round')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', default=2, type=int, help='the GPU index')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--strategy', default='greedy', type=str)

    parser.add_argument('--task', type=str, default='dude-smi',choices=['dude-smi','lit-smi','lit-pkl'])
    parser.add_argument('--name', type=str, default='aa2ar')
    parser.add_argument('--dim', type=int, default=2048)
    parser.add_argument('--types', type=str, default='bit',choices=['norm','bit'])
    parser.add_argument('--clip', action='store_true', default=False)
    parser.add_argument('--neighbor', type=int, default=50)
    parser.add_argument('--scale', type=float,default=1.0)
    parser.add_argument('--out_dim', default=1, type=int)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    init_seed(args.seed)

    device = torch.device('cuda:{}'.format(args.device))
    feature, label, score, ecfp, smiles_list = get_dataset(args.task,args.types,args.name,args.dim,args.clip)
    if args.name in ['ALDH1']:
        par = np.loadtxt(f'./vs/raw_data/cluster/{args.task}-{args.name}-{128}-{args.types}-{args.scale}-{args.clip}.txt')
    elif args.name in ['VDR', 'PKM2', 'MAPK1', 'MTORC1','ESR_antago','TP53'] and args.strategy == 'ranking':
        # knn = faiss_search_approx_knn(feature, feature, args.neighbor, num_gpu=2, norm=(args.types=='norm' or args.clip))
        neigh = NearestNeighbors(n_neighbors=args.neighbor).fit(feature)
        _, knn = neigh.kneighbors(feature)
        # par, cluster_dict = fast_prob_clustering(knn, ecfp, scale=args.scale)
        par, cluster_dict = classic_clustering(method='minik', feature=feature)
        rate = recall_resource_ratio(label, par, cluster_dict)
        print(f'active molecules purity: {rate}.')
    

    # wet lab activity as feedback

    # baseline: iterative choose top score mol for wet lab
    if args.strategy == 'ranking':
        active_regression(args, feature, label, ecfp, par, device)
    else:
        active_regression(args, feature, label, ecfp, None, device)

