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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# import torch.nn as nn
# import torch.optim as optim

def ft_classfication(model, dataloader, args, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epoch):
        losses = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            losses += loss.cpu().item()
        # if epoch % 5 == 1:
        #     print('Epoch:{},average loss:{}.'.format(epoch,losses))

    return model


def evaluate_dropout(dataloader, model, label, device):
    eval_score = np.zeros(len(dataloader.dataset))

    # Simulate ten different model predictions without using model.eval()
    for i in range(5):
        iteration_outputs = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                outputs = model(inputs.to(device))
                outputs = nn.functional.softmax(outputs, dim=1)
                iteration_outputs.extend(outputs)
        iteration_outputs = [ts[1].item() for ts in iteration_outputs]
        iteration_outputs = np.array(iteration_outputs)
        if i == 0:
            total_prob = iteration_outputs / 5
        else:
            total_prob += iteration_outputs / 5
    # select the active probability as the evaluate score
    eval_score = total_prob

    auc = roc_auc_score(label, eval_score)

    sorted_indices_pred = np.argsort(-eval_score)
    top_100_indices_pred = sorted_indices_pred[:100]
    top_05_indices_pred = sorted_indices_pred[:len(label)//200]
    recall_num_1 = np.sum(label[top_100_indices_pred])  
    recall_num_2 = np.sum(label[top_05_indices_pred])
    return eval_score, auc, recall_num_1, recall_num_2


def active_classification(args, feature, label, ecfp, par, device):
    # small library
    # traditional active classification

    test_dataloader = create_dataset(feature, np.arange(len(label)), label, 4096, False)
    model = MLP2(input_size=args.dim, output_size=args.out_dim)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # evaluate the model
    eval_score, auc, re_n1, re_n2 = evaluate_dropout(test_dataloader, model, label, device)
    print(f"Curent model {args.task}-{args.name}, Active query batch size:{args.bs}, initial model, AUC: {auc}, active hit rate among top-100:{re_n1}, top-0.5%:{re_n2}")

    # create a dict of cluster
    N = feature.shape[0]
    classes = len(np.unique(par))
    cluster_dict = dict()
    for i in range(classes):
        cluster_dict[i] = set()
    for i in range(N):
        cluster_dict[par[i]].add(i)

    unlabel_index = np.ones(N)
    l1,l2,l3= [re_n1],[re_n2],[0]

    for i in range(args.rounds):
        # print("Round:", i)
        if i == 0:
            selected_indices = Strategy().Random(unlabel_index, args.bs, eval_score)
        else:
            # acquire new batch of molecules
            if args.strategy == 'random':
                selected_indices = Strategy().Random(unlabel_index, args.bs, eval_score)
            elif args.strategy == 'greedy':
                selected_indices = Strategy().Greedy(unlabel_index, args.bs, eval_score)
            elif args.strategy == 'greedy-diverse':
                selected_indices = Strategy().Greedy_diverse(unlabel_index, args.bs, eval_score, feature)
            elif args.strategy == 'ranking':
                selected_indices = Strategy().Cluster_and_rank(unlabel_index, args.bs, eval_score, par, cluster_dict)
        unlabel_index[selected_indices] = 0
        label_index = np.where(unlabel_index==0)[0]

        cur_dataloader = create_dataset(feature, label_index, label, 50, shuffle=True)
        model = ft_classfication(model, cur_dataloader, args, device)

        #evaluate the model
        eval_score, auc, re_n1, re_n2 = evaluate_dropout(test_dataloader, model, label, device)
        test_active = np.sum(label[label_index])
        # print(f"epoch: {i}, AUC: {auc}, active hit rate among top-100:{re_n1}, top-0.5%:{re_n2}, found active: {test_active}")
        l1.append(re_n1); l2.append(re_n2); l3.append(test_active)

    model.cpu()
    print(f'task-name-strategy:{args.task}-{args.name}-{args.strategy},{l1},{l2},{l3}')
        
def train_active_classifier(args, feature, label, par, knn, smiles, device):
    # bio-activity as the feedback for finetuning the score function net

    
    def select_sample(par, label):
        # Create a dictionary to store samples for each cluster
        clusters = defaultdict(list)
        
        # Group samples by their cluster
        for i, cluster_id in enumerate(par):
            clusters[cluster_id].append(i)
        
        # select active molecules for AC initialization: make sure that AC cannot directly assign high probability to other unseen active molecules
        valid_clusters = []
        for cluster_id, samples in clusters.items():
            if len(samples) <= 2 and all(label[i] == 1 for i in samples):
                valid_clusters.extend(samples)
        
        # If no valid clusters found, return None
        if not valid_clusters:
            raise ValueError("No valid clusters with only 1 or 2 active molecules found")
        
        # Randomly select a sample from valid clusters
        return np.random.choice(valid_clusters)
    
    def get_real(path='./vs/raw_data/real.txt'):
        # randomly sample a few molecules from REAL library
        # we do not use decoys in the original dataset as they already leak some information about bioactivity related to active molecules
        with open(path, 'r') as file:
            smiles_list = [line.split()[0] for line in file.readlines()]
        return smiles_list
        
    decoy_smile = get_real()

    for i in range(1):
        active_indices = select_sample(par, label)
        active_smile = smiles[int(active_indices)]
        decoy_smile.append(active_smile)


    train_feature, _ = vectorize(decoy_smile,dim=args.dim,types=args.types)
    train_label = np.zeros(100)
    train_label[-1] = 1
    # train_label[-2] = 1
    # train_label[-3] = 1
    dataloader = create_dataset(train_feature, np.arange(100), train_label, 100, False)
    
    
    
    model = MLP2(input_size=args.dim, output_size=args.out_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        losses = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            losses += loss.cpu().item()
        # if epoch % 5 == 4:
        #     print('Epoch:{},average loss:{}.'.format(epoch,losses))
    label[active_indices] = 0
    torch.save(model.state_dict(), args.model_path)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='set the file for processing.')
    parser.add_argument('--rounds', default=10, type=int, help='number of active learning cycles')
    parser.add_argument('--epoch', default=30, type=int, help='number of training epochs')                  # default=30
    parser.add_argument('--bs', default=5, type=int, help='the number of molecules selected each round')
    parser.add_argument('--lr', type=float, default=0.01)                                                 # default=0.01
    parser.add_argument('--device', default=0, type=int, help='the GPU index')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--strategy', default='ranking', type=str)

    parser.add_argument('--task', type=str, default='dude-smi',choices=['dude-smi','lit-smi','lit-pkl'])
    parser.add_argument('--name', type=str, default='aa2ar')
    parser.add_argument('--dim', type=int, default=2048)
    parser.add_argument('--types', type=str, default='bit',choices=['norm','bit'])
    parser.add_argument('--clip', action='store_true', default=False)
    parser.add_argument('--neighbor', type=int, default=50)
    parser.add_argument('--scale', type=float,default=1.0)
    parser.add_argument('--out_dim', default=2, type=int)                                                    
    parser.add_argument('--model_path', type=str, default='./model/aa2ar.pth')                       
    args = parser.parse_args()
    init_seed(args.seed)

    device = torch.device('cuda:{}'.format(args.device))
    feature, label, score, ecfp, smiles = get_dataset(args.task,args.types,args.name,args.dim,args.clip)


    par = np.loadtxt(f'./vs/raw_data/cluster/{args.task}-{args.name}-{128}-{args.types}-{args.scale}-{args.clip}.txt')
    knn = np.loadtxt(f'./vs/raw_data/knn/{args.task}-{args.name}-{args.clip}.txt')
    knn = knn.astype(int)
    

    # baseline: active classification
    train_active_classifier(args, feature, label, par, knn, smiles, device)
    active_classification(args, feature, label, ecfp, par, device)
