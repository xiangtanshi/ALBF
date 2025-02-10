from utils import *
import torch.optim as optim
from data_loader import *
import argparse
from tqdm import tqdm
from accelerate import Accelerator
import time
import warnings
warnings.filterwarnings('ignore')

def remove_nan_entries(feature, label, score):
    # 检查label中的NaN项
    keep_indices = ~np.isnan(score)
    remove_indices = ~keep_indices
    keep_indices = np.where(keep_indices)[0]
    remove_indices = np.where(remove_indices)[0]
    # print(remove_indices)
    # 根据keep_indices筛选数据
    feature_filtered = feature[keep_indices]
    label_filtered = label[keep_indices]
    score_filtered = score[keep_indices]

    return feature_filtered, label_filtered, score_filtered

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='training arguments for the regression task')
    
    parser.add_argument('--epoch', default=50, type=int, help='number of training epochs')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--task', type=str, default='dude-smi',choices=['dude-smi','lit-smi','lit-pkl'])
    parser.add_argument('--name', type=str, default='aa2ar')
    parser.add_argument('--dim', type=int, default=2048)
    parser.add_argument('--types', type=str, default='bit',choices=['norm','bit'])
    parser.add_argument('--clip', action='store_true', default=False)
    parser.add_argument('--neighbor', type=int, default=50)
    parser.add_argument('--scale', type=float,default=1.0)
    parser.add_argument('--out_dim', default=1, type=int)
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument('--out_path', type=str, default='./model/model.pth')
    args = parser.parse_args()
    init_seed(args.seed)

    start_t = time.time()

    feature, label, score, _ = get_dataset(args.task,args.types,args.name,args.dim,args.clip)
    feature, label, score = remove_nan_entries(feature, label, score)
    if args.task == 'dude-smi':
        feature, score, label = feature[:-2000], score[:-2000], label[:-2000]
    
    test_dataloader = create_dataset(feature, np.arange(len(score)), score, 1024, False)
    model = MLP(input_size=args.dim, output_size=args.out_dim)
    model.load_state_dict(torch.load(args.out_path))
    model.eval()
    all_outputs = []

    with torch.no_grad():
        for inputs, _ in test_dataloader:
            outputs = model(inputs)
            all_outputs.extend(outputs)  # Convert to numpy array and accumulate
    gathered_outputs = torch.cat(all_outputs)
    all_outputs = gathered_outputs.cpu().numpy()
    real_auc = roc_auc_score(label, score)
    nn_auc = roc_auc_score(label, all_outputs)
    dev = np.sum(np.square((score[:50] - all_outputs[:50])) + np.square((score[-50:] - all_outputs[-50:])))/100
    # 计算top-1000召回
    sorted_indices_pred = np.argsort(-all_outputs)
    top_1k_indices_pred = sorted_indices_pred[:10000]
    sorted_indices_score = np.argsort(-score)  # sort in descending order
    top_1k_indices_score = sorted_indices_score[:10000]
    overlap = len(np.intersect1d(top_1k_indices_score, top_1k_indices_pred))/10000
    
    print(f"real auc: {real_auc}, predict_auc: {nn_auc}, error: {dev}, top-1k-overlap:{overlap}")
    
