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

def get_subset(feature, label, score):
    # Ensure all inputs have the same length
    assert len(feature) == len(label) == len(score)
    
    total_elements = len(score)
    sorted_indeces = np.argsort(-score)

    if total_elements >= 100000:
        k = 100000
    else:
        k = total_elements
    
    # Get indices of top k scores
    top_indices = sorted_indeces[:k]
    selected_indices = top_indices
    
    # Sort the indices to maintain original order
    selected_indices.sort()
    
    # Subset the input arrays/list using the selected indices
    feature_subset = feature[selected_indices]
    label_subset = label[selected_indices]
    score_subset = score[selected_indices]
    
    return feature_subset, label_subset, score_subset

def random_score_regression(args, feature, test_dataloader, label, score):
    # active training for top-score regression
    # Device configuration
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model, loss, optimizer, and scheduler
    model = MLP(input_size=args.dim, output_size=args.out_dim).to(device)
    criterion = nn.MSELoss()
    
    # Sort data by true scores in descending order
    # mean = 7  
    # std_dev = np.sqrt(3)   
    # score_g = np.random.normal(loc=mean, scale=std_dev, size=score.shape)
    # score_g = score_g.astype('float32')
    # score_g = np.clip(score_g,3,15)

    score_g = np.random.uniform(low=3, high=15, size=score.shape)
    score_g = score_g.astype('float32')

    sorted_indices = np.argsort(-score_g)
    
    # Initial training set: top 1000 molecules
    train_indices = set(sorted_indices[:1000])
    all_indices = set(range(len(score)))
    
    for iteration in range(3):  # Initial + 2 iterations
        if iteration > 0:
            # Predict scores for all molecules
            model.eval()
            all_predictions = []
            with torch.no_grad():
                for inputs, _ in test_dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    all_predictions.extend(outputs)
            predictions = torch.cat(all_predictions)
            all_predictions = predictions.cpu().numpy()
            
            # Select top 'args.num' predictions from unselected molecules
            unselected_indices = list(all_indices - train_indices)
            unselected_predictions = [all_predictions[i] for i in unselected_indices]
            top_indices = np.argsort(-np.array(unselected_predictions))[:args.num]
            new_indices = [unselected_indices[i] for i in top_indices]
            
            # Update training set
            train_indices.update(new_indices)
            print(f"Iteration {iteration}: Added {len(new_indices)} new training samples.")
        
        # Create new dataloader with current training set
        train_subset_indices = np.array(list(train_indices))
        print(f'Number of training molecules:{train_subset_indices.shape[0]}')
        train_loader = create_dataset(feature, train_subset_indices, score_g, args.bs, shuffle=True)
        
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.2)
        for ep in range(args.epoch):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)  # Ensure labels have shape (batch_size, 1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)  # Accumulate loss over the epoch
            
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader.dataset)
            
            # if ep % 3 == 0 or ep == args.epoch -1:
            #     lr = scheduler.get_last_lr()[0]
            #     print(f'Iteration:{iteration}, Epoch:{ep}, Loss:{avg_loss:.5f}, LR:{lr}')
    
    # Final evaluation
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.extend(outputs)  # 这里注意append和extend完全不一样，会导致后面出现广播机制
    
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    
    # Calculate metrics
    real_auc = roc_auc_score(label, score)
    nn_auc = roc_auc_score(label, all_outputs)
    
    print(f"Real AUC: {real_auc}, Predict AUC: {nn_auc}")
    
    # Save the trained model
    model.cpu()
    torch.save(model.state_dict(), args.out_path)
    print(f"Model saved to {args.out_path}")

def active_score_regression(args, feature, test_dataloader, label, score):
    # active training for top-score regression
    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')
    
    # Initialize model, loss, optimizer, and scheduler
    model = MLP(input_size=args.dim, output_size=args.out_dim).to(device)
    criterion = nn.MSELoss()
    
    # Sort data by true scores in descending order
    sorted_indices = np.argsort(-score)
    
    # Initial training set: top 1000 molecules
    train_indices = set(sorted_indices[:1000])
    all_indices = set(range(len(score)))
    
    for iteration in range(3):  # Initial + 2 iterations
        if iteration > 0:
            # Predict scores for all molecules
            model.eval()
            all_predictions = []
            with torch.no_grad():
                for inputs, _ in test_dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    all_predictions.extend(outputs)
            predictions = torch.cat(all_predictions)
            all_predictions = predictions.cpu().numpy()
            
            # Select top 'args.num' predictions from unselected molecules
            unselected_indices = list(all_indices - train_indices)
            unselected_predictions = [all_predictions[i] for i in unselected_indices]
            top_indices = np.argsort(-np.array(unselected_predictions))[:args.num]
            new_indices = [unselected_indices[i] for i in top_indices]
            
            # Update training set
            train_indices.update(new_indices)
            print(f"Iteration {iteration}: Added {len(new_indices)} new training samples.")
        
        # Create new dataloader with current training set
        train_subset_indices = np.array(list(train_indices))
        print(f'Number of training molecules:{train_subset_indices.shape[0]}')
        train_loader = create_dataset(feature, train_subset_indices, score, args.bs, shuffle=True)
        
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.2)
        for ep in range(args.epoch):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)  # Ensure labels have shape (batch_size, 1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)  # Accumulate loss over the epoch
            
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader.dataset)
            
            # if ep % 3 == 0 or ep == args.epoch -1:
            #     lr = scheduler.get_last_lr()[0]
            #     print(f'Iteration:{iteration}, Epoch:{ep}, Loss:{avg_loss:.5f}, LR:{lr}')
    
    # Final evaluation
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.extend(outputs)  # 这里注意append和extend完全不一样，会导致后面出现广播机制
    
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    
    # Calculate metrics
    real_auc = roc_auc_score(label, score)
    nn_auc = roc_auc_score(label, all_outputs)
    dev = np.sum(np.square((score[:50] - all_outputs[:50])) + np.square((score[-50:] - all_outputs[-50:])))/100
    sorted_indices_pred = np.argsort(-all_outputs)
    sorted_indices_score = np.argsort(-score)
    overlap_50 = len(np.intersect1d(sorted_indices_pred[:50], sorted_indices_score[:50])) / 50
    overlap_100 = len(np.intersect1d(sorted_indices_pred[:100], sorted_indices_score[:50])) / 50
    
    print(f"Real AUC: {real_auc}, Predict AUC: {nn_auc}, Error: {dev} Overlap: top-50: {overlap_50:.2f}, top-100: {overlap_100:.2f}")
    
    # Save the trained model
    model.cpu()
    torch.save(model.state_dict(), args.out_path)
    print(f"Model saved to {args.out_path}")


def score_regression(args, dataloader, test_dataloader, label, score):
    # parallel training for large datasets
    accelerator = Accelerator()

    model = MLP(input_size=args.dim, output_size=args.out_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
    
    dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

    model.train()
    for ep in range(args.epoch):
        epoch_loss = 0
        for inputs, labels in dataloader:
        # for inputs, labels in tqdm(dataloader, desc="Processing batches"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            accelerator.backward(loss)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.detach() * inputs.size(0)

        # Gather losses from all processes
        gathered_loss = accelerator.gather(epoch_loss)
        gathered_loss = torch.sum(gathered_loss) / len(dataloader.dataset)
        scheduler.step()
        if ep%3==0:
            lr = scheduler.get_last_lr()[0]
            print(f'Epoch:{ep}, loss:{gathered_loss:.5f}, lr:{lr}')
    
    # Switch to evaluation mode
    model = accelerator.unwrap_model(model)
    model.eval()
    device = next(model.parameters()).device

    all_outputs = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_dataloader):
            outputs = model(inputs.to(device))
            all_outputs.extend(outputs)  # Convert to numpy array and accumulate
    gathered_outputs = torch.cat(all_outputs)
    all_outputs = gathered_outputs.cpu().numpy()

    # 计算真实分数的AUC
    real_auc = roc_auc_score(label, score)
    nn_auc = roc_auc_score(label, all_outputs)
    dev = np.sum(np.square((score[:50] - all_outputs[:50])) + np.square((score[-50:] - all_outputs[-50:])))/100
    # 计算top-1000召回
    sorted_indices_pred = np.argsort(-all_outputs)
    sorted_indices_score = np.argsort(-score)  # sort in descending order
    overlap_200 = len(np.intersect1d(sorted_indices_pred[:200], sorted_indices_score[:50]))/50
    overlap_100 = len(np.intersect1d(sorted_indices_pred[:100], sorted_indices_score[:50]))/50
    
    print(f"real auc: {real_auc}, predict_auc: {nn_auc}, error: {dev}, overlap: top-100:{overlap_100}, top-200:{overlap_200}")
    model.cpu()
    torch.save(model.state_dict(), args.out_path)
    # print(f"real top score: {score[top_1k_indices_score[:20]]}, corres pred score: {all_outputs[top_1k_indices_score[:20]]}")
    # print(f"corres real score: {score[top_1k_indices_pred[:20]]}, pred top score: {all_outputs[top_1k_indices_pred[:20]]}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='training arguments for the regression task')
    
    parser.add_argument('--epoch', default=50, type=int, help='number of training epochs')
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--num', type=int, default=25000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--task', type=str, default='lit-smi',choices=['dude-smi','lit-smi','lit-pkl'])
    parser.add_argument('--name', type=str, default='akt1')
    parser.add_argument('--dim', type=int, default=2048)
    parser.add_argument('--types', type=str, default='bit',choices=['norm','bit'])
    parser.add_argument('--clip', action='store_true', default=False)
    parser.add_argument('--neighbor', type=int, default=50)
    parser.add_argument('--scale', type=float,default=1.0)
    parser.add_argument('--out_dim', default=1, type=int)
    parser.add_argument('--out_path', type=str, default='./models/model.pth')
    args = parser.parse_args()
    init_seed(args.seed)

    start_t = time.time()

    feature, label, score, ecfp, smiles = get_dataset(args.task,args.types,args.name,args.dim,args.clip)
    feature, label, score = remove_nan_entries(feature, label, score)
    if args.task == 'dude-smi':
        feature, score, label = feature[:-2000], score[:-2000], label[:-2000]
        
    if args.task == 'lit-pkl':
        if np.max(score) < 0:
            raise ValueError('score too bad for virtual screening!')
        score = (score - np.min(score)) / np.max(score) * 13
    score = score.astype('float32')
    # select a subset for regression
    # feature, label, score = get_subset(feature, label, score)
    print(f'Number of molecules: {feature.shape[0]}')

    train_dataloader = create_dataset(feature, np.arange(len(score)), score, args.bs, True)
    test_dataloader = create_dataset(feature, np.arange(len(score)), score, 4096, False)
    # score_regression(args, train_dataloader, test_dataloader, label, score)
    # active_score_regression(args, feature, test_dataloader, label, score)
    random_score_regression(args, feature, test_dataloader, label, score)
    end_t = time.time()
    duration = end_t - start_t
    print(f'training time: {duration}')