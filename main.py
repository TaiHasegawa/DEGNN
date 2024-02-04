import json
import pickle
import argparse
import random
import torch
import numpy as np
import scipy.sparse as sp
from models.DEGNN import DEGNN
from models.utils import get_activation


def get_model(model, log, device, activation):
    """
    getter function for models
    """
    if model == 'DEGNN1':
        return DEGNN(device, activation, log=log, model_type=1)
    if model == 'DEGNN2':
        return DEGNN(device, activation, log=log, model_type=2)
    raise ValueError(f"model should be in [DEGNN1, DEGNN2], but got {model}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset',
                        choices=['cora', 'citeseer', 'amazon_photo', 'amazon_computers'])
    parser.add_argument('--model', type=str, default='DEGNN2',
                        choices=['DEGNN1', 'DEGNN2'])
    parser.add_argument("--num_exp", type=int, default=10, help='number of training experiments')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'none'])
    parser.add_argument("--edge_noise_ratio", type=float, default=0.)
    parser.add_argument("--node_noise_ratio", type=float, default=0.)

    args = parser.parse_args()
    
    args.cuda = args.device == 'cuda' and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"device: {device}")

    activation = get_activation(args.activation)
    
    tvt_nids = pickle.load(open(f'data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{args.dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())

    params_all = json.load(open('best_parameters.json', 'r'))
    params_dict = params_all[args.model][args.dataset]

    train_accs = []
    val_accs = []
    test_accs = []
    test_macrof1s = []
    for i in range(args.num_exp):
        random.seed(i)
        # init model
        model = get_model(args.model, False, device, activation)
        # load data
        model.load_data(tvt_nids, adj_orig, features, labels, args.edge_noise_ratio, args.node_noise_ratio)
        model.set_hyper_parameters(params_dict)
        # train model
        train_acc, val_acc, test_acc, test_macrof1 = model.fit()
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        test_macrof1s.append(test_macrof1)
    model.logger.info(f'-----------------------Results-----------------------')
    model.logger.info(f'train Micro F1: {np.mean(train_accs):.6f}, std: {np.std(train_accs):.6f}')
    model.logger.info(f'val Micro F1: {np.mean(val_accs):.6f}, std: {np.std(val_accs):.6f}')
    model.logger.info(f'test Micro F1: {np.mean(test_accs):.6f}, std: {np.std(test_accs):.6f}')
    model.logger.info(f'test Macro F1: {np.mean(test_macrof1s):.6f}, std: {np.std(test_macrof1s):.6f}')
    model.logger.info(f'-----------------------Parameters-----------------------\n\
                      {params_dict}')
    
    print(f"train Micro F1: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"val Micro F1: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"test Micro F1: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f'test Macro F1: {np.mean(test_macrof1s):.4f} ± {np.std(test_macrof1s):.4f}')