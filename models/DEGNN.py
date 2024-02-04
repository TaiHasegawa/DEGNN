import logging
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import sys
from copy import deepcopy

from models.utils import scipysp_to_pytorchsp, eval_node_cls, get_logger
from models.noise import add_noise
from models.degnn_layers import GCN_model, GCN_1layer, Reconstruct_adj, Discriminator_innerprod, BCEExeprtLoss


FIRST_PATIENCE = 100
PRE_PATIENCE = 50
FINE_TUNE_PATIENCE = 50

class DEGNN(object):
    """
    Implementation of DEGNN-I and DEGNN-II
    """
    def __init__(self, device, activation, seed=-1, log=True, name='debug', model_type=2):
        if log:
            self.logger = get_logger(name)
        else:
            self.logger = logging.getLogger()
            sh = logging.StreamHandler(sys.stdout)
            if not self.logger.hasHandlers():
                self.logger.addHandler(sh)
            self.logger.setLevel(logging.INFO)
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.model_type = model_type
        self.device = device
        self.activation = activation
        self.nc_criterion = nn.CrossEntropyLoss()

    def load_data(self, tvt_nids, adj_matrix, features, labels,
                  edge_noise_ratio=0., node_noise_ratio=0.):
        """
        Load and preprocess data.
        """
        self.input_dim = features.shape[1]

        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        self.n_nodes = labels.shape[0]
        self.n_classes = torch.unique(labels).shape[0]
        
        if isinstance(adj_matrix, tuple):
            adj_matrix = sp.coo_matrix(adj_matrix).astype(np.int64)
        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(1)
        adj_matrix, features = add_noise(adj_matrix, features.clone().detach(),
                                         edge_noise_ratio, node_noise_ratio)

        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        self.adj = scipysp_to_pytorchsp(adj_matrix).to_dense()
        self.n_edges = (self.adj - torch.eye(self.n_nodes)).count_nonzero().item()

        # normalize adj
        degrees = np.array(adj_matrix.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        self.adj_norm = scipysp_to_pytorchsp(adj_norm)
        
        # number of classes
        if len(self.labels.size()) == 1:
            self.out_size = len(torch.unique(self.labels))
        else:
            self.out_size = labels.size(1)

    def set_hyper_parameters(self, params_dict):
        """
        Set hyperparameters.
        """
        self.dropout = params_dict['dropout']
        self.emb_dim = params_dict['emb_dim']
        self.hidden_dim = params_dict['hidden_dim']
        self.epochs_pre = params_dict['epochs_pre']
        self.epochs = params_dict['epochs']
        self.lr = params_dict['lr']
        self.lr_pretrain = params_dict['lr_pretrain']
        self.weight_decay = params_dict['weight_decay']
        self.feature_noise_ratio = params_dict['feature_noise_ratio']
        self.edge_noise_ratio = params_dict['edge_noise_ratio']
        self.nf_loss_weight = params_dict['nf_loss_weight']
        self.ee_loss_weight = params_dict['ee_loss_weight']
        self.nf_discriminator = Discriminator_innerprod()
        self.edge_discriminator = Discriminator_innerprod()
        self.expert_criterion = BCEExeprtLoss(self.n_nodes, self.device)
        self.rec_adj = Reconstruct_adj(self.n_nodes, self.n_edges, params_dict['k_ratio'], self.device)

    def pre_train(self, nfe_encoder, ee_encoder, adj, adj_norm, features):
        """
        Pre-train the node feature expert and the edge expert.
        """
        optim_nfe = torch.optim.Adam(nfe_encoder.parameters(),
                                    lr=self.lr_pretrain,
                                    weight_decay=self.weight_decay)
        optim_ee = torch.optim.Adam(ee_encoder.parameters(),
                                    lr=self.lr_pretrain,
                                    weight_decay=self.weight_decay)
        nfe_encoder.train()
        ee_encoder.train()
        self.nfe_weights = deepcopy(nfe_encoder.state_dict())
        self.ee_weights = deepcopy(ee_encoder.state_dict())
        # ========== pre-train NF Expert ==========
        self.logger.info('Pre-training NF Expert')
        best_loss_val = 1e9
        best_epoch = 0
        cnt_wait = 0
        is_first_update = True
        for epoch in range(self.epochs_pre):
            nfe_encoder.train()
            features_aug, adj_aug = augment_graph(features, adj, self.feature_noise_ratio, self.edge_noise_ratio, self.device)
            features_neg, adj_neg = augment_negative_graph(features, adj, self.n_nodes, device=self.device)
            adj_aug_norm = normalize_adj(adj_aug, self.device)
            adj_neg_norm = normalize_adj(adj_neg, self.device)
            H = nfe_encoder(adj_norm, features)
            Haa = nfe_encoder(adj_norm, features_aug)
            Hta = nfe_encoder(adj_aug_norm, features)
            Hata = nfe_encoder(adj_aug_norm, features_aug)
            Hneg = nfe_encoder(adj_neg_norm, features_neg)
            logits_aa, logits_ta, logits_ata, logits_neg = self.nf_discriminator(H, Haa, Hta, Hata, Hneg)
            loss = self.expert_criterion(logits_aa, logits_ta, logits_ata, logits_neg)

            optim_nfe.zero_grad()
            loss.backward()
            optim_nfe.step()
            nfe_encoder.eval()

            # early stopping
            if loss < best_loss_val:
                if best_loss_val != 1e9:
                    is_first_update = False
                best_loss_val = loss
                best_epoch = epoch
                cnt_wait = 0
                self.nfe_weights = deepcopy(nfe_encoder.state_dict())
            else:
                cnt_wait += 1
            
            if is_first_update:
                if cnt_wait == FIRST_PATIENCE:
                    self.logger.info('Early stopping!')
                    break
            else:
                if cnt_wait == PRE_PATIENCE:
                    self.logger.info('Early stopping!')
                    break
            if epoch % 10 == 0:
                self.logger.info("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))
        self.logger.info('Loading {}th epoch'.format(best_epoch))
        nfe_encoder.load_state_dict(self.nfe_weights)

        # ========== pre-train Edge Expert ==========
        self.logger.info('Pre-training Edge Expert')
        best_loss_val = 1e9
        best_epoch = 0
        cnt_wait = 0
        is_first_update = True
        for epoch in range(self.epochs_pre):
            ee_encoder.train()
            features_aug, adj_aug = augment_graph(features, adj, self.feature_noise_ratio, self.edge_noise_ratio, self.device)
            features_neg, adj_neg = augment_negative_graph(features, adj, self.n_nodes, device=self.device)
            adj_aug_norm = normalize_adj(adj_aug, self.device)
            adj_neg_norm = normalize_adj(adj_neg, self.device)
            H = ee_encoder(adj_norm, features)
            Haa = ee_encoder(adj_norm, features_aug)
            Hta = ee_encoder(adj_aug_norm, features)
            Hata = ee_encoder(adj_aug_norm, features_aug)
            Hneg = ee_encoder(adj_neg_norm, features_neg)
            logits_aa, logits_ta, logits_ata, logits_neg = self.edge_discriminator(H, Haa, Hta, Hata, Hneg)
            loss = self.expert_criterion(logits_aa, logits_ta, logits_ata, logits_neg)    

            optim_ee.zero_grad()
            loss.backward()
            optim_ee.step()
            ee_encoder.eval()
            
            # early stopping
            if loss < best_loss_val:
                if best_loss_val != 1e9:
                    is_first_update = False
                best_loss_val = loss
                best_epoch = epoch
                cnt_wait = 0
                self.ee_weights = deepcopy(ee_encoder.state_dict())
            else:
                cnt_wait += 1
            
            if is_first_update:
                if cnt_wait == FIRST_PATIENCE:
                    self.logger.info('Early stopping!')
                    break
            else:
                if cnt_wait == PRE_PATIENCE:
                    self.logger.info('Early stopping!')
                    break
            if epoch % 10 == 0:
                self.logger.info("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))
        self.logger.info('Loading {}th epoch'.format(best_epoch))
        ee_encoder.load_state_dict(self.ee_weights)

        return nfe_encoder, ee_encoder


    def fine_tune(self, nfe_encoder, ee_encoder, gnn, adj, adj_norm, features, labels):
        """
        Fine-tune the entire network for DEGNN-I.
        """
        optims = MultipleOptimizer(torch.optim.Adam(nfe_encoder.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay),
                            torch.optim.Adam(ee_encoder.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay),
                            torch.optim.Adam(gnn.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay))        

        best_acc_val = 0
        best_acc_test = 0
        best_epoch = 0
        cnt_wait = 0
        loss_nf = 0
        loss_ee = 0
        for epoch in range(self.epochs):
            nfe_encoder.train()
            ee_encoder.train()
            gnn.train()

            features_aug, adj_aug = augment_graph(features, adj, self.feature_noise_ratio, self.edge_noise_ratio, self.device)
            features_neg, adj_neg = augment_negative_graph(features, adj, self.n_nodes, device=self.device)
            adj_aug_norm = normalize_adj(adj_aug, self.device)
            adj_neg_norm = normalize_adj(adj_neg, self.device)

            H = nfe_encoder(adj_norm, features)
            Haa = nfe_encoder(adj_norm, features_aug)
            Hta = nfe_encoder(adj_aug_norm, features)
            Hata = nfe_encoder(adj_aug_norm, features_aug)
            Hneg = nfe_encoder(adj_neg_norm, features_neg)
            logits_aa, logits_ta, logits_ata, logits_neg = self.nf_discriminator(H, Haa, Hta, Hata, Hneg)
            loss_nf = self.expert_criterion(logits_aa, logits_ta, logits_ata, logits_neg)

            H_ee = ee_encoder(adj_norm, features)
            Haa_ee = ee_encoder(adj_norm, features_aug)
            Hta_ee = ee_encoder(adj_aug_norm, features)
            Hata_ee = ee_encoder(adj_aug_norm, features_aug)
            Hneg_ee = ee_encoder(adj_neg_norm, features_neg)
            logits_aa_ee, logits_ta_ee, logits_ata_ee, logits_neg_ee = self.edge_discriminator(H_ee, Haa_ee, Hta_ee, Hata_ee, Hneg_ee)
            loss_ee = self.expert_criterion(logits_aa_ee, logits_ta_ee, logits_ata_ee, logits_neg_ee)
            new_adj = self.rec_adj(adj, H_ee)
            new_adj = normalize_adj(new_adj, self.device)

            logits = gnn(new_adj, H)
            loss_nc = self.nc_criterion(logits[self.train_nid], labels[self.train_nid])
            loss = loss_nc + loss_nf * self.nf_loss_weight + loss_ee * self.ee_loss_weight

            optims.zero_grad()
            loss.backward()
            optims.step()
            nfe_encoder.eval()
            ee_encoder.eval()
            gnn.eval()
            # use acc of previous model for efficiency
            val_acc = eval_node_cls(logits[self.val_nid], labels[self.val_nid])
            test_acc = eval_node_cls(logits[self.test_nid], labels[self.test_nid])

            if val_acc > best_acc_val:
                best_acc_val = val_acc
                best_acc_test = test_acc
                best_epoch = epoch
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == FINE_TUNE_PATIENCE:
                    self.logger.info('Early stop!')
                    break
            if epoch % 10 == 0:
                self.logger.info("Epoch {:05d} | ACC {:.4f} Loss {:.4f},".format(epoch, val_acc, loss.item()))
        self.logger.info('Best Acc at {}th epoch | {:.4f}'.format(best_epoch, best_acc_test))
        return best_acc_val, best_acc_test


    def modular_train_gnn(self, nfe_encoder, ee_encoder, gnn, adj, adj_norm, features, labels):
        """
        Train the downstream network for DEGNN-II.
        """
        optims = MultipleOptimizer(torch.optim.Adam(nfe_encoder.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay),
                            torch.optim.Adam(ee_encoder.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay),
                            torch.optim.Adam(gnn.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay))
 
        nfe_encoder.train()
        ee_encoder.train()
        gnn.train()
        best_acc_val = 0
        best_acc_test = 0
        best_epoch = 0
        cnt_wait = 0

        H = nfe_encoder(adj_norm, features)
        H_ee = ee_encoder(adj_norm, features)
        new_adj = self.rec_adj(adj, H_ee)
        new_adj = normalize_adj(new_adj, self.device)

        H = H.clone().detach().to(self.device)
        new_adj = new_adj.clone().detach().to(self.device)
        for epoch in range(self.epochs):
            nfe_encoder.train()
            ee_encoder.train()
            gnn.train()
            logits = gnn(new_adj, H)
            loss = self.nc_criterion(logits[self.train_nid], labels[self.train_nid])
            optims.zero_grad()
            loss.backward()
            optims.step()
            nfe_encoder.eval()
            ee_encoder.eval()
            gnn.eval()

            # use acc of previous model for efficiency
            val_acc, test_acc = self.evaluate(logits, labels, self.val_nid, self.test_nid)
            if val_acc > best_acc_val:
                best_acc_val = val_acc
                best_acc_test = test_acc
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == FINE_TUNE_PATIENCE:
                    self.logger.info('Early stop!')
                    break
            if epoch % 10 == 0:
                self.logger.info("Epoch {:05d} | ACC {:.4f} Loss {:.4f},".format(epoch, val_acc, loss.item()))
        self.logger.info('Best Acc at {}th epoch | {:.4f}'.format(best_epoch, best_acc_test))
        return best_acc_val, best_acc_test


    def fit(self):
        """
        Train the model.
        """
        adj = self.adj.to(self.device)
        adj_norm = self.adj_norm.to(self.device)
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)

        nfe_encoder = GCN_1layer(features.shape[1], self.emb_dim, self.dropout).to(self.device)
        ee_encoder = GCN_1layer(features.shape[1], self.emb_dim, self.dropout).to(self.device)
        gnn = GCN_model(self.emb_dim, self.hidden_dim, self.n_classes, self.dropout).to(self.device)

        nfe_encoder, ee_encoder = self.pre_train(nfe_encoder, ee_encoder, adj, adj_norm, features)
        if self.model_type == 1:
            val_acc, test_acc = self.fine_tune(nfe_encoder, ee_encoder, gnn, adj, adj_norm, features, labels)
        elif self.model_type == 2:
            val_acc, test_acc = self.modular_train_gnn(nfe_encoder, ee_encoder, gnn, adj, adj_norm, features, labels)
        else:
            raise ValueError(f"model_type should be in [1, 2], but got {self.model_type}.")

        return 0, val_acc, test_acc, 0
    
    def evaluate(self, logits, labels, val_nid, test_nid):
        """
        Evaluate the model.
        """
        labels_val = labels[val_nid]
        _, indices_val = logits[val_nid].max(dim=1)
        correct_val = torch.sum(indices_val == labels_val)
        labels_test = labels[test_nid]
        _, indices_test = logits[test_nid].max(dim=1)
        correct_test = torch.sum(indices_test == labels_test)
        return correct_val.item() * 1.0 / labels_val.shape[0], correct_test.item() * 1.0 / labels_test.shape[0]
    

def normalize_adj(adj, device='cpu'):
    """
    normalize adj GCN
    """
    n_nodes = adj.shape[0]
    adj_norm = adj
    adj_norm = adj_norm * (torch.ones(n_nodes).to(device) - torch.eye(n_nodes).to(device)) + torch.eye(n_nodes).to(device)
    D_norm = torch.diag(torch.pow(adj_norm.sum(1), -0.5)).to(device)
    adj_norm = D_norm @ adj_norm @ D_norm
    return adj_norm

def augment_graph(features, adj, feature_noise_ratio, edge_noise_ratio, device='cpu'):
    """
    Augmentation process of experts.
    """
    features_aug, adj_aug = features, adj
    if feature_noise_ratio > 0:
        features_aug = augment_features(features, feature_noise_ratio, device)
    if edge_noise_ratio > 0:
        adj_aug = augment_adj(adj, edge_noise_ratio, device)
    return features_aug, adj_aug

def augment_negative_graph(features, adj, n_nodes, device='cpu'):
    """
    Augment the negative graph.
    """
    features_aug = shuffle_NF(features, device)
    adj_aug = augment_negative_adj(adj, n_nodes, device)
    return features_aug, adj_aug

def augment_features(features, noise_ratio, device='cpu'):
    """
    Augment node features for the positive graph by shuffling elements in the node feature matrix in each row.
    """
    features_shuffled = features.clone().detach()
    num_elements_to_shuffle = int(features_shuffled.shape[1] * noise_ratio)
    for i in range(features_shuffled.shape[0]):
        # get indices to be shuffled
        indices = torch.randperm(features_shuffled.shape[1])[:num_elements_to_shuffle]
        # shuffle selected element
        selected_elements = features_shuffled[i, indices]
        features_shuffled[i, indices] = selected_elements[torch.randperm(selected_elements.shape[0])]
    return features_shuffled.to(device)

def shuffle_NF(features, device='cpu'):
    """
    Augment node features for the negative graph by shuffling the rows of the node feature matrix.
    """
    features_shuffled = features.clone().detach()
    idx = np.random.permutation(features.shape[0])
    features_shuffled = features_shuffled[idx, :]
    return features_shuffled.to(device)

def augment_adj(adj, noise_ratio=0.1, device='cpu'):
    """
    Augment edges for the positive graph by rewiring the edges.
    """
    mask = (torch.bernoulli(torch.ones_like(adj).to(device) * noise_ratio) == 1).to(device)
    perturbed_adj = torch.where(mask, 1 - adj, adj)
    return perturbed_adj

def augment_negative_adj(adj, n_nodes, device='cpu'):
    """
    Augment edges for the negative graph by completely rewiring the edges.
    """
    noise_ratio = (adj.sum().item() - n_nodes) / (1 - adj).sum().item()
    mask = (torch.bernoulli(torch.ones_like(adj).to(device) * noise_ratio) == 1).to(device)
    perturbed_adj = torch.where(mask, 1 - adj, 0) + torch.eye(n_nodes).to(device)
    return perturbed_adj

class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr