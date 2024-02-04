import logging
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F


def scipysp_to_pytorchsp(sp_mx):
    """
    converts scipy sparse matrix to pytorch sparse matrix
    """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx

def eval_node_cls(nc_logits, labels):
    """
    evaluate node classification results
    """
    if len(labels.size()) == 2:
        preds = torch.round(torch.sigmoid(nc_logits))
        tp = len(torch.nonzero(preds * labels))
        tn = len(torch.nonzero((1-preds) * (1-labels)))
        fp = len(torch.nonzero(preds * (1-labels)))
        fn = len(torch.nonzero((1-preds) * labels))
        pre, rec, f1 = 0., 0., 0.
        if tp+fp > 0:
            pre = tp / (tp + fp)
        if tp+fn > 0:
            rec = tp / (tp + fn)
        if pre+rec > 0:
            fmeasure = (2 * pre * rec) / (pre + rec)
    else:
        preds = torch.argmax(nc_logits, dim=1)
        correct = torch.sum(preds == labels)
        fmeasure = correct.item() / len(labels)
    return fmeasure

def get_logger(name):
    """
    create a logger
    """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if name is not None:
        fh = logging.FileHandler(f'DEGNN-{name}.log')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger

def get_activation(activation):
    """
    Gen an activation function.
    """
    if activation == 'relu':
        return F.relu
    if activation == 'tanh':
        return torch.tanh
    return None
