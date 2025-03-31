import copy

import numpy
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve, accuracy_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_accuracies(preds, y_true, unk_class):
    known_mask = y_true != unk_class
    unknown_mask = y_true == unk_class

    known_preds = preds[known_mask]
    known_true = y_true[known_mask]

    unknown_preds = preds[unknown_mask]
    unknown_true = y_true[unknown_mask]

    acc_known = accuracy_score(known_true.cpu().numpy(), known_preds.cpu().numpy())
    acc_unknown = accuracy_score(unknown_true.cpu().numpy(), unknown_preds.cpu().numpy())
    overall_acc = accuracy_score(y_true.cpu().numpy(), preds.cpu().numpy())
    return acc_known, acc_unknown, overall_acc


def calculate_h_score(acc_known, acc_unknown):
    if acc_known + acc_unknown == 0:
        return 0
    return 2 * (acc_known * acc_unknown) / (acc_known + acc_unknown)

def direct_node_classification_evaluation_openset_uk(model, graph, x, threshold,device,known_class,unknown_class):
    model.eval()
    encoder = model.encoder
    dd = model.encoder_to_decoder
    cls_model = model.cls_model

    encoder.to(device)
    dd.to(device)
    cls_model.to(device)
    graph = graph.to(device)
    x = x.to(device)
    labels = graph.ndata["label"].to(device)
    known_class_labels = torch.arange(known_class + 1).to(device)
    known_class_mask = labels.unsqueeze(1) == known_class_labels
    known_class_mask = known_class_mask.any(dim=1)

    train_masks_target = graph.ndata["train_mask"].to(device)
    valid_train_mask_t = train_masks_target & known_class_mask

    with torch.no_grad():
        enc_rep, _ = encoder(graph, x, return_hidden=True)
        rep_t = dd(enc_rep)
        pred = cls_model(rep_t)
        y_true = labels.squeeze().long()
        print('y_true[valid_train_mask_s]: ', y_true[valid_train_mask_t])
        preds = pred.max(1)[1].type_as(y_true)

        probs = F.softmax(pred, dim=1)
        probs = torch.clamp(probs, min=1e-9, max=1.0)
        entrs = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        print('entrs: ', entrs)

        pred_unk = torch.where(entrs > threshold)
        preds[pred_unk] = known_class
        unique_entropy_values, counts = torch.unique(preds[valid_train_mask_t], return_counts=True)
        real_entropy_values, counts_real = torch.unique(y_true[valid_train_mask_t], return_counts=True)

        for value_pred, value_real, count, count_real in zip(unique_entropy_values, real_entropy_values,counts,counts_real):
            print(f"Entropy Value: {value_pred},  Count: {count}, Real_label, {value_real}, Cout_real: {count_real}")

        print('preds: ', preds, ' labels: ', labels, ' y_true: ', y_true)

        acc_known, acc_unknown, acc_total = calculate_accuracies(preds[valid_train_mask_t], y_true[valid_train_mask_t], known_class)
        h_score = calculate_h_score(acc_known, acc_unknown)

    return acc_known, acc_unknown, acc_total, h_score
