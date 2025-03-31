import logging
import torch
import dgl
import numpy as np
from tqdm import tqdm

from dataset import Dataset

from utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from evaluation import direct_node_classification_evaluation_openset_uk
from models import build_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def pretrain(model, graph_target, feat_target, graph_source, feat_source, y_source,threshold,cluster, known_class, unknown_class,optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")

    args = build_args()
    graph_target = graph_target.to(device)
    x_target = feat_target.to(device)
    graph_source = graph_source.to(device)
    x_source = feat_source.to(device)
    y_source = y_source.to(device)
    epoch_iter = tqdm(range(max_epoch))
    best_target_acck = 0.0
    best_target_accu = 0.0
    best_target_acch = 0.0
    best_target_acc = 0.0
    best_epoch = 0.0
    model = model.to(device)
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph_target, x_target, graph_source, x_source, y_source, threshold,cluster,known_class)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        acc_known, acc_unknown, acc_total, h_score = direct_node_classification_evaluation_openset_uk(model, graph_target, x_target,threshold,
                                                                                        device,known_class,unknown_class)
        if h_score > best_target_acch:
            best_epoch = epoch
            best_target_acck = acc_known
            best_target_accu = acc_unknown
            best_target_acc = acc_total
            best_target_acch = h_score

        if (epoch + 1) % args.max_epoch == 0:
            print('best_epochL ',best_epoch,' best_acck: ',best_target_acck, best_target_accu, best_target_acc,best_target_acch)
    return model


def preprocess(graph):
    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph

def u_cat_e(edges):
  return {'m': torch.hstack([edges.src['feature'],edges.data['feature']])}

def mean_udf(nodes):
    return {'neigh_features': nodes.mailbox['m'].mean(1)}

def data_split(y,train_size):
    seeds = args.seeds
    for i, seed in enumerate(seeds):
        set_random_seed(seed)
    random_node_indices = np.random.permutation(y.shape[0])
    training_size = int(len(random_node_indices) * train_size)
    train_node_indices = random_node_indices[:training_size]
    test_node_indices = random_node_indices[training_size:]
    train_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    train_masks[train_node_indices] = 1
    test_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    test_masks[test_node_indices] = 1
    return train_masks,test_masks

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    target = args.target
    source = args.source
    threshold = args.threshold
    cluster = args.cluster
    known_class = args.known_class
    unknown_class = args.unknown_class
    source_train_rate = args.source_train_rate
    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    dataset_target = Dataset("data_/{}".format(args.target), name=target)
    dataset_source = Dataset("data_/{}".format(args.source), name=source)

    target_data = dataset_target[0]
    t_src = target_data.edge_index[0]
    t_dst = target_data.edge_index[1]
    graph_target = dgl.graph((t_src, t_dst))
    graph_target = dgl.to_bidirected(graph_target)
    graph_target = graph_target.remove_self_loop().add_self_loop()
    graph_target.create_formats_()

    source_data = dataset_source[0]
    s_src = source_data.edge_index[0]
    s_dst = source_data.edge_index[1]
    graph_source = dgl.graph((s_src, s_dst))
    graph_source = dgl.to_bidirected(graph_source)
    graph_source = graph_source.remove_self_loop().add_self_loop()
    graph_source.create_formats_()

    '''target data split'''
    t_train_masks, t_test_masks = data_split(y=target_data.y,train_size=1.0)
    s_train_masks, s_test_masks = data_split(y=source_data.y, train_size=source_train_rate)
    print('graph_target: ', graph_target,' graph_source: ',graph_source)

    print('target_data: ',target_data,' source_data: ',source_data)
    graph_target.ndata['feat'] = target_data.x
    graph_target.ndata['label'] = target_data.y
    graph_target.ndata['train_mask'] = t_train_masks
    graph_target.ndata['test_mask'] = t_test_masks
    graph_source.ndata['feat'] = source_data.x
    graph_source.ndata['label'] = source_data.y
    graph_source.ndata['train_mask'] = s_train_masks
    target_labels = target_data.y.clone()

    if known_class == 1 and unknown_class == 1:
        target_labels[(target_labels == 1)] = 10
        target_labels[(target_labels == 4)] = 1
        target_labels[(target_labels == 10)] = 4
    elif known_class == 1 and unknown_class == 2:
        target_labels[(target_labels == 1)] = 10
        target_labels[(target_labels == 4) | (target_labels == 3)] = 1
        target_labels[(target_labels == 10)] = 4
    elif known_class == 1 and unknown_class == 3:
        target_labels[(target_labels == 1)] = 10
        target_labels[(target_labels == 4) | (target_labels == 3) | (target_labels == 2) ] = 1
        target_labels[(target_labels == 10)] = 4
    elif known_class == 1 and unknown_class == 4:
        target_labels[(target_labels == 1)] = 10
        target_labels[(target_labels == 4) | (target_labels == 3) | (target_labels == 2) | (target_labels == 1)] = 1
        target_labels[(target_labels == 10)] = 4
    elif known_class == 2 and unknown_class == 1:
        target_labels[(target_labels == 2)] = 10
        target_labels[(target_labels == 4)] = 2
        target_labels[(target_labels == 10)] = 4
    elif known_class == 2 and unknown_class == 2:
        target_labels[(target_labels == 2)] = 10
        target_labels[(target_labels == 4) | (target_labels == 3)] = 2
        target_labels[(target_labels == 10)] = 4
    elif known_class == 2 and unknown_class ==3:
        target_labels[(target_labels == 2)] = 10
        target_labels[(target_labels == 4) | (target_labels == 3) | (target_labels == 2)] = 2
        target_labels[(target_labels == 10)] = 4
    elif known_class == 3 and unknown_class == 1:
        target_labels[(target_labels == 3)] = 10
        target_labels[(target_labels == 4) ] = 3
        target_labels[(target_labels == 10)] = 4
    elif known_class == 3 and unknown_class == 2:
        target_labels[(target_labels == 4) | (target_labels == 3)] = 3
    graph_target.ndata['label'] = target_labels

    num_features = args.features
    num_classes = args.classes
    args.num_features = num_features

    acc_known_list = []
    acc_unknown_list = []
    acc_total_list = []
    h_score_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None
        model = build_model(args).to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x_target = graph_target.ndata["feat"]
        x_source = graph_source.ndata["feat"]
        y_source = graph_source.ndata["label"]
        if not load_model:
            model = pretrain(model, graph_target, x_target, graph_source, x_source, y_source, threshold,cluster, known_class,unknown_class,optimizer, max_epoch, device, scheduler, num_classes, lr_f,
                             weight_decay_f, max_epoch_f, linear_prob, logger)
            # model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load('models/' + args.type + '/' + str(args.drop_edge_rate) + '_' + str(
                args.lr) + '_' + "checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        acc_known, acc_unknown, acc_total, h_score = direct_node_classification_evaluation_openset_uk(model, graph_target, x_target, threshold,device,known_class,unknown_class)
        acc_known_list.append(acc_known)
        acc_unknown_list.append(acc_unknown)
        acc_total_list.append(acc_total)
        h_score_list.append(h_score)

        if logger is not None:
            logger.finish()
    acc_known, acc_known_std = np.mean(acc_known_list), np.std(acc_known_list)
    acc_unknown, acc_unknown_std = np.mean(acc_unknown_list), np.std(acc_unknown_list)
    acc_total, acc_total_std = np.mean(acc_total_list), np.std(acc_total_list)
    h_score, h_score_std = np.mean(h_score_list), np.std(h_score_list)
    print(f"# early-stopping_acck: {acc_known:.4f}±{acc_known_std:.4f}")
    print(f"# early-stopping_accu: {acc_unknown:.4f}±{acc_unknown_std:.4f}")
    print(f"# early-stopping_acc: {acc_total:.4f}±{acc_total_std:.4f}")
    print(f"# early-stopping_acch: {h_score:.4f}±{h_score_std:.4f}")

if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
