from typing import Optional
from itertools import chain
from functools import partial
import dgl
import torch
import torch.nn as nn
from .centroid import CC
from .gat import GAT
from .loss_func import sce_loss
from utils import create_norm, drop_edge
import torch.nn.functional as F

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rate):
        ctx.rate = rate
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.rate
        return grad_output, None

class GRL(nn.Module):
    def forward(self, input, lambd = 1.0):
        return GradReverse.apply(input, lambd)


class Node_Alignment_Choose(nn.Module):
    def __init__(self, nodes=150):
        super().__init__()
        self.nodes = nodes
        self.e2e = nn.Sequential(
            E2E(1, 8, (nodes, nodes),nodes),
            nn.LeakyReLU(0.33),
            E2E(8, 8, (nodes, nodes),nodes),  # 0.642
            nn.LeakyReLU(0.33),
        )

        self.e2n = nn.Sequential(
            nn.Conv2d(8, 256, (1, nodes)),  # 32 652
            nn.LeakyReLU(0.33),
        )

    def forward(self, A):
        x = self.e2e(A)
        x = self.e2n(x)
        x = x.reshape(self.nodes,-1)
        return x

class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, nodes,**kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.nodes = nodes

    def forward(self, A):
        A = A.view(1, self.in_channel, self.nodes, self.nodes)

        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)

        return concat1 + concat2


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            classes: int = 5,
            known_classes: int = 2
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._classes = classes
        self._known_classes = known_classes
        self.num_hidden = num_hidden

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder_s = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )
        self.decoder_t = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )
        self.cls_model = nn.Sequential(
            nn.Linear(enc_nhead * enc_num_hidden, self._known_classes),
        )
        self.domain_model = nn.Sequential(
            GRL(),
            nn.Linear(enc_nhead * enc_num_hidden, 10),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(10, 2),
            nn.Softmax(dim=1),
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.E2EN = Node_Alignment_Choose(nodes=50)
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.cls_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.NLLLoss()
        self.node_loss = nn.MSELoss()

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def E2EN_subgraph_to_graph(self, g, x):
        g_e = g.to('cuda')
        g_e.ndata['feat'] = x.cuda()
        g_attr = g.to('cuda')
        g_attr.ndata['feat'] = x.cuda()
        walks = dgl.sampling.random_walk(g_e.to('cpu'), torch.arange(g_e.number_of_nodes()).to('cpu'), length=50,
                                         restart_prob=0)
        for i in range(len(walks[0])):
            if g_e.out_degree(walks[0][i][0].to('cuda')) == 0:
                continue
            else:
                walks[0][i] = torch.masked_fill(walks[0][i], torch.eq(walks[0][i], -1), walks[0][i][0])
                only_random_walk = walks[0][i][1:]
                sub_graph_e = dgl.node_subgraph(g_e, only_random_walk.to('cuda'))
                sub_graph_attr = dgl.node_subgraph(g_e, only_random_walk.to('cuda'))
                A_e = sub_graph_e.adjacency_matrix()
                A_e_ = A_e.to_dense().cuda()
                x = self.E2EN(A_e_)
                sub_graph_e.ndata['feat'] = x
                output_node_e = dgl.mean_nodes(sub_graph_e, 'feat')
                g_e.nodes[walks[0][i][0].to('cuda')].data['feat'] = output_node_e
                output_node_attr = dgl.mean_nodes(sub_graph_attr, 'feat')
                g_attr.nodes[walks[0][i][0].to('cuda')].data['feat'] = output_node_attr
        return g_e, g_attr

    def forward(self, g_t, x_t, g_s, x_s, y_s,threshold,cluster,known_class):
        loss = self.mask_edge_prediction(g_t, x_t, g_s, x_s,y_s,threshold,cluster,known_class)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def create_node_to_edge(self, rec_x):
        x = rec_x
        x_t = x.permute(1, 0)
        A = torch.mm(x, x_t)
        A = torch.softmax(A,dim=-1)
        return A

    def node_alignment(self, use_g_t, rep_t, g_s, rep_s, pred_t, y_s, y_t_pseudo,unk_class):
        align_g = use_g_t.clone()
        align_g.ndata['feat'] = rep_t
        y_t = torch.argmax(pred_t, dim=1)
        '''node_alignment_E2E_E2N'''
        sub_graph_target_e, sub_graph_target_attr= self.E2EN_subgraph_to_graph(use_g_t, rep_t)
        sub_graph_source_e, sub_graph_source_attr= self.E2EN_subgraph_to_graph(g_s, rep_s)
        idx_e = self.calculate_two_similary(sub_graph_source_e, sub_graph_target_e)
        idx_attr = self.calculate_two_similary(sub_graph_source_attr, sub_graph_target_attr)
        one_hot_idx_e = torch.nn.functional.one_hot(idx_e, num_classes=g_s.number_of_nodes()).float()
        one_hot_idx_attr = torch.nn.functional.one_hot(idx_attr, num_classes=g_s.number_of_nodes()).float()
        loss_e = self.node_loss(one_hot_idx_e, one_hot_idx_attr)
        loss_attr = self.node_loss(one_hot_idx_attr, one_hot_idx_e)
        loss_node = loss_e + loss_attr
        unique_values = torch.unique(idx_e)
        num_unique_values = len(unique_values)
        print(f'Number of unique values in idx: {num_unique_values}')
        print('idx: =====', idx_e)
        for v in range(align_g.number_of_nodes()):
            idx_ = idx_e[v]
            if y_t_pseudo[v] != unk_class:
                if y_t[v] == y_s[idx_]:
                    align_g.nodes[v].data['feat'] = rep_s[idx_].unsqueeze(0)
        feature = align_g.ndata['feat']
        A = self.create_node_to_edge(feature)
        return A,loss_node

    def calculate_two_similary(self, g1, g2):
        x_s = g1.ndata['feat']
        x_t = g2.ndata['feat']
        graph_attention = torch.mm(x_t, x_s.t())
        a_x = nn.Softmax(dim=1)(graph_attention)  # i->j
        a_y = nn.Softmax(dim=0)(graph_attention)  # j->i
        idx = a_x.argmax(axis=1)
        return idx
    def set_source_open_set(self, x_s, y_s,known_class):
        if known_class == 1:
            mask_nodes = (y_s == 1)| (y_s == 2) | (y_s == 3) | (y_s == 4)
        elif known_class == 2:
            mask_nodes = (y_s == 2) | (y_s == 3) | (y_s == 4)
        elif known_class == 3:
            mask_nodes = (y_s == 3) | (y_s == 4)
        elif known_class == 4:
            mask_nodes = (y_s == 4)
        x_s[mask_nodes] = 0
        x_s_op = x_s[~mask_nodes]
        y_s_op = y_s[~mask_nodes]
        return x_s, x_s_op, y_s_op, mask_nodes

    # our model
    def mask_edge_prediction(self, g_t, x_t, g_s, x_s, y_s,threshold,cluster,known_class):
            # ---- edge reconstruction ----
            if self._drop_edge_rate > 0:
                use_g_t, masked_edges = drop_edge(g_t, self._drop_edge_rate, return_edges=True)
            else:
                use_g_t = g_t
            enc_rep_t, all_hidden_t = self.encoder(use_g_t, x_t, return_hidden=True)
            enc_rep_s, all_hidden_s = self.encoder(g_s, x_s, return_hidden=True)

            if self._concat_hidden:
                enc_rep_t = torch.cat(all_hidden_t, dim=1)
                enc_rep_s = torch.cat(all_hidden_s, dim=1)

            rep_t = self.encoder_to_decoder(enc_rep_t)
            rep_s = self.encoder_to_decoder(enc_rep_s)

            if self._decoder_type in ("mlp", "linear"):
                recon_t = self.decoder_t(rep_t)
                recon_s = self.decoder_s(rep_s)
            else:
                recon_t = self.decoder_t(use_g_t, rep_t)
                recon_s = self.decoder_s(g_s, rep_s)

            x_s_, x_s_op, y_s_op, mask_nodes_s = self.set_source_open_set(x_s, y_s,known_class)
            source_logits = self.cls_model(rep_s)
            train_masks = g_s.ndata["train_mask"]
            valid_train_mask = train_masks & (~mask_nodes_s)
            loss_source_node = self.cls_loss(source_logits[valid_train_mask], y_s[valid_train_mask])

            ''' compute entropy loss for T '''
            target_probs = self.cls_model(rep_t)
            probs = F.softmax(target_probs, dim=1)
            y_t_pseudo = target_probs.max(1)[1]
            probs = torch.clamp(probs, min=1e-9, max=1.0)
            entrs = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            unk_class = known_class
            pred_unk = torch.where(entrs > threshold)
            y_t_pseudo[pred_unk] = unk_class

            '''node alignment'''
            rec_adj_t, loss_node = self.node_alignment(use_g_t, rep_t, g_s, rep_s, target_probs, y_s,y_t_pseudo,known_class)
            mask_adj_t = use_g_t.adjacency_matrix()
            mask_adj_t_ = mask_adj_t.to_dense().cuda()
            rec_adj_t_ = rec_adj_t * mask_adj_t_
            loss_edge = self.criterion(rec_adj_t_, mask_adj_t_)

            loss_attr_t = self.criterion(recon_t, x_t)
            loss_attr_s = self.criterion(recon_s, x_s)

            loss_rec = loss_edge + loss_attr_s + loss_attr_t

            known_class_mask = (y_t_pseudo != unk_class)  # 生成布尔掩码
            cc = CC(num_known_classes=known_class, num_unknown_clusters=cluster, feature_dim=self.num_hidden, use_cuda=True)
            loss_cc, mask_known, mask_unknown = cc.loss_ncc(rep_s, y_s, rep_t, known_class_mask, y_t_pseudo)

            '''total'''
            loss =  loss_source_node + loss_cc + loss_rec + loss_node

            return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
