from math import sqrt
import pdb
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

num_atom_type = 121 #including the extra motif tokens and graph token
num_formal_charge = 11
num_degree = 11  #degree
num_chirality = 4
num_hybridization = 7
num_numH = 9
num_implicit_valence = 7
num_bond_type = 7 # motif graph and self-loop
num_bond_inring = 3 
num_bond_dir = 3
num_bond_stereo = 6

num_bond_features = 4


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 

    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__(aggr=aggr)
        #multi-layer perceptron
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2*emb_dim), nn.ReLU(), nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_inring, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_dir, emb_dim)
        self.edge_embedding4 = nn.Embedding(num_bond_stereo, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding4.weight.data)

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), num_bond_features)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + \
                        self.edge_embedding3(edge_attr[:,2])  + self.edge_embedding4(edge_attr[:,3])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class AFF(nn.Module):
    """
    Attentional Feature Fusion
    """
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def mean_pooling(self, last_hidden_state, attention_mask):
        if last_hidden_state.shape[-2] != attention_mask.shape[-1]:
            last_hidden_state = last_hidden_state.transpose(-1, -2)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return torch.unsqueeze(mean_embeddings, -1)

    def forward(self, x, residual, ngram_words_mask):
        xa = x + residual
        xl = self.local_att(xa)
        AvgPool1d_xa = self.mean_pooling(xa, ngram_words_mask)
        xg = self.global_att(AvgPool1d_xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class GNN(nn.Module):
    """
    

    Args:
        gnn_num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, args):
        super(GNN, self).__init__()
        self.gnn_num_layer = args.gnn_num_layer
        self.drop_ratio = args.drop_ratio
        self.emb_dim = args.emb_dim

        if self.gnn_num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_degree, self.emb_dim)
        self.x_embedding3 = nn.Embedding(num_formal_charge, self.emb_dim)
        self.x_embedding4 = nn.Embedding(num_chirality, self.emb_dim)
        self.x_embedding5 = nn.Embedding(num_hybridization, self.emb_dim)
        self.x_embedding6 = nn.Embedding(num_numH, self.emb_dim)
        self.x_embedding7 = nn.Embedding(num_implicit_valence, self.emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        nn.init.xavier_uniform_(self.x_embedding7.weight.data)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.gnn_num_layer):
            self.gnns.append(GINConv(self.emb_dim, aggr = "add"))
            ###List of batchnorms
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))
            

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):

        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + \
            self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) + \
            self.x_embedding5(x[:,4]) + self.x_embedding6(x[:,5]) + \
            self.x_embedding7(x[:,6])
         
        
        
        h_list = [x]
        
        for layer in range(self.gnn_num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.gnn_num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.elu(h), self.drop_ratio, training = self.training)  #relu->elu
            
            h_list.append(h)

        ### Different implementations of Jk-concat
        node_representation = h_list[-1]

        return node_representation
    
class ngram_CNN(nn.Module):

    def __init__(self, args):
        super(ngram_CNN, self).__init__()
        self.drop_ratio = args.drop_ratio
        self.words_dict_num = args.words_dict_num
        self.emb_dim = args.emb_dim
        
        self.words_embedding = nn.Embedding(self.words_dict_num, self.emb_dim)
        nn.init.xavier_uniform_(self.words_embedding.weight.data)
        self.cnns = nn.ModuleList(
            [nn.Conv1d(self.emb_dim, self.emb_dim//2, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(self.emb_dim//2, self.emb_dim//4, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(self.emb_dim//4, self.emb_dim//8, kernel_size=15, stride=1, padding=7)]
        )
        self.batch_norms = nn.ModuleList(            
            [nn.BatchNorm1d(self.emb_dim//2),
            nn.BatchNorm1d(self.emb_dim//4),
            nn.BatchNorm1d(self.emb_dim//8)]
            )
        self.trans_convs = nn.ModuleList([
            nn.ConvTranspose1d(self.emb_dim//8, self.emb_dim//4, kernel_size=15, stride=1, padding=7), 
            nn.ConvTranspose1d(self.emb_dim//4, self.emb_dim//2, kernel_size=15, stride=1, padding=7)
        ])
        self.aff_1 = AFF(self.emb_dim//4, 4)
        self.aff_2 = AFF(self.emb_dim//2, 8)
        self.linear_1 =  nn.Linear(self.emb_dim//2, self.emb_dim)
        self.cnn_num_layer = len(self.cnns)

    def forward(self, words, ngram_words_mask):
        hidden = self.words_embedding(words) # [batch, words_length, emb_dim]
        hidden = hidden.permute(0, 2, 1) # [batch, emb_dim, words_length]
        h_list = []
        for layer in range(self.cnn_num_layer):
#             hidden = F.elu(self.cnns[layer](hidden))
            # last_hidden = hidden
            hidden = self.cnns[layer](hidden)
            hidden = self.batch_norms[layer](hidden)
            if layer != self.cnn_num_layer - 1:
                hidden = F.relu(hidden)
            # hidden =  last_hidden + hidden
            h_list.append(hidden)
        h_last = self.trans_convs[0](h_list[-1])
        aff_result = self.aff_1(h_last, h_list[-2], ngram_words_mask)
        aff_trans_convs = self.trans_convs[1](aff_result)
        final_aff_result = self.aff_2(aff_trans_convs, h_list[-3], ngram_words_mask)
        protein_representation = final_aff_result.permute(0, 2, 1)
        
        protein_representation = self.linear_1(protein_representation)
        
        return protein_representation
    

class GNN_graphpred(nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        gnn_type: gin
        
    """
    def __init__(self, args):
        super(GNN_graphpred, self).__init__()
        self.gnn_num_layer = args.gnn_num_layer
        self.drop_ratio = args.drop_ratio
        self.emb_dim = args.emb_dim
        self.num_tasks = args.num_tasks
        self.num_head = 8



        if self.gnn_num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(args)
        self.ngram_cnn = ngram_CNN(args)
        self.attention = nn.Linear(self.emb_dim, self.emb_dim)
        self.pred_linear = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ELU(),
            nn.Linear(self.emb_dim, self.num_tasks)
            )
        

    def super_node_rep(self, node_rep, ptr):
        
        super_group = []
        for i in ptr:
            if i != 0:
                super_group.append(node_rep[i-1])
        super_rep = torch.stack(super_group, dim=0)
        return super_rep
    
    def split_diffrent_layer_node(self, node_rep, num_part, num_graph):
        count = 0
        atom_group = []
        motif_group = []
        super_group = []
        atom_length = []
        motif_length = []
        num_part = num_part.view(-1, 3)
        for i in range(num_graph):
            atom_part = num_part[i][0].item()
            atom_group.append(node_rep[count: count + atom_part])
            atom_length.append(torch.tensor([1]*atom_part))
            count += atom_part
            motif_part = num_part[i][1].item()
            motif_group.append(node_rep[count: count + motif_part])
            motif_length.append(torch.tensor([1]*motif_part))
            count += motif_part
            super_group.append(node_rep[count])
            count += 1
        atom_rep = nn.utils.rnn.pad_sequence(atom_group, batch_first=True, padding_value=0.0)
        atom_mask = nn.utils.rnn.pad_sequence(atom_length, batch_first=True, padding_value=0.0).to(node_rep.device)
        motif_rep = nn.utils.rnn.pad_sequence(motif_group, batch_first=True, padding_value=0.0)
        motif_mask = nn.utils.rnn.pad_sequence(motif_length, batch_first=True, padding_value=0.0).to(node_rep.device)
        super_rep = torch.stack(super_group, dim=0)
        return atom_rep, atom_mask, motif_rep, motif_mask, super_rep
    
    def attention_protein(self, node_rep, protein_rep):
        node_rep_h =  torch.relu(self.attention(node_rep))
        protein_rep_h = torch.relu(self.attention(protein_rep))
        protein_rep_h =  protein_rep_h.transpose(1, 2)
        weights = torch.tanh(torch.bmm(node_rep_h, protein_rep_h))
        return weights

    def mean_pooling(self, last_hidden_state, attention_mask):
        if last_hidden_state.shape[-2] != attention_mask.shape[-1]:
            last_hidden_state = last_hidden_state.transpose(-1, -2)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def forward(self, *argv):
        
        if len(argv) == 3:
            data= argv[0]
            ngram_words, ngram_words_mask = argv[1], argv[2]
            x, edge_index, edge_attr, num_part, ptr = data.x, data.edge_index, data.edge_attr, data.num_part, data.ptr
        else:
            raise ValueError("unmatched number of arguments.")
        
        node_representation = self.gnn(x, edge_index, edge_attr)
        ngram_rep = self.ngram_cnn(ngram_words, ngram_words_mask)
        atom_rep, atom_mask, motif_rep, motif_mask, super_rep =  self.split_diffrent_layer_node(node_representation, num_part, len(ptr)-1)
        super_rep = torch.unsqueeze(super_rep, 1)

        atom_ngram_weights = self.attention_protein(atom_rep, ngram_rep) # [batch, atom_num, seq_length]
        motif_ngram_weights = self.attention_protein(motif_rep, ngram_rep) # [batch, motif_num, seq_length]
        super_ngram_weights = self.attention_protein(super_rep, ngram_rep) # [batch, 1, seq_length]
        
        mean_atom_weights = torch.softmax(self.mean_pooling(atom_ngram_weights, atom_mask), 1) #[batch, seq_len]
        mean_motif_weights = torch.softmax(self.mean_pooling(motif_ngram_weights, motif_mask), 1) #[batch, seq_len]
        ngram_weights = torch.softmax(super_ngram_weights[:,0,:], 1) + mean_atom_weights + mean_motif_weights
        
        ngram_weights = torch.unsqueeze(ngram_weights, 1)

        ngram_rep = torch.bmm(ngram_weights, ngram_rep)
        ngram_rep  = torch.squeeze(ngram_rep, 1)
        super_rep = torch.squeeze(super_rep, 1)

        concat_rep = torch.cat((super_rep, ngram_rep), 1)
    
        return self.pred_linear(concat_rep)
    
    def __call__(self, data, ngram_words, ngram_words_mask, labels, train=True):

        predicted_interaction = self.forward(data, ngram_words, ngram_words_mask)
        correct_interaction = labels

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

    

if __name__=='__main__':
    pass