import dgl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv, GCN2Conv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

class GCN_Encoder2(torch.nn.Module):

    def __init__(self, param):
        super(GCN_Encoder2, self).__init__()

        self.param = param
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.gnnlayers = nn.ModuleList()

        self.fc_dim = nn.Linear(param['resid_hidden_dim'], param['prot_hidden_dim'])
        self.norm_in = nn.BatchNorm1d(param['prot_hidden_dim'])
        for i in range(self.num_layers):
            self.gnnlayers.append(GCN2Conv(param['prot_hidden_dim'], layer=i+1, allow_zero_in_degree=True))

        self.fc = nn.Linear(param['prot_hidden_dim'], 2*param['prot_hidden_dim'])
        self.norm = nn.BatchNorm1d(2*param['prot_hidden_dim'])
        self.fc2 = nn.Linear(2*param['prot_hidden_dim'], param['prot_hidden_dim'])

        self.Decoder2 = GCN_Decoder2(param)
        self.fc_out = nn.Linear(param['prot_hidden_dim'], param['output_dim'])

    def forward(self, g, x, ppi_list, idx):
        x = self.norm_in(self.fc_dim(x))

        mask_x = x.clone()

        mask_index = torch.bernoulli(torch.ones_like(mask_x, dtype=torch.bool),
                                     1 - self.param['mask2_ratio'])
        mask_x = mask_x * mask_index
   
        x_enc = self.encoding(g, x)
        mask_x_enc = self.encoding(g, mask_x)

        recon2_x = self.Decoder2.decoding(g, x_enc)
        recon2_loss = F.mse_loss(recon2_x, x)

        node_id = np.array(ppi_list)[idx]
        x1 = x_enc[node_id[:, 0]]
        x2 = x_enc[node_id[:, 1]]
        mask_x1 = mask_x_enc[node_id[:, 0]]
        mask_x2 = mask_x_enc[node_id[:, 1]]

        x = self.fc_out(torch.mul(x1, x2))

        mask_x = self.fc_out(torch.mul(mask_x1, mask_x2))

        return x, mask_x, recon2_loss
        
    def encoding(self, g, x):

        x_in = x
        for l, layer in enumerate(self.gnnlayers):

            x = layer(g, x, x_in)

        x = self.dropout(self.norm(F.relu(self.fc(x))))
        x = self.dropout(F.relu(self.fc2(x)))

        return x


class GCN_Decoder2(torch.nn.Module):

    def __init__(self, param):
        super(GCN_Decoder2, self).__init__()

        self.param = param
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.gnnlayers = nn.ModuleList()

        for i in range(self.num_layers):
            self.gnnlayers.append(GCN2Conv(param['prot_hidden_dim'], layer=i+1, allow_zero_in_degree=True))
    
        self.fc = nn.Linear(param['prot_hidden_dim'], 2*param['prot_hidden_dim'])
        self.norm = nn.BatchNorm1d(2*param['prot_hidden_dim'])
        self.fc2 = nn.Linear(2*param['prot_hidden_dim'], param['prot_hidden_dim'])

    def decoding(self, g, x):


        x_in = x
        for l, layer in enumerate(self.gnnlayers):
            x = layer(g, x, x_in)

        x = self.dropout(self.norm(F.relu(self.fc(x))))
        x = self.dropout(F.relu(self.fc2(x)))

        return x


class GCN_Encoder1(nn.Module):
    def __init__(self, param, data_loader):
        super(GCN_Encoder1, self).__init__()

        self.data_loader = data_loader
        self.num_layers = param['resid_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.gnnlayers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.fc_dim = nn.Linear(param['input_dim'], param['resid_hidden_dim'])

        for i in range(self.num_layers):

            self.gnnlayers.append(GATConv(param['resid_hidden_dim'], param['resid_hidden_dim'], param['num_heads'], allow_zero_in_degree=True))
            self.fcs.append(nn.Linear(param['resid_hidden_dim'], param['resid_hidden_dim']))
            self.norms.append(nn.BatchNorm1d(param['resid_hidden_dim']))

    def forward(self):

        prot_embed_list = []
        for iter, batch_graph in enumerate(self.data_loader):
            batch_graph.to(device)
            x = batch_graph.ndata['x']
            batch_graph.ndata['h'] = self.encoding(batch_graph, x)

            prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
            prot_embed_list.append(prot_embed)

        return torch.cat(prot_embed_list, dim=0)

    def encoding(self, batch_graph, x):

        x = self.fc_dim(x)
        for l, layer in enumerate(self.gnnlayers):

            x = torch.mean(layer(batch_graph, x), dim=1)
            x = self.norms[l](F.relu(self.fcs[l](x)))
            if l != self.num_layers:
                x = self.dropout(x)

        return x


class GCN_Decoder1(nn.Module):
    def __init__(self, param):
        super(GCN_Decoder1, self).__init__()

        self.num_layers = param['resid_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.gnnlayers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fcs = nn.ModuleList()

        for i in range(self.num_layers):

            self.gnnlayers.append(GATConv(param['resid_hidden_dim'], param['resid_hidden_dim'], param['num_heads'], allow_zero_in_degree=True))
            self.fcs.append(nn.Linear(param['resid_hidden_dim'], param['resid_hidden_dim']))
            self.norms.append(nn.BatchNorm1d(param['resid_hidden_dim']))

        self.fc_dim = nn.Linear(param['resid_hidden_dim'], param['input_dim'])

    def decoding(self, batch_graph, x):

        for l, layer in enumerate(self.gnnlayers):

            x = torch.mean(layer(batch_graph, x), dim=1)
            x = self.norms[l](F.relu(self.fcs[l](x)))
            if l != self.num_layers:
                x = self.dropout(x)

        x = self.fc_dim(x)

        return x


class RecNet1(nn.Module):
    def __init__(self, param, data_loader):
        super(RecNet1, self).__init__()

        self.param = param

        self.Encoder1 = GCN_Encoder1(param, data_loader)
        self.Decoder1 = GCN_Decoder1(param)

    def forward(self, batch_graph):
        x = batch_graph.ndata['x']

        z = self.Encoder1.encoding(batch_graph, x)
        recon1_x = self.Decoder1.decoding(batch_graph, z) 

        recon1_loss = F.mse_loss(recon1_x, x)

        return z, recon1_loss
