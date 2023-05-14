from model.base import BaseModel

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList, Sequential, ReLU, Linear

from torch_geometric.nn import GATConv, MetaLayer
from torch_geometric.nn import BatchNorm, global_mean_pool


class WormLikeChainModel(BaseModel):
    def __init__(self, chain_len=10):
        super().__init__()
        self.fc1 = nn.Linear(chain_len * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NodeGATModel(BaseModel):

    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 edge_feature_dim: Optional[int] = 1,
                 num_layers: int = 4,
                 heads: int = 1,
                 concat: bool = False,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 fill_value: Union[float, Tensor, str] = 0,
                 bias: bool = True,
                 batchnorm: bool = True,
                 residuals: bool = True,
                 **kwargs,
                 ):

        super().__init__()

        self.convs = ModuleList()
        self.batchnorms = ModuleList() if batchnorm else None
        self.residuals = residuals

        self.convs.append(
            GATConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    concat=concat,
                    edge_dim=edge_feature_dim,
                    heads=heads,
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=fill_value,
                    bias=bias,
                    **kwargs))

        for _ in range(num_layers - 1):

            conv = GATConv(in_channels=node_embedding_dim,
                           out_channels=node_embedding_dim,
                           concat=concat,
                           edge_dim=edge_feature_dim,
                           heads=heads,
                           negative_slope=negative_slope,
                           dropout=dropout,
                           add_self_loops=add_self_loops,
                           fill_value=fill_value,
                           bias=bias,
                           **kwargs)

            self.convs.append(conv)

            if batchnorm:
                self.batchnorms.append(BatchNorm(node_embedding_dim))

        self.mlp = Sequential(Linear(node_embedding_dim, node_embedding_dim), ReLU(),
                              Linear(node_embedding_dim, node_embedding_dim // 2), ReLU(),
                              Linear(node_embedding_dim // 2, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        layers = zip(self.convs, self.batchnorms) if self.batchnorms else self.convs

        for i, layer in enumerate(layers):
            if self.batchnorms:
                conv, batchnorm = layer
            else:
                conv = layer

            x = conv(x, edge_index, edge_attr)
            h = F.relu(batchnorm(x)) if self.batchnorms else F.relu(x)

            x = h + x if self.residuals else h

        x = global_mean_pool(x, batch)
        return self.mlp(x)


class EdgeConvLayer(nn.Module):

    def __init__(self,
                 node_feature_dim,
                 edge_feature_dim_in,
                 edge_hidden_dim,
                 edge_feature_dim_out,
                 residuals
                 ):
        super().__init__()
        self.residuals = residuals
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_dim + edge_feature_dim_in, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_feature_dim_out),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out


class EdgeConvNodeGATModel(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 edge_feature_dim: int,
                 num_layers: int = 4,
                 heads: int = 1,
                 concat: bool = False,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 fill_value: Union[float, Tensor, str] = 0,
                 bias: bool = True,
                 linear_layer_nodes: bool = True,
                 linear_layer_edges: bool = True,
                 batchnorm: bool = True,
                 residuals_edges: bool = True,
                 residuals_nodes: bool = False,
                 **kwargs,
                 ):

        super().__init__()

        self.num_layers = num_layers
        self.convs = ModuleList()
        self.residuals_edges = residuals_edges
        self.residuals_nodes = residuals_nodes

        self.linear_layer_nodes = ModuleList() if linear_layer_nodes else None
        self.linear_layer_edges = ModuleList() if linear_layer_edges else None
        self.batchnorms = ModuleList() if batchnorm else None

        self.convs.append(MetaLayer(
            EdgeConvLayer(node_feature_dim=node_feature_dim,
                          edge_feature_dim_in=edge_feature_dim,
                          edge_hidden_dim=node_embedding_dim,
                          edge_feature_dim_out=node_embedding_dim,
                          residuals=self.residuals_edges),
            GATConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    concat=concat,
                    edge_dim=node_embedding_dim,
                    heads=heads,
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=fill_value,
                    bias=bias,
                    **kwargs)

        ))

        for _ in range(self.num_layers - 1):

            meta = MetaLayer(
                EdgeConvLayer(node_feature_dim=node_embedding_dim,
                              edge_feature_dim_in=node_embedding_dim,
                              edge_hidden_dim=node_embedding_dim,
                              edge_feature_dim_out=node_embedding_dim,
                              residuals=self.residuals_edges),
                GATConv(in_channels=node_embedding_dim,
                        out_channels=node_embedding_dim,
                        concat=concat,
                        edge_dim=node_embedding_dim,
                        fill_value=0))

            self.convs.append(meta)
            if linear_layer_nodes:
                self.linear_layer_nodes.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if linear_layer_edges:
                self.linear_layer_edges.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if batchnorm:
                self.batchnorms.append(BatchNorm(node_embedding_dim))

        self.mlp = Sequential(Linear(node_embedding_dim, node_embedding_dim), ReLU(),
                              Linear(node_embedding_dim, node_embedding_dim // 2), ReLU(),
                              Linear(node_embedding_dim // 2, 1))

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        layers = zip(self.convs, self.batchnorms) if self.batchnorms else self.convs

        for i, layer in enumerate(layers):
            if self.batchnorms:
                conv, batchnorm = layer
            else:
                conv = layer

            h, edge_attr, _ = conv(x, edge_index, edge_attr=edge_attr)
            h = F.relu(batchnorm(h)) if self.batchnorms else F.relu(h)

            if self.linear_layer_nodes:
                if i < self.num_layers - 1:
                    h = F.relu(self.linear_layer_nodes[i](h))

            if self.linear_layer_edges:
                if i < self.num_layers - 1:
                    edge_attr = F.relu(self.linear_layer_edges[i](edge_attr))

            x = h + x if self.residuals_nodes else h

        x = global_mean_pool(x, batch)
        return self.mlp(x)
