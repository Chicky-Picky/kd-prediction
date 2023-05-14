from tqdm  import tqdm
from torch_geometric.loader import DataLoader
from utils.setup import SetupRun
from utils.util import read_json
from dataset import graph_dataset as module_dataset
import utils.transform as module_transform
from model import model as module_arch

from torchvision import transforms
import torch
import torch.nn.functional as F
import os

import numpy as np


def embeddings_extraction(run_setup: SetupRun,
                          path_to_the_best_checkpoint: str,
                          device: str):
    dataset_transforms = [run_setup.init_obj(transform, module_transform) for transform in
                          run_setup['dataset_transforms']]
    dataset = run_setup.init_obj('dataset_embeddings', module_dataset, transform=transforms.Compose(dataset_transforms))
    mol_csvs = dataset.raw_file_names

    test_loader = DataLoader(dataset=dataset, batch_size=1)

    model = run_setup.init_obj('arch', module_arch)
    model.load_state_dict(torch.load(path_to_the_best_checkpoint, map_location=torch.device('cpu'))['state_dict'])
    model.eval()

    path = '../kd/EdgeConvNodeGAT/embeddings/'
    os.makedirs(path, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):

            data, target = data.to(device), target.to(device)
            molname = mol_csvs[batch_idx].split(".")[0]
            
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            layers = zip(model.convs, model.batchnorms) if model.batchnorms else model.convs

            for i, layer in enumerate(layers):
                if model.batchnorms:
                    conv, batchnorm = layer
                else:
                    conv = layer

                h, edge_attr, _ = conv(x, edge_index, edge_attr=edge_attr)
                h = F.relu(batchnorm(h)) if model.batchnorms else F.relu(h)

                if model.linear_layer_nodes:
                    if i < model.num_layers - 1:
                        h = F.relu(model.linear_layer_nodes[i](h))

                if model.linear_layer_edges:
                    if i < model.num_layers - 1:
                        edge_attr = F.relu(model.linear_layer_edges[i](edge_attr))

                x = h + x if model.residuals_nodes else h

            np.savetxt(path + molname + '.pdb.csv', x.detach().cpu().numpy(), delimiter=',', fmt='%f')


if __name__ == '__main__':
    device = "cpu"
    path_to_experiment = "../gbsa/experiments/GbsaDgGraphDataset_Net/checkpoint/EdgeConvNodeGATModel_example"

    config = read_json(os.path.join(path_to_experiment, "config.json"))

    run_setup = SetupRun(config=config, checkpoint_dir=path_to_experiment)

    embeddings_extraction(run_setup=run_setup,
                          path_to_the_best_checkpoint=os.path.join(path_to_experiment, "model_best.pth"),
                          device=device)
