import torch
import copy
import torch.nn as nn
import os

from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, \
    _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

import numpy as np
import math
import pandas as pd
from pyxmolpp2 import PdbFile
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, mpnn):
        super().__init__()
        self.mpnn = mpnn

    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        for layer in self.mpnn.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V


temperatures = [0.1]
omit_AAs_list = "X"
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
alphabet_dict = dict(zip(alphabet, range(21)))
chain_id_dict = None
fixed_positions_dict = None
pssm_dict = None
omit_AA_dict = None
bias_AA_dict = None
tied_positions_dict = None
bias_by_res_dict = None
bias_AAs_np = np.zeros(len(alphabet))
omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

dataset = StructureDataset("parsed_pdbs.jsonl", truncate=None, max_length=200000, verbose=False)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
checkpoint = torch.load("ca_model_weights/v_48_002.pt", map_location=device)
model = ProteinMPNN(ca_only=True, num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                    num_encoder_layers=3, num_decoder_layers=3, augment_eps=0, k_neighbors=checkpoint['num_edges'])
model.load_state_dict(checkpoint['model_state_dict'])

encoder_model = Encoder(model)
encoder_model.eval()
encoder_model.to(device)

embs_path = 'embeddings/'
os.makedirs(embs_path, exist_ok=True)

for ix, protein in enumerate(tqdm(dataset)):
    batch_clones = [copy.deepcopy(protein)]
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict,
        bias_by_res_dict, ca_only=True)

    randn_2 = torch.randn(chain_M.shape, device=X.device)

    E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = model.W_e(E)

    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend

    x = encoder_model(h_V, h_E, E_idx, mask, mask_attend).detach().cpu().numpy()[0]

    df = pd.DataFrame(columns=['x', 'y', 'z', 'embedding'])

    cnt = 0
    coords_names = [item for item in protein.keys() if 'coords' in item]
    for c_name in coords_names:
        CA_field = list(protein[c_name].keys())[0]
        for coords in protein[c_name][CA_field]:
            if not math.isnan(coords[0][0]):
                df.loc[-1] = [coords[0][0], coords[0][1], coords[0][2], " ".join([str(num) for num in x[cnt]])]
                df.index = df.index + 1
            cnt += 1

    df.to_csv(embs_path + protein['name'] + '.pdb.csv')

kd_data_path = "data_kd/data/"
os.makedirs(kd_data_path, exist_ok=True)

for ix, protein in enumerate(tqdm(dataset)):
    interface = PdbFile("interface_pdb/" + protein['name'] + ".pdb").frames()[0]
    interface_coords = [[atom.r.x, atom.r.y, atom.r.z] for atom in interface.molecules.atoms]

    path_to_embedding_csv = f"{embs_path}/{protein['name']}.pdb.csv"
    protein_df = pd.read_csv(path_to_embedding_csv)
    interface_df = pd.DataFrame(columns=['x', 'y', 'z', 'embedding'])

    for i, row in protein_df.iterrows():
        if [row['x'], row['y'], row['z']] in interface_coords:
            interface_df.loc[-1] = [row['x'], row['y'], row['z'], row['embedding']]
            interface_df.index = interface_df.index + 1

    interface_df.to_csv(os.path.join(kd_data_path, f"{protein['name']}.pdb.csv"), index=False)
