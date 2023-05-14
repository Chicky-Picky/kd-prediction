from pyxmolpp2 import Molecule, MoleculeSelection, ResidueSelection, PdbFile, mName, rName, aId
from scipy.spatial import cKDTree
from typing import List, Union
from glob import glob
import os


def extract_residues_on_interface(partner_A: Union[Molecule, MoleculeSelection],
                                  partner_B: Union[Molecule, MoleculeSelection],
                                  cutoff: float
                                  ) -> List[ResidueSelection]:
    """
    Extract residues on interface of complex between partner_A and partner_B
    :param partner_A: first partner of intermolecular interaction
    :param partner_B: second partner of intermolecular interaction
    :param cutoff: distance cutoff in angstroms
    :return: list of [residues of partner_A, residues of partner_B] - residues on interface
                  of interaction between partner_A and partner_B within the distance cutoff
    """
    partner_A_ats = partner_A.atoms
    partner_B_ats = partner_B.atoms

    tree_A = cKDTree(partner_A_ats.coords.values)
    tree_B = cKDTree(partner_B_ats.coords.values)

    dist = tree_A.sparse_distance_matrix(tree_B, max_distance=cutoff, output_type='coo_matrix')

    interface_A_ids = [partner_A_ats[k].id for k in dist.row]
    interface_B_ids = [partner_B_ats[k].id for k in dist.col]

    interface_A_residues = partner_A_ats.filter(aId.is_in(set(interface_A_ids))).residues
    interface_B_residues = partner_B_ats.filter(aId.is_in(set(interface_B_ids))).residues

    return [interface_A_residues, interface_B_residues]


if __name__ == "__main__":
    import pandas as pd
    from pyxmolpp2 import aName
    from tqdm import tqdm


    pdb_files = glob(os.path.join("clean_pdb", "*.pdb"))
    out_dir = "interface_pdb"
    os.makedirs(out_dir, exist_ok=True)

    for pdb_file in tqdm(pdb_files):
      complex_structure = PdbFile(pdb_file).frames()[0]

      mol_A = complex_structure.molecules[0]
      mol_B = complex_structure.molecules[1]

      interface_A, interface_B = extract_residues_on_interface(mol_A, mol_B, 5)

      non_hydrogen_names = set([atom.name for atom in complex_structure.atoms if not atom.name.startswith("H")])

      with open(os.path.join(out_dir, f"{os.path.basename(pdb_file)}"), "w") as fout:
        interface_A.atoms.filter(aName.is_in(non_hydrogen_names)).to_pdb(fout)
        fout.write("TER\n")
        interface_B.atoms.filter(aName.is_in(non_hydrogen_names)).to_pdb(fout)