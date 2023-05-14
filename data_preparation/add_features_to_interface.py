from pyxmolpp2 import PdbFile
from glob import glob
import json
import pandas as pd
import os


if __name__ == '__main__':
	from tqdm import tqdm
	
	with open("aname_to_atype.json") as fin:
		aname_to_atype = json.load(fin)

	with open("atype_to_one_hot.json") as fin:
		atype_to_one_hot = json.load(fin)

	out_dir = "interface_with_features"
	os.makedirs(out_dir, exist_ok=True)

	interface_pdbs = glob("../interface_pdb/*pdb")
	interface_pdbs.sort()

	for interface_pdb in tqdm(interface_pdbs):
		interface = PdbFile(interface_pdb).frames()[0]
		one_hot_encoding_atype = []


		for residue in interface.residues:
			for atom in residue.atoms:
				try:
					atype = aname_to_atype[residue.name][atom.name]
					one_hot_encoded_aname = atype_to_one_hot[atype]
					one_hot_encoding_atype.append("".join(list(map(str, one_hot_encoded_aname[0]))))
				except:
					print(atom.name, residue.name,  interface_pdb)


		coords = interface.atoms.coords.values

		interface_df = pd.DataFrame({"x": coords[:, 0], 
									 "y": coords[:, 1],
									 "z": coords[:, 2],
									 "one-hot-atype": one_hot_encoding_atype})

		os.makedirs(os.path.join("one-hot-atype", out_dir), exist_ok=True)
		interface_df.to_csv(os.path.join("one-hot-atype", out_dir, f"{os.path.basename(interface_pdb)}.csv"), index=False)
