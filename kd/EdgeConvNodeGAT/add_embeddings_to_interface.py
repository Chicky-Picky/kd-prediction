import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import basename

path_to_data = 'data_dG/data/' # please, make sure it matches dataset_embedding section in config.json
path_to_new_data = 'data_kd/data/'

for emb_file in tqdm(glob("embeddings/*.csv")):
    df = pd.DataFrame(columns=['x', 'y', 'z', 'embedding'])
    test_df = pd.read_csv(path_to_data + basename(emb_file))
    embeddings = np.loadtxt(emb_file, delimiter=',')

    for index, row in test_df.iterrows():
        emb = embeddings[index]
        df.loc[-1] = [row['x'], row['y'], row['z'], " ".join([str(num) for num in emb])]
        df.index = df.index + 1

    df.to_csv(path_to_new_data + basename(emb_file), index=False)
