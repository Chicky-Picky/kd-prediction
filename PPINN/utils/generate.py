from glob import glob
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import os


def generate_random_chargers(n_points: int,
                             min_value: int = 0,
                             max_value: int = 1,
                             ) -> np.ndarray:
    """
    This function generates point charges uniformly distributed in range from [min_value; max_value].
    :param n_points: number of point charges
    :param min_value: minimal value of point charge
    :param max_value: maximal value of point charge
    :return: numpy array of point charges
    """
    return np.random.random(n_points) * (max_value - min_value) + min_value


def generate_vector(point,
                    bond_length
                    ):
    """
    This function generates a point on the sphere of current radius with center in a given point.
    :param point: center of the sphere in numpy array format
    :param bond_length: radius of the sphere
    :return: point in numpy array format
    """
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    vec *= bond_length
    return point + vec


def check_tooclose(visited_points,
                   newpos,
                   avoidradius,
                   ):
    """
    This function checkes if the new point is far enough from points in array visited.
    :param visited_points: a set of points in numpy array format
    :param newpos: new point in numpy array format
    :param avoidradius: minimal allowed distance between points
    :return: bool True or False
    """
    dist = np.linalg.norm(visited_points - newpos)
    isclose = dist <= avoidradius
    return np.count_nonzero(isclose) > 0


def generate_worm_like_chain_3d(n_points,
                                bond_length,
                                ) -> np.ndarray:
    """
    This function generates worm-like chain in 3D-space, that imitates a polymer chain.
    The idea is the same as for self-avoiding walk algorithm https://compphys.quantumtinkerer.tudelft.nl/proj2-polymers/
    :param n_points: number of points defining the length of worm-like chain
    :param bond_length: length of bond connecting two points in worm-like chain
    :return: numpy array of cartesian coordinates. For example for two-point chain [[x1, y1, z1], [z2, y2, z2]]
    """
    visited_points = np.zeros((n_points, 3))
    visited_points[1] = generate_vector(visited_points[0], bond_length)

    for i in range(2, n_points):
        current_point = generate_vector(visited_points[i - 1], bond_length)

        while check_tooclose(visited_points[:i - 1], current_point, bond_length):
            current_point = generate_vector(visited_points[i - 1], bond_length)

        visited_points[i] = current_point

    return visited_points


def save_charged_pairs(path_to_out_csv: str,
                       size: int = 1000,
                       min_value: int = 0,
                       max_value: int = 1,
                       bond_length: float = 1,
                       ) -> None:
    """
    This function saves csv-file with pairs of point charges and target potential.
    :param path_to_out_csv: path to csv-file to be saved
    :param size: size of dataset, i.e. number of pairs of point charges
    :param min_value: minimal value of point charge
    :param max_value: maximal value of point charge
    :param bond_length: length of bond connecting two point charges
    :return: None
    """
    q = generate_random_chargers(size * 2, min_value, max_value)
    q1 = np.array(q[:size])
    q2 = np.array(q[size:])
    dist = np.ones(size) * bond_length
    t = q1 * q2 / dist

    df = pd.DataFrame({'q1': q1, 'q2': q2, 'distance': dist, 'target': t})
    df.to_csv(path_to_out_csv, sep='\t')


def save_charged_chains(size: int = 1000,
                        chain_len: int = 10,
                        min_value: int = -1,
                        max_value: int = 1,
                        bond_length: float = 1,
                        path_to_outdir: str = ".",
                        outfile_format="mol_%05d.csv"
                        ) -> None:
    """
    This function saves csv-files with charge chains.
    :param size: size of dataset, i.e. number of charger chains
    :param chain_len: length of chains in dataset, i.e. number of chargers in chain
    :param min_value: minimal value of point charge
    :param max_value: maximal value of point charge
    :param bond_length: length of bond connecting two point charges
    :param path_to_outdir: path to output directory
    :param outfile_format: format of out csv file
    :return: None
    """
    charges = generate_random_chargers(size * chain_len, min_value, max_value)
    os.makedirs(path_to_outdir, exist_ok=True)

    for ind in range(size):
        charge = np.array(charges[ind * chain_len:(ind + 1) * chain_len])
        chain = generate_worm_like_chain_3d(chain_len, bond_length)
        mol_df = pd.DataFrame(np.concatenate([chain, charge[:, np.newaxis]], axis=1), columns=["x", "y", "z", "q"])
        mol_df.to_csv(os.path.join(path_to_outdir, outfile_format % (ind + 1)), index=False)


def calc_el_energy(points,
                   charges,
                   ) -> float:
    """
    This function calculates target potential.
    :param points: Nx3 coordinate matrix of point charges in space like [[x1, y1, z1], [x2, y2, z2]]
    :param charges: Nx1 array of charges
    :return: float
    """
    charge_matrix = charges @ charges.T
    pairwise_dist = cdist(points, points)
    pairwise_dist[pairwise_dist == 0] = -1
    return np.where(pairwise_dist > 0, charge_matrix / pairwise_dist, 0).sum() / 2


def save_potential(path_to_mol_dir,
                   path_to_outdir=".",
                   outname="el_energy.csv"):
    """
    This function saves csv-file with electrostatic potential for list of molecules.
    :param path_to_mol_dir: directory with csv files of molecules coords and charges
    :param path_to_outdir: path to output directory
    :param outname: output file name
    :return:
    """
    os.makedirs(path_to_outdir, exist_ok=True)
    mol_csvs = glob(os.path.join(path_to_mol_dir, "*.csv"))
    mol_csvs.sort()
    el_energy = []
    molnames = []
    for mol_csv in mol_csvs:
        mol_df = pd.read_csv(mol_csv)
        el_energy.append(calc_el_energy(points=mol_df[["x", "y", "z"]].values,
                                        charges=mol_df[["q"]].values))
        molnames.append(os.path.basename(mol_csv).split(".")[0])
    return pd.DataFrame({"molname": molnames, "el_energy": el_energy}).to_csv(os.path.join(path_to_outdir, outname),
                                                                              index=False)


if __name__ == "__main__":
    save_charged_chains(path_to_outdir="../electrostatic/data/mols", bond_length=1.53, size=10, chain_len=2)
    save_potential(path_to_mol_dir="../electrostatic/data/mols", path_to_outdir="data")
