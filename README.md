# Prediction of protein-protein binding affinity
_This repository provides results and code for the project "Prediction of protein-protein binding affinity"_

**Authors:**
* Anastasiia Kazovskaia _(Saint Petersburg State University, Saint Petersburg, Russia)_
* Mikhail Polovinkin _(Saint Petersburg State University, Saint Petersburg, Russia)_

**Supervisors:**
* Olga Lebedenko _(BioNMR laboratory, Saint Petersburg State University, Saint Petersburg, Russia)_
* Nikolai Skrynnikov _(BioNMR laboratory, Saint Petersburg State University, Saint Petersburg, Russia)_

- [Background](#sec1) </br>
- [Methods](#sec2) </br>
- [System requirements](#sec3) </br>
- [Repository structure](#sec4) </br>
- [Results](#sec5) </br>
- [Conclusions](#sec6) </br>
- [Literature](#sec7) </br>

<a name="sec1"></a>
### Background:
The binding constant is a value characterizing the complex sustainability, which is significantly important in drug design and mutation analysis. It is strongly related to the difference of free energies in the system before and after complex formation. $\Delta G = RT ln K_d$, where $\Delta G$ is a difference of free energies before and after binding, R - molar gas constant, T - tempreture in system in Kelvin units, $K_d$ - dissociation constant. The free energy contains a number of contributions: electrostatic energy, energy of Van-der-Waals interactions, solvation entropy energy and others. Despite the fact that There are different approaches to evaluation of free energy difference in systems based on heuristics a big amount of parameters in free energy estimation makes solution of this problem almost impossible. Hence, $K_d$ is usually measured experimentally, some techniques are developed by now: (TECHNIQUES). The experimental methods are quite expensive and are not productive enough, so some approaches of $K_d$ prediction are still still desirable for development. The development of computer science made it possible to employ the state of art methods to predict the free energy and consequently the dissociation constant. The case of interaction between small-molecule ligand and protein seems to be studied better, meanwhile protein–protein interactions are predicted badly. In this study we intend to examine the problem of the binding affinity prediction for dimer complexes. 

**Goal:** The main goal of this project is to develop a neural network model predicting protein-protein interactions $K_d$.

**Objectives:**
1. Develop train-test pipeline
2. Predict electrostatic energy
3. Predict MM/GBSA free energy
4. Predict $K_d$

<a name="sec2"></a>
### Methods: 
The workflow consists of  steps: pic

1. **Electrostatic energy prediction.** As a rough approximation of our problem, we considered a well-defined one: prediction of electrostatic energy for chain-like molecules. This problem has a direct solution, the Coulomb law, and it is suitable and convenient for model and pipeline development and adjustment. The dataset consisted of a number of chain-like molecules with electrostatic energy calculated for them. The self-avoiding random walk algorithm [[1]](selfavoid1) was implemented in order to obtain srutctures of chain-like molecules. After chain-like molecules generation we assigned charges from uniform distribution $U[-1;1]$ to all point-atoms in molecules. Targets were calculated for all chain-like molecules according to the Coulomb law. We implemented, trained and tested two types of neural networks to predict electrostatic energy: one consisting of several fully-connected layers (FC model) considered as a baseline and graph attention neural network with both vertex and edge convolutions (GAT). The fully-connected model operated with the set of atom features: coordinates $x_i, y_i, z_i$ and charges $q_i$, $i$ lists all atoms. The augmentation was applied to the dataset in order to improve results: all the molecules were rotated 4 times around the random axis by the random angle. This trick helped the FC model learn the rotational invariance. 15000 objects (3000 unique molecules) were in dataset for FC training. In the case of the GAT model the input data consisted of graphs. For all generated chain molecules the fully connected graphs were created, the nodes corresponded to the point-atoms, the charge on the atom was the only node feature. The edge feature corresponded to the distance between the atoms connected with current edge. There were 5000 graphs in train sample in case of GAT model.  
2. **GBSA free energy prediction.** As a more complex and consistent problem, we considered prediction of free energy of protein-protein complexes estimated with MM/GBSA method [[2]](GBSA1),  [[3]](GBSA2).
The PDB ids of dimer complexes (both Hetero and homo dimers)  were obtained using the [Dockground database](https://dockground.compbio.ku.edu/bound/index.php). In order to avoid redundancy in data we clustered complexes with respect to sequence. To solve the clusterization problem we used the [RCSB database]() of sequence clusters with 30% similarity threshold. Both sequences in complex were matched to their clusters with respect to RCSB database resulting in a pair of cluster numbers for complex, this unordered pair is thought to be a complex cluster label. Free energy (dG) was estimated with the GBSA method for all of the structures employing the Amber package. The calculation of the dG for GBSA dataset had the following pipeline. The raw PDBs were processed via biobb [[4]](biobb) to fix the heavy atom absence and complex geometries were minimized employing the phenix package [[5]](phenix). Original hydrogens obtained from solving the structure were removed using Amber package [[6]](amber). We used the PDB2PQR tool [[7]](PDB2PQR) to protonate complexes in target acidity (pH = 7.2). Finally, dG for complexes were calculated within the MM/GBSA method in the Amber. The obtained dataset was then splitted into train, validation and test samples with respect to clasterization, this guaranteed homogeneity of the data and absence of outliers in the samples. The dataset consisted of 10725 complexes in general: 5 668 in complexes train, 3 188 in validation and 1 869 in test. Protein-protein interaction interfaces were extracted from PDBs of complexes, residue of one interacting chain is considered to be on the interface if distance between any of its atoms and any atom of other interaction chain is less than cutoff parameter (cutoff = 5A). Large interfaces containing more than 1000 atoms were dropped from datasets due to their high memory usage in training. Interfaces were processed into graphs for graph neural network. One-hot encoding technique with the respect to Amber atom type was applied to construct features for graph nodes, pairwise distances between atoms were used as edge features. We implemented, trained and tested one more type of neural networks to predict GBSA free energy: GAT model with both vertex and edge attention mechanisms.

3. **$K_d$ prediction.** Finally, to predict $K_d$ we used FC and GAT models accepting as input features either embeddings extracted from GAT model trained on GBSA dataset, or from ProteinMPNN's encoder. We decided to predict $ln(K_d)$ to avoid issues with metrics and make the problem a little closer to the previous one. The PDBbind database [[8]](PDBbind) was chosen as a core of $K_d$ prediction dataset. PDBbind contains a complex Ids, resolutions of structure solving, experimentally measured $K_d$ values and other useful data. We filtered out the poorly solved structures in order to reduce the noise level in data. Structures with resolution lower than 5A were dropped from data. PDBs were processed to extract interfaces (the same cutoff parameter) and construct graphs. The resulting dataset contained about 1300 objects (pairs of interface graph and $\ln(K_d)$) in total. Train, validation and test datasets were obtained by random splitting with relative sizes 0.7, 0.2 and 0.1 respectively.

<a name="sec3"></a>
### System requirements:
**Key packages and programs:**
- the majority of scripts are written on `Python3` and there are also `bash` scripts
- `slurm` (20.11.8) cluster management and job scheduling system
- `Amber20` (a build with MPI is employed and is highly preferrable)
- [amber-runner](https://github.com/sizmailov/amber-runner) (0.0.8)
- [pyxmolpp2](https://github.com/sizmailov/pyxmolpp2) (1.6.0, note that this library works best under Linux)
- other python libraries used for plotting and analysis are listed in requirements.txt

<a name="sec4"></a>
### Repository structure:  

<a name="sec5"></a>
### Results:
This section analyzes the performance of the proposed method to predict electrostatic energy, MM/GBSA free energy and $K_d$. The Loss and MSD metrics are used to evaluate precizion of predictions.

In case of electrostatincs energy prediction the molecules of length 2, 4, 8, 16 and 32 atoms were concidered. As we expected GAT models had a way better performance than FC.
![What is this](electrotrain.png)


<a name="sec6"></a>
### Conclusions:

<a name="sec7"></a>
### Literature
<a id="selfavoid1">[1]</a>
Rosenbluth M. and Rosenbluth A. Monte Carlo Calculation of the Average Extension of Molecular Chains. 1995. J. Chem. Phys. 23(2): 356–359. 
https://doi.org/10.1063/1.1741967

<a id="GBSA1">[2]</a>
Kollman, P. A., Massova, I., Reyes, C. et al. Calculating Structures and Free Energies of Complex Molecules:  Combining Molecular Mechanics and Continuum Models. 2000. Accounts of Chemical Research. 33(12): 889-897.
https://doi.org/10.1021/ar000033j

<a id="GBSA2">[3]</a>
Genheden, S. and Ryde, U. The MM/PBSA and MM/GBSA methods to estimate ligand-binding affinities. 2015. Expert opinion on drug discovery.  10(5): 1-13
https://doi.org/10.1517/17460441.2015.1032936

<a id="biobb">[4]</a>
Andrio, P., Hospital, A., Conejero, J. et al. BioExcel Building Blocks, a software library for interoperable biomolecular simulation workflows. 2019. Sci Data. 6, 169. https://doi.org/10.1038/s41597-019-0177-4

<a id="phenix">[5]</a>
Liebschner, D., Afonine, P. V., Baker, M. L. et al. 2019. Macromolecular structure determination using x-rays, neutrons and electrons: recent developments in phenix. Acta Crystallogr D Struct Biol. 75, 861–877. https://doi:10.1107/S2059798319011471.

<a id="amber">[6]</a>
D.A. Case, H.M. Aktulga, K. Belfon et al. 2023. Amber 2023, University of California, San Francisco.

<a id="PDB2PQR">[7]</a>
Dolinsky, T. D., Czodrowski, P., Li, H. et al. PDB2PQR: expanding and upgrading automated preparation of biomolecular structures for molecular simulations. 2007. Nucleic Acids Research, Volume 35, Issue suppl_2, 1 July , Pages W522–W525, https://doi.org/10.1093/nar/gkm276

<a id="PDBbind">[7]</a>
Su, M., Yang, Q., Du, Y. et al. Comparative Assessment of Scoring Functions: The CASF-2016 Update. 2019. Journal of chemical information and modeling, 59(2): 895–913. https://doi.org/10.1021/acs.jcim.8b00545
