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
The binding constant is a value characterizing the complex sustainability, which is significantly important in drug design and mutation analysis. It is strongly related to the difference of free energies in the system before and after complex formation. $\Delta G = RT ln K_d$, where $\Delta G$ is a difference of free energies before and after binding, R - molar gas constant, T - temperature in system in Kelvin units, $K_d$ - dissociation constant. The free energy contains a number of contributions: electrostatic energy, energy of Van-der-Waals interactions, solvation entropy energy and others. Despite the fact that There are different approaches to evaluation of free energy difference in systems based on heuristics a big amount of parameters in free energy estimation makes solution of this problem almost impossible. Hence, $K_d$ is usually measured experimentally, some techniques are developed by now: (TECHNIQUES). The experimental methods are quite expensive and are not productive enough, so some approaches of $K_d$ prediction are still still desirable for development. The development of computer science made it possible to employ the state of art methods to predict the free energy and consequently the dissociation constant. The case of interaction between small-molecule ligand and protein seems to be studied better, meanwhile protein–protein interactions are predicted badly. In this study we intend to examine the problem of the binding affinity prediction for dimer complexes. 

**Goal:** The main goal of this project is to develop a neural network model predicting protein-protein interactions $K_d$.

**Objectives:**
1. Develop train-test pipeline
2. Predict electrostatic energy
3. Predict MM/GBSA free energy
4. Predict $K_d$

<a name="sec2"></a>
### Methods: 
The workflow consists of steps:
| ![workflow.jpg](???) |
|:--:|
| *Figure 1. Realization steps.* |

1. **Electrostatic energy prediction.** As a rough approximation of our problem, we considered a well-defined one: prediction of electrostatic energy for chain-like molecules. This problem has a direct solution, the Coulomb law, and it is suitable and convenient for model and pipeline development and adjustment. The dataset consisted of a number of chain-like molecules with electrostatic energy calculated for them. The self-avoiding random walk algorithm [[1]](selfavoid1) was implemented in order to obtain structures of chain-like molecules, the distance between two adjacent points was set to 1.53 A (the length of C-C bond). After chain-like molecules generation we assigned charges from uniform distribution $U[-1;1]$ to all point-atoms in molecules. Targets were calculated for all chain-like molecules according to the Coulomb law. We implemented, trained and tested two types of neural networks to predict electrostatic energy: one consisting of several fully-connected layers (FC model) considered as a baseline and graph attention neural network with both vertex and edge convolutions (GAT). The fully-connected model operated with the set of atom features: coordinates $x_i, y_i, z_i$ and charges $q_i$, $i$ lists all atoms. The augmentation was applied to the dataset in order to improve results: all the molecules were rotated 4 times around the random axis by the random angle. This trick helped the FC model learn the rotational invariance. 15000 objects (3000 unique molecules) were in dataset for FC training. In the case of the GAT model the input data consisted of graphs. For all generated chain molecules the fully connected graphs were created, the nodes corresponded to the point-atoms, the charge on the atom was the only node feature. The edge feature corresponded to the distance between the atoms connected with current edge. There were 5000 graphs in train sample.  
2. **GBSA free energy prediction.** As a more complex and consistent problem, we considered prediction of free energy of protein-protein complexes estimated with MM/GBSA method [[2]](GBSA1),  [[3]](GBSA2). The PDB ids of dimer complexes (both Hetero and homo dimers)  were obtained using the [Dockground database](https://dockground.compbio.ku.edu/bound/index.php). In order to avoid redundancy in data we clustered complexes with respect to sequence. To solve the clusterization problem we used the [RCSB database]() of sequence clusters with 30% similarity threshold. Both sequences in complex were matched to their clusters with respect to RCSB database resulting in a pair of cluster numbers for complex, this unordered pair is thought to be a complex cluster label. Free energy (dG) was estimated with the GBSA method for all of the structures employing the Amber package [[4]](amber). The calculation of the dG for GBSA dataset had the following pipeline. The raw PDBs were processed via biobb [[5]](biobb) to fix the heavy atom absence and complex geometries were minimized employing the phenix package [[6]](phenix). Original hydrogens obtained from solving the structure were removed using Amber package. We used the PDB2PQR tool [[7]](PDB2PQR) to protonate complexes in target acidity (pH = 7.2). Finally, dG for complexes were calculated within the MM/GBSA method in the Amber. The obtained dataset was then splitted into train, validation and test samples with respect to clasterization, this guaranteed homogeneity of the data and absence of outliers in the samples. The dataset consisted of 10725 complexes in general: 5 668 in complexes train, 3 188 in validation and 1 869 in test. Protein-protein interaction interfaces were extracted from PDBs of complexes, residue of one interacting chain is considered to be on the interface if distance between any of its atoms and any atom of other interaction chain is less than cutoff parameter (cutoff = 5A). Large interfaces containing more than 1000 atoms were dropped from datasets due to their high memory usage in training. Interfaces were processed into graphs for graph neural network. One-hot encoding technique with the respect to Amber atom type was applied to construct features for graph nodes, pairwise distances between atoms were used as edge features. We implemented, trained and tested one more type of neural networks to predict GBSA free energy: GAT model with both vertex and edge attention mechanisms. The internal embedding size was set to 16.
3. **$K_d$ prediction.** We decided to predict $ln(K_d)$ to avoid issues with metrics and make the problem a little closer to the previous one. In this part of study we used GAT predicting module accepting as input features either embeddings extracted from GAT model trained on GBSA dataset, or from ProteinMPNN's encoder. ProteinMPNN model [[8]](ProteinMPNN) is developed to predict the amino acid sequence by the structure of a protein or protein-protein complex. Therefore, its embeddings potentially contain the relevant information about the complex structure, so can be used to solve our problem. We extracted embeddings from `ca_model_weights/v_48_002.pt` model checkpoint. This model considers $C_\alpha$ atoms only. The embedding dimension is default and equals to 128. As a core of $K_d$ prediction dataset the PDBbind database [[8]](PDBbind) was chosen. PDBbind contains a complex Ids, resolutions of structure solving, experimentally measured $K_d$ values and other useful data. We filtered out the poorly solved structures in order to reduce the noise level in data. Structures with resolution lower than 5A were dropped from data. PDBs were processed to extract interfaces (the same cutoff parameter) and construct graphs. The resulting dataset contained about 1300 objects (pairs of interface graph and $\ln(K_d)$) in total. Train, validation and test datasets were obtained by random splitting with relative sizes 0.7, 0.2 and 0.1 respectively.

<a name="sec3"></a>
### System requirements:
**Key packages and programs:**
- the majority of scripts are written on `Python3` and there are also `bash` scripts
- `slurm` (20.11.8) cluster management and job scheduling system
- [pyxmolpp2](https://github.com/sizmailov/pyxmolpp2) (1.6.0) in-house python library for processing molecular structures and MD trajectories
- [pytorch](https://pytorch.org) (1.13.1+cu11.7) the core of NN workflow
- [pytorch geometric](https://pytorch.org) (2.2.0) the core of graph NN models
- [biobb](https://biobb.readthedocs.io/en/latest/readme.html) software library for biomolecular simulation workflows
- [phenix](https://phenix-online.org) software package for macromolecular structure determination
- [PDB2PQR](https://pdb2pqr.readthedocs.io/en/latest/) a preparing structures for continuum solvation calculations library
- `Amber20` (a build with MPI is employed and is highly preferrable)
- other python libraries used are listed in `requirements.txt`

<a name="sec4"></a>
### Repository structure:  
- [PPINN](/PPINN) contains core project files
- [data_preparation](data_preparation) contains examples for data preparation from raw pdb-files for both GBSA and $`K_d`$ prediction
- [electrostatics](electrostatics) contains examples for electrostatic energy of chain-like molecules prediction
- [figures](figures) contains figures for README.md you are currently reading
- [gbsa](gbsa) contains examples for GBSA energy of real molecules prediction
- [kd](kd) contains examples for $`ln(K_d)`$  prediction


<a name="sec5"></a>
### Results:
This section analyzes the performance of the proposed method to predict electrostatic energy, MM/GBSA free energy and $K_d$. The MSE-loss (Mean Squared Error) is used to train the models, and relative MSE metric is used to evaluate precision of predictions. Relative MSE is defined as follows: $`\frac{1}{\vert \text{batch} \vert}\sum_{\text{item } \in \text{ batch}} \frac{(\text{model}(\text{item}) - \text{target})^2}{\text{target}^2}`$. We also employ correlation coefficient between prediction and target values to estimate the performance.

In case of electrostatics energy prediction the molecules of length 2, 16 and 32 atoms were considered. The datasets were obtained in the way mentioned above. Distributions of electrostatic energies over the training sets are provided in the figure below.
| ![Eel_distribs_data.png](/figures/Eel_distribs_data.png) |
|:--:|
| *Figure 2. Electrostatic energy distributions in the train, validation, and test samples, respectively.* |

The distribution for pairs of charges corresponds to theoretical predictions (the problem of distribution of product of two uniform distributions). Coulomb energy distributions in case of molecules with the bigger number of atoms are much more complex. These have peaks near zero energy and small non zero kurtosis. Cases of 16 and 32 molecules are quite similar, but the bigger molecules have a wider conformation space and as a result more spread energy distribution. In further analysis some similarities in electrostatic energy and dG distributions were observed. These similarities can be explained by the presence of  electrostatic interaction contributions in the free energy of binding.

| ![Electrostatics_train](/figures/electrostatics_train.png)|
|:--:|
| *Figure 3. Electrostatic energy prediction: metrics of the train and validation samples during training.* |

| ![Electrostatics_test](/figures/electrostatics_test.png)|
|:--:|
| *Figure 4. Electrostatic energy prediction: correlation plots on the test samples.* |
On the figures above MSE for train and validation sets dependence on the epoch is presented FC and GAT models. The correlation plots for test datasets are also provided. Both models learned to perfectly predict Coulomb law in case of two atom interactions, the correlation are equal to 1. With an increase of chain-like molecule size the lack of FC model capacity appeared. Let’s emphasize that FC model struggled to learn rotational invariance, without the augmentation procedure the perfomance was even worse. GAT architecture, on the contrary, did not experience such issues. The results of testing on samples of chain-like molecules with 16 and 32 atoms are the following: 0.69 and 0.50 for FC opposed to 0.94 and 0.84 for GAT, respectively. Hence, we can make a conclusion that predictive power of GAT architecture higher than the FC models one. In further problems we will employ GAT architectures due the higher perfomance.

The dG dataset was prepared in the procedure described in the methods section. Here we provide the results of training GAT model on the dG problem.

| ![dG_result](/figures/dGGBSA.png)|
|:--:|
| *Figure 5. $`\Delta G_{GBSA}`$ prediction. A. Losses of the train and validation samples during training. B. Correlation plots on the train, validation, and test samples, respectively. C. Target and predicted $`\Delta G_{GBSA}`$ distributions in the train, validation, and test samples, respectively.* |
Despite the fact that GAT model did not manage to achieve zero MSE-loss while training (Figure 5.(a)(ссылка?)), it demonstrated a consistent correlation on the test sample. Thanksgiving to careful and crafty dataset splitting into training, validation and test samples, GAT model fitted the training sample well averting an overfitting issue. Green charts at the back of the graphs corresponding to target dG distributions (for training, validation and test samples) are identical and are accurately spanned with red, purple, and yellow prediction distributions (for training, validation and test samples, respectively), which is presented in Figure 5.(c)(ссылка?). Contiguous correlation coefficients together with quite precise predictions attest the fact that the model trained without overfitting: 0.74 on the training sample, 0.74 on the validation sample, and 0.72 on the test sample (Figure 5.(b)(ссылка?). These results let us consider dG prediction task properly solved.

We expected embeddings to be capable of capturing all of the necessary information about the interfaces, and so, we were able to proceed to the main k_d prediction problem. The complexes of PDBbind dataset were processed to graphs and embeddings were extracted from GAT model’s last convolutional layer. These embeddings were then employed as input features for the $K_d$ prediction model. Unfortunately, training with these embeddings did not show any relevant result. All considered models with both different parameters and training hyperparameters tended to predict the average $K_d$ over the training sample.

Alternatively, to predict $K_d$, we exploit embeddings extracted from the pretrained ProteinMPNN model. The prepared PDBbind complexes were passed through the ProteinMPNN encoder, embeddings were used as an input for k_d prediction models (размер представлений?). The training was rather successful, results are provided in the figure below.

| ![kd_ProteinMPNN_Result](/figures/kd_prediction.png)|
|:--:|
| *Figure 6. $`\ln(K_d)`$ prediction. (a) Losses of the train and validation samples during training. (b) Correlation plots on the train, validation, and test samples, respectively. (c) Target and predicted $`\ln(K_d)`$ distributions in the train, validation, and test samples, respectively.* |

As can be seen from Figure 6.(a)(ссылка?), the model tends to overfit: MSE-loss decreases faster in the training sample than in the validation one. Correlation coefficients also attest this: 0.81 on the training sample, 0.38 on the validation sample, and 0.34 on the test sample (Figure 6.(b)(ссылка?)). We deem this result is due to a random splitting into training, validation and test samples: ln(k_d) distributions of the samples significantly differ from each other. To ascertain the difference between the distributions we provide the charts in Figure 6.(c)(ссылка?). Green charts at the back of the graphs corresponding to target k_d distributions (for training, validation and test samples) are easily distinguished, unlike analogous distributions of target dG. Consequently, in spite of quite neat fitting on the training sample (red and green distributions are very alike), distributions of predicted k_d on the validation (purple chart) and test (yellow chart) samples do not match the target ones well. However, the obtained correlation of 0.34 is a quite competitive result, since existing solutions do not provide better conformance to experiments.

<a name="sec6"></a>
### Conclusions:
All the assigned goals were accomplished. Indeed, our results verify a common insight on graph neural networks: this kind of models is an extremely powerful and robust method for predicting physical characteristics of molecules. In our project we successfully predicted energy of electrostatic interaction for a set of charged points and binding free energy for protein-protein complex within the MM/GBSA approximation. In this study we provide a promising approach to $k_d$ prediction based on the complex structure. The proposed method was implemented and tested in real data. Within our workflow a desired competitive result has been obtained, which is depicted by a correlation coefficient of 0.34 on the test sample.  

We intend to continue the project. In our opinion, there is a groundwork for improving our result. First of all, we demand to expand the $K_d$ dataset by adding examples from [SKEMPI](https://life.bsc.es/pid/skempi2) [[10]](SKEMPI) and [MPAD](https://web.iitm.ac.in/bioinfo2/mpad/) [[11]](MPAD) datasets already prepared by us. Second, we hope that more scrupulous model calibration, training hyperparameters selection, and more deliberate data splitting will enable us to achieve a higher level of correlation for dissociation constant prediction.

<a name="sec7"></a>
### References
<a id="selfavoid1">[1]</a>
Rosenbluth M. and Rosenbluth A. Monte Carlo Calculation of the Average Extension of Molecular Chains. 1995. J. Chem. Phys. 23(2): 356–359. https://doi.org/10.1063/1.1741967

<a id="GBSA1">[2]</a>
Kollman, P. A., Massova, I., Reyes, C. et al. Calculating Structures and Free Energies of Complex Molecules:  Combining Molecular Mechanics and Continuum Models. 2000. Accounts of Chemical Research. 33(12): 889-897. https://doi.org/10.1021/ar000033j

<a id="GBSA2">[3]</a>
Genheden, S. and Ryde, U. The MM/PBSA and MM/GBSA methods to estimate ligand-binding affinities. 2015. Expert opinion on drug discovery.  10(5): 1-13 https://doi.org/10.1517/17460441.2015.1032936

<a id="amber">[4]</a>
D.A. Case, H.M. Aktulga, K. Belfon et al. 2023. Amber 2023, University of California, San Francisco.

<a id="biobb">[5]</a>
Andrio, P., Hospital, A., Conejero, J. et al. BioExcel Building Blocks, a software library for interoperable biomolecular simulation workflows. 2019. Sci Data. 6, 169. https://doi.org/10.1038/s41597-019-0177-4

<a id="phenix">[6]</a>
Liebschner, D., Afonine, P. V., Baker, M. L. et al. 2019. Macromolecular structure determination using x-rays, neutrons and electrons: recent developments in phenix. Acta Crystallogr D Struct Biol. 75, 861–877. https://doi:10.1107/S2059798319011471

<a id="PDB2PQR">[7]</a>
Dolinsky, T. D., Czodrowski, P., Li, H. et al. 2007. PDB2PQR: expanding and upgrading automated preparation of biomolecular structures for molecular simulations. 2007. Nucleic Acids Research, Volume 35, Issue suppl_2, 1 July , Pages W522–W525, https://doi.org/10.1093/nar/gkm276

<a id="ProteinMPNN">[8]</a>
Dauparas, J.,  Anishchenko, I.,  Bennett, N.,  et al. 2022. Robust deep learning–based protein sequence design using ProteinMPNN. Science. 378: 49-56. DOI:10.1126/science.add2187

<a id="PDBbind">[9]</a>
Su, M., Yang, Q., Du, Y. et al. Comparative Assessment of Scoring Functions: The CASF-2016 Update. 2019. Journal of chemical information and modeling, 59(2): 895–913. https://doi.org/10.1021/acs.jcim.8b00545

<a id="SKEMPI">[10]</a>
Jankauskaitė, J., Jiménez-García, B., Dapkūnas, J., Fernández-Recio, J., Moal, I.H. 2019. SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and thermodynamics upon mutation. Bioinformatics 35: 462–469 https://doi.org/10.1093/bioinformatics/bty635       

<a id="MPAD">[11]</a>
Ridha, F., Kulandaisamy, A., Gromiha,  M. M. MPAD: A Database for Binding Affinity of Membrane Protein–protein Complexes and their Mutants. 2022. Journal of Molecular Biology. https://doi.org/10.1016/j.jmb.2022.167870
