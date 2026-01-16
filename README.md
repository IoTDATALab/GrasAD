# GrasAD
This is the experimental code for the paper "GrasAD: Gradient Similarity and Accumulated Distribution-Based Device Selection for Asynchronous Federated Learning".

## For the installation of the experimental environment, please refer to alibaba/FederatedScope: An easy-to-use federated learning platform (github.com). 

## After installation, you can run the corresponding experiments with the following command:
python federatedscope/main.py --cfg scripts/noniidagg_algorithms/celeba_usrsele_grasad_vgg11_alpha_0_5_leaf.yaml

Here, "celeba" represents the dataset name, "grasad" is the algorithm name, "vgg11" denotes the model, and "0_5" stands for the alpha value in Dirichlet. Users can open the yaml file to modify the corresponding hyperparameters.

### Datasets
The available dataset options are leaf_celeba, leaf_femnist, leaf_synthetic, celeba, femnist, and synthetic. Datasets with "leaf" in their names allow modification of data non - independent and identically distributed properties; otherwise, the official data splitting method is used.
### Comparable Algorithms
Algorithm names include usrsele_compa_grasad, usrsele_compa_aouprior, usrsele_compa_schandagg, usrsele_compa_hfl, usrsele_compa_kafl, etc. When using an algorithm, set the corresponding parameter value in the yaml file to True. Note that each time you use an algorithm, you must ensure that asy.use == True to confirm asynchronous federated updates. 
For example, when using the GrasAD algorithm, you must set:

asy:
use: True
usrsele_compa_hfl:
use: True
# Acknowledgement
We would like to thank the authors for releasing the public repository: FederatedScope-LLM.
