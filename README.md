
<h3 align="center">NNPowerspectra</h3>

<p align="center">
    Code for training neural networks, all related data processing and model evaluation
</p>

<!-- ABOUT THE PROJECT -->
## About The Project

This code uses the halo model and HODs to model the powerspectra of galaxy- and matter distributions to train neural networks with this data in order to accelerate modeling of powerspectra. This code is an extension to <a href="https://github.com/llinke1/G3LHalo_python"> G3LHalo_python </a>, for further insights see <a href="https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L/abstract"> Linke, Simon, Schneider + (2023) </a>. This code is based on my bachelor thesis for further insights see my thesis in the root folder of the repo. 


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* **python3**: This code will not work for python2! 
* **pyccl**: Check <a href="https://ccl.readthedocs.io/en/latest/"> here </a> for how to install it
* **numpy**
* **scipy**
* **pyDOE**
* **tensorflow**
* **optuna**
* For example notebooks: **matplotlib**

### Installation

To install this code you first have to get the code for the <a href="https://github.com/llinke1/G3LHalo_python"> G3LHalo_python </a>.
Afterwards you can clone this repo and use the provided notebooks and scripts.

<!-- USAGE EXAMPLES -->
## Usage
This code let's u generate powerspectra data, analyse the generated datasets, preprocess datasets for NN training, train NNs with Bayesian Optimization, evaluate NN performance (accuracy and speed), finetune NNs, compare different NNs to each other. It also provides an emulator-script to predict powerspectra given a dataset of parameter configurations and an NN model.

### GenerateNNTrainData
This Notebook calculates powerspectra for given cosmological configurations. Generating large datasets using Latin-Hypercube-Sampling to train neural networks on. For data generation the desired dataset size and parameter priors can be defined on top of the notebook.

### GenerateCosmologicalConfigurations
This Notebook generates cosmological parameter configurations given the defined parameter priors using Latin-Hypercube-Sampling. These parameter configurations are later used in combination with NN models as input parameters to predict powerspectra. For parameter generation the desired parameterset size and parameter priors can be defined on top of the notebook.

### AnalyseDataSet
This Notebook analyses a dataset given powerspectra data and matching cosmologial parameters. On top of the notebook the dataset and powerspectrum mode are defined. Powerspectrum_mode representing which correlation we're looking at: ss = matter-matter (source-source), sl = galaxy-matter (lens-source), ll = galaxy-galaxy (lens-lens). The data gets split into training, validation and test subsets to prepare NN training. Samples with negative powerspectra get filtered. The parameterspace of the dataset is displayed for further analysis. The data gets log transformed and preprocessed for training meaning a cross-check over all subsets and rescaling on the training subset to 0 mean and a standard deviation of 1. The results are plotted for further analysis and the rescaling vectors are saved.

### NN_Bayesian_Optimization
Trains NN via Bayesian Optimization given a dataset, parameterset, rescaling vectors, powerspectrum mode, subset size, hyperparmeter ranges, bayesian opti logfile and bayesian opti study lenght.
NNs can be trained via Bayesian Optimization for 2 architectures: One complete approach and a separate approach, which predicts the one and two halo term of the spectra with separate NNs and calcultes the total powerspectra by summing the values of the one and two halo terms.

### Bayesian-Optimization_Finetuning
Finetunes an existing NN model by only training it on the n worst samples. $N_{BAD}$ is added as a hyperparameter for the bayesian Opztimization.

### NNAnalyse
This Notebook evaluates the accuracy of a NN model with the test data subset. It displays the training process via training and validation convergence, shows relative errors of the predictions in log- and realspace, shows the powerspectra predictions for a random and the worst sample and shows the parameterspace with the prediction error for each sample for further analysis. Additionaly at the bottom you can filter the dataset by a certain parameter and new range for that and see how the model performs in this region of the parameterspace.

### NNComparison
This Notebook compares different NN models accuracy whise. Showing the training processes, the realative errors of each 1halo 2halo and total for each model in log- and realspace. Furthermore it plots the parameterspace with the prediction error for each sample and each model for further analysis. It also compares the realtive errors to each other to determine which model performs best.

### NNNCalcLoss
In progress  (Notebook will get renamed and used to analyse the Bayesian Optimization progress of the training)

### NNPerformance
In progress (will be a Notebook to compare the speed of diffrent models)

### PredPowerspectra
Emulator-Script to predict powerspectra given a parameterset, processing vectors, powerspectrum mode and NN model. The predicted powerspectra are then saved into a file as if they would be calculated with G3LHalo-Code.

### TestPredPowerspectra
Notebook to read in a calculated and predicted file to compare and test if predictions have desired accuracy.

<!-- WORKFLOW -->
## Workflow
In the following there is a brief explanation of example workflows this code can be used for:

### Training NN models to predict powerspectra
* Generate a dataset with your defined parameter priors (Generate NNTrainData.py)
* Analyse dataset: aim for homogeneous distribution of samples and cross check and rescaling should be equal over all subsets. This notebooks also creates the processing vectors. (AnalyseDataSet.ipynb)
* Train NN models via Bayesian Optimization with the created dataset and processing vectors. (NN_Bayesian_Optimization.py)
* Finetune your NN via Bayesian Optimization. (Bayesian_optimization_Finetuning.py)
* Evaluate your best models prediction accuracy. (NNAnalyse.ipynb)
* Compare different models prediction accuracy. (NNComparison.ipynb)

### Predicting powerspectra
* Generate a parameterset with your defined parameter priors (GenerateCosmologicalConfigurations.py)
* Emulate powerspectra with parameterset, NN model and processing vetors used training the model. (PredPowerspectra.py)
* Test your predicted powerspectra. (TestPredPowerspectra.ipynb)


IMPORTANT: Note that a NN model is allways trained on a dataset with limited parameter priors and that the exact rescaling vectors used in training are needed for predicting. Otherwise new rescaling vectors can hold hidden information about the new data and could falsify the results.

## Fileshare
On the following link you can find the best NN models for GG, GM and MM my Bayesian Optimization produced.
You can also find the dataset (10^6 samples, size: 15GB) these models are trained on and the associated processing vectors.
https://fileshare.uibk.ac.at/d/d0cbb73f649c4a98aa9e/

The dataset has the following prior ranges:
```
# Define parameter intervals
Mth_min = 1e10
Mth_max = 1e15
param_intervals = {
    'Om_c': (0.1, 0.6),
    'Om_b': (0.04, 0.06),
    'h': (0.64, 0.82),
    'sigma_8': (0.8, 1),
    'n_s': (0.84, 1.1),
    'alpha': (0, 1),
    'sigma': (0.01, 1),
    'Mth': (Mth_min, Mth_max),
    'Mprime': (Mth_min * 1, Mth_max * 100),
    'beta': (0.1, 2)
}
```

<!-- LICENSE -->
## License

Distributed under the GNU General Public License v 3.0.

Please cite <a href="https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L/abstract">  Linke et al.(2023) </a> if you use the code for a publication. 

<!-- CONTACT -->
## Contact

Robin Hoffmann - [robin.hoffmann@student.uibk.ac.at](mailto:robin.hoffmann@student.uibk.ac.at)
Laila Linke - [laila.linke@uibk.ac.at](mailto:laila.linke@uibk.ac.at)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* The code is based on code by Jens Rödiger and his <a href="https://hdl.handle.net/20.500.11811/4086"> PhD Thesis </a> .
* This ReadMe is based on <a href="https://github.com/othneildrew/Best-README-Template"> Best-README-Template </a>

