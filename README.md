<h3 align="center">NNPowerspectra</h3>

<p align="center">
    Code for training neural networks (NNs) to predict powerspectra of galaxy- and matter distributions on the basis of the HOD-Halo model.
</p>

<!-- ABOUT THE PROJECT -->
## About The Project

This code is an extension to <a href="https://github.com/llinke1/G3LHalo_python"> G3LHalo_python by llinke1
</a> and let's you train neural networks to predict powerspectra of galaxy- and matter distributions on the basis of the HOD-Halo model.


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* **python3**: This code will not work for python2! 
* **pyccl**: Check <a href="https://ccl.readthedocs.io/en/latest/"> here </a> for how to install it
* **numpy**
* **scipy**
* **Oportuna**
* **Tensorflow**
* **pyDOE**
* For example notebooks: **matplotlib**

### Installation

To install this code, first goto <a href="https://github.com/llinke1/G3LHalo_python"> G3LHalo_python by llinke1
</a> clone the gitrepo, go to the root folder and execute
```
pip install .
```

Then incorporate any necessary components from this repository.
A brief explanation of the components of this repo is given in the following

<!-- USAGE EXAMPLES -->
## Usage

The folder `exampleNotebooks` contains examples Notebooks to generate Data, analyse Datasets, train NNs via Bayesian Optimization, analyse NN Performance, compare NN Performance. 

### GenerateNNTrainData
This notebook let's you generate homogenous datasets with Latin Hypercube Sampling to train NNs or to test existing ones with the new dataset.

Define the desired dataset size and number of cosmological input parameter here:
```
N_total = 50000
num_params = 10
```
Note: Larg dataset sizes need increasing computation time.
Note: The existing NNs are trained on 10 inputs described in the following.

Define cosmological parameter space here:
```
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
Note: Different parameter spaces can be illogical and therefore result in non-physical results for the power spectra.
Note: The existing NNs are trained on the parameter space given in the example.

For every cosmological configuration the matter-matter, galaxy-matter and galaxy-galaxy power spectra and the related terms of 1halo, 2halo and total get saved in the form of 50 function values in a dictonary of the form:
```
data_dict = {
        'Pk_ss_1h': Pk_ss_1h,
        'Pk_ss_2h': Pk_ss_2h,
        'Pk_ss': Pk_ss,
        'Pk_sl_1h': Pk_sl_1h,
        'Pk_sl_2h': Pk_sl_2h,
        'Pk_sl': Pk_sl,
        'Pk_ll_1h': Pk_ll_1h,
        'Pk_ll_2h': Pk_ll_2h,
        'Pk_ll': Pk_ll,
        'Pk_lin': Pk_lin,
    }
```
The generated power spectra data and the related cosmological parameter configuration then get saved in the Data folder as follows:
```
"../Data/GeneratedData_{num_data_points}_{timestamp}.json"
```
and
```
"../Data/Parameter_{num_data_points}_{timestamp}.json"
```

### GenerateCosmologicalConfigurations
This notebook is similar to GenerateNNTrainData but only generates the Latin Hypercube Sampled cosmological parameter configurations. This is used to feed NNs to get predictions.

### AnalyseDataSet
The AnalyseDataSet notebook let's you analyse your generated dataset to preprocess for NN training or testing. 
It let's you split the data into test, train and validation subsets, perform a cross check over these subsets to confirm reliable performance evaluation and rescale the feature and targets mitigate biases and prepare the data for the NNs. The parameter space of the dataset gets also ploted.

Read in the dataset and parameter configurations here:
```
data_fn = f"../Data/yourDataset.json"
```
and
```
para_fn = f"../Data/yourParameterset.json"
```

Define which 2-point correlation you want to use here:
```
target_keys = ['Pk_ll_1h', 'Pk_ll_2h', 'Pk_ll']
```
In this example we use the Galaxy-Galaxy power spectrum.

Define your desired data split here:
```
training_indices = indices[:50000]
validation_indices = indices[50000:60000]
testing_indices = indices[60000:]
```

Note: Ploting the parameter space can take quite some time depending on the dataset size, it makes therefore sense to only plot the smalles subset, which still gives a good overview of the parameter space:
```
data = validation_features
```

Make sure the rescaling of the features and targets is done with the right subset. For datasets used for training NNs its absolutely crutial to rescale on the training subset and for datasets only used for testing to rescale on the testing subset.
Rescaling is done here for targets:
```
target_processing_vectors = {
    'mean': np.mean(train_targets, axis=1, keepdims=True),
    'sigma': np.std(train_targets, axis=1, keepdims=True)
}
```
and here for features:
```
feature_processing_vectors = {
    'mean': np.array(mean_training_features).reshape(1, -1),
    'sigma': np.array(std_training_features).reshape(1, -1)
}
```
Note: If you rescale training data with the test subset, there is hidden information contained in the rescaled train data biasing the testing of the trained NNs.

### NN_Baysian_Optimization
The following scripts train many NNs to optimze the hyperparameters and find the perfect architecture for the NN with Bayesian Optimization by evaluating its performance over the test subset.
Its therefore in the form of a python script rather then a notebook, because it takes a very long time conducting its hyperparameter study.
It is recommended to run this via "no interrupt" in the background and monitor the process via an output file.

There are 3 different Scripts training different NN setups:
* NN_Bayesian-Optimization trains a combined setup (10 input -> 150 output (50*3 for every term))
* NN_Bayesian-Optimization1halo trains a separat setup (10 input -> 50 output (only 1-halo term))
* NN_Bayesian-Optimization2halo trains a separat setup (10 input -> 50 output (only 2-halo term))

Reading in the Data and rescaling works as decribed for the AnalyseDataSet notebook.

In our case NNs got trained in 3 learning steps to first learn coarse structures in the dataset adn then to finetune. This also ensures comparability between model for the Bayesian Optimization study. 
These learning steps can be adjusted here:
```
patience_values = [100, 100, 1000]
max_epochs = [1000, 1000, 10000]
```

The learning rate gets reduced by a factor of 10 after every learning step here:
```
tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate / (10 ** (i + 1)))
```
The hyperparameter space for the study can be defined here:
```
LAYER_RANGE = (1, 20)
NEURON_RANGE = (16, 2048)
LEARNING_RATE_RANGE = (1e-5, 1e-2)
ACTIVATIONS = ["relu", "selu", "leaky_relu", "elu", "gelu", "swish", "sigmoid", "tanh"]
OPTIMIZERS = ["adam", "RAdam"]
BATCH_SIZE_RANGE = (16, 2048)
DROPOUT_RANGE = (0.0, 0.5)
```
Note: Increasingly larger and deeper NNs need higher training times resulting in a longer time overall to conduct the study.

Trained NNs get listed in the defined logfile (.txt) in the form of:
```
2025-07-08_02-24-56 test_loss=0.000579, validation loss=0.000652, loss=0.000398, train_mae=0.007285, val_mae=0.007716, test_mae=0.007676, params={'n_layers': 4, 'n_units': 1074, 'activation': 'gelu', 'optimizer': 'adam', 'lr': 0.00012, 'batchsize': 1290, 'dropout_rate': 0.131}
```

Furthermore the model and the full training history (loss and validation loss over epochs) get saved:
```
model.save(f'../NN_builds/NN_{timestamp}.keras')

with open(f'../Traininghist/full_training_history_{timestamp}.json', 'w') as f:
        json.dump(training_history, f)
```

The length of the study can be defined here:
```
study.optimize(objective, n_trials=1000)
```

### Bayesian_Optimization_Finetuning

This script let's you finetune NNs by introducing a new hyperparameter N_BAD. The NN is then only trained on the N_BAD worst samples and therefore finetuned.
Define N_BAD here: 
```
LEARNING_RATE_RANGE = (1e-6, 1e-4)
BATCH_SIZE_RANGE = (16, 2048)
EPOCHS=(100, 20000)
NBAD=(10, N_train)
```

### NNAnalyse

The notebook NNAnalyse gives you an detailed analysis for the quality of predictions of a NN.

### NNComparison

With the notebook NNComparison you can compare different NNs (also different setups) by their quality of predictions.

### NNPerformance

This notebook let's you compare different NNs (also different setups) by their speed of predictions.


<!-- LICENSE -->
## License

Distributed under the GNU General Public License v 3.0.

Please cite <a href="https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L/abstract">  Linke et al.(2023) </a> if you use the code for a publication. 

<!-- CONTACT -->
## Contact

Laila Linke - [laila.linke@uibk.ac.at](mailto:laila.linke@uibk.ac.at)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* The code is based on code by Jens Rödiger and his <a href="https://hdl.handle.net/20.500.11811/4086"> PhD Thesis </a> .
* This ReadMe is based on <a href="https://github.com/othneildrew/Best-README-Template"> Best-README-Template </a>

