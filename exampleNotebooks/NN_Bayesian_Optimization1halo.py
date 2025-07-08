# %% [markdown]
# ## This Notebook trains a Neural Network for Galaxy-Galaxy-Powerspektra Predictions

# %%
import g3lhalo
import pyccl as ccl
import matplotlib.pyplot as plt
import numpy as np
import random
from pyDOE import lhs
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.activations import gelu
import optuna
from tensorflow.keras.layers import LeakyReLU, ELU
import re
import ast
import optuna
from optuna.samplers import TPESampler

# Für spezielle Aktivierungen wie Swish, GELU
try:
    from tensorflow.keras.activations import swish, gelu
except ImportError:
    # Alternative falls alte TensorFlow-Version
    def swish(x): return x * tf.nn.sigmoid(x)
    def gelu(x): return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / tf.constant(np.pi)) * (x + 0.044715 * tf.pow(x, 3))))

# Für AdamW und RAdam Optimizer:
try:
    from tensorflow_addons.optimizers import AdamW, RectifiedAdam
except ImportError:
    # Falls nicht installiert:
    # pip install tensorflow-addons
    print("Installiere tensorflow-addons mit: pip install tensorflow-addons")

# %%
#%pip install tensorflow
#%pip install pyDOE
#%pip install pandas
#%pip install seaborn
#%pip install tensorflow-addons
#%pip install filelock
#%pip install pyccl

# %%
# checking that we are using a GPU
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
print('using', device, 'device \n')

# %%
# setting the seed for reproducibility
np.random.seed(9721)
tf.random.set_seed(9721)

# %% [markdown]
# ## Read in generated Targets

# %%
# Create directory to save the generated data
data_dir = "../Data/"
os.makedirs(data_dir, exist_ok=True)

# %%
# 1. Speicherort und Dateiname

data_dir = "../Data/"
data_fn = f"{data_dir}GeneratedData_1000000_2025-06-11_05-16-31.json"

# 2. Relevante Keys für Galaxy-Matter

target_keys = ['Pk_ll_1h', 'Pk_ll_2h', 'Pk_ll']
lin_keys = ['Pk_lin']


# 3. Daten einlesen und filtern

clean_samples = []
kept_indices = []
with open(data_fn, "r") as json_file:
    data_samples = json.load(json_file)
    for i, sample in enumerate(data_samples):
        if all(np.all(np.array(sample[key]) >= 0) for key in target_keys):
            clean_samples.append(sample)
            kept_indices.append(i)

print(f"Anzahl der verbleibenden (bereinigten) Samples: {len(clean_samples)}")


# Create numpy arrays for each of the Pk_* columns
targets_array = {
    key: np.array([sample[key] for sample in clean_samples])
    for key in target_keys
}


# Logarithmic transformation of the data
def log_transform(targets_array):
    return {key: np.log(data) for key, data in targets_array.items()}

#epsilon = 1e-8
#for key in targets_array:
    #targets_array[key] = np.where(targets_array[key] <= 0, epsilon, targets_array[key])

# Calculate logarithms of the data
log_transformed_data = log_transform(targets_array)

num_samples = len(clean_samples)
indices = np.arange(num_samples)
np.random.shuffle(indices)

training_indices = indices[:50000]
validation_indices = indices[50000:60000]
testing_indices = indices[60000:]

N_train = len(training_indices)
N_val = len(validation_indices)
N_test = len(testing_indices)

print(f"Training: {len(training_indices)}")
print(f"Validation: {len(validation_indices)}")
print(f"Test: {len(testing_indices)}")

# 8. Hilfsfunktion zum Splitten
def split_data(array_dict, split_indices):
    return np.array([data[split_indices] for data in array_dict.values()])

# 9. Targets und Modes für Splits
training_targets = {
    'modes': split_data(targets_array, training_indices),
    'targets': split_data(log_transformed_data, training_indices)
}

validation_targets = {
    'modes': split_data(targets_array, validation_indices),
    'targets': split_data(log_transformed_data, validation_indices)
}

testing_targets = {
    'modes': split_data(targets_array, testing_indices),
    'targets': split_data(log_transformed_data, testing_indices)
}

# %%
print('number of training targets:', len(training_targets['modes'][:][0]), '. Should be', len(training_indices))
print('number of validation targets:', len(validation_targets['modes'][:][0]), '. Should be', len(validation_indices))

print('number of test targets:', len(testing_targets['modes'][:][0]), '. Should be', len(testing_indices))
# %%
N_modes=len(training_targets['modes'])
print(f"Shape of training targets: {training_targets['targets'].shape}. Should be ({N_modes}, {len(training_indices)}, 50)")
print(f"Shape of testing targets: {testing_targets['targets'].shape}. Should be ({N_modes}, {len(testing_indices)}, 50)")
print(f"Shape of validation targets: {validation_targets['targets'].shape}. Should be ({N_modes}, {len(validation_indices)}, 50)")


# %% [markdown]
# ## Read in related features

# %%
para_fn = f"{data_dir}Parameter_1000000_2025-06-11_05-16-31.json"

# Read in parameters
with open(para_fn, "r") as json_file:
    param_samples = json.load(json_file)

filtered_param_samples = [param_samples[i] for i in kept_indices]

# Extract feature arrays
feature_keys = ['Om_c', 'Om_b', 'h', 'sigma_8', 'n_s', 'alpha', 'sigma', 'Mth', 'Mprime', 'beta']
feature_array = {
    key: np.array([sample[key] for sample in filtered_param_samples])
    for key in feature_keys
}

# Split
training_features = {key: feature[training_indices] for key, feature in feature_array.items()}
validation_features = {key: feature[validation_indices] for key, feature in feature_array.items()}
testing_features = {key: feature[testing_indices] for key, feature in feature_array.items()}

# Convert to final numpy arrays
training_features = np.array(list(zip(*training_features.values())))
validation_features = np.array(list(zip(*validation_features.values())))
testing_features = np.array(list(zip(*testing_features.values())))



# %%
log_indices = [7, 8]  # Ersetze mit deinen echten Indizes

training_features[:, log_indices] = np.log10(training_features[:, log_indices])
validation_features[:, log_indices] = np.log10(validation_features[:, log_indices])
testing_features[:, log_indices] = np.log10(testing_features[:, log_indices])

print("input min/max:", training_features.min(), training_features.max())

# %%
print('number of training features:', len(training_features), '. Should be', len(training_indices))
print('number of validation features:', len(validation_features), '. Should be', len(validation_indices))

print('number of test features:', len(testing_features), '. Should be', len(testing_indices))

# %% [markdown]
# ## Cross check
# Now, for a cross check we plot the mean of the training, testing and validation data sets. These should all look roughly the same, otherwise we have selected strange test and validation sets.

# %%
mean_training_targets = []
std_training_targets = []

mean_testing_targets = []
std_testing_targets = []

mean_validation_targets = []
std_validation_targets = []

# Calculate mean and standard deviation for training targets
for i in range(len(training_targets['modes'])):  # for all 3-Halo terms
    halo_data = np.array(training_targets['targets'][i])
    
    mean = np.mean(halo_data, axis=0)  # mean over complete training sample for every k-value
    std = np.std(halo_data, axis=0)    # standard deviation over complete training sample for every k-value
    
    mean_training_targets.append(mean)
    std_training_targets.append(std)

# Calculate mean and standard deviation for testing targets
for i in range(len(testing_targets['modes'])):  # for all 3-Halo terms
    halo_data = np.array(testing_targets['targets'][i])

    mean = np.mean(halo_data, axis=0) # mean over complete testing sample for every k-value
    std = np.std(halo_data, axis=0)    # standard deviation over complete testing sample for every k-value

    mean_testing_targets.append(mean)
    std_testing_targets.append(std)

# Calculate mean and standard deviation for validation targets
for i in range(len(validation_targets['modes'])):  # for all 3-Halo terms
    halo_data = np.array(validation_targets['targets'][i])

    mean = np.mean(halo_data, axis=0)  # mean over complete validation sample for every k-value
    std = np.std(halo_data, axis=0)    # standard deviation over complete validation sample for every k-value

    mean_validation_targets.append(mean)
    std_validation_targets.append(std)


# %%
ks = np.geomspace(1e-2, 1e2)

fig, axes=plt.subplots()

plt.xlabel(r'$k$ [Mpc$^{-1}$]')
plt.ylabel(r'$\ln(P(k))$')
plt.title('Cross Check: Mean of targets')


ax2=axes.twinx()
ax2.plot(np.NaN, np.NaN, ls='--', label='1-halo', color='grey')
ax2.plot(np.NaN, np.NaN, ls=':', label='2-halo', color='grey')
ax2.plot(np.NaN, np.NaN, ls='-', label='total', color='grey')




ax2.legend(loc='lower left')


plt.xlim(1e-2,100)
plt.ylim(-10, 15)
axes.set_ylim(-10, 15)
ax2.set_yticks([])
axes.set_xscale('log')
axes.set_yscale('linear')


# Plot training targets
plt.errorbar(ks, mean_training_targets[0], ls='--',
             yerr=std_training_targets[0], color='C0', label='Mean of training targets', fmt='')
plt.errorbar(ks, mean_training_targets[1], ls=':', 
             yerr= std_training_targets[1], color='C0', fmt='')
plt.errorbar(ks, mean_training_targets[2], 
             yerr=std_training_targets[2], color='C0', fmt='')

# Plot testing targets
plt.errorbar(ks, mean_testing_targets[0]+0.5, ls='--',
             yerr=std_testing_targets[0], color='C1', label='Mean of testing targets', fmt='')
plt.errorbar(ks, mean_testing_targets[1]+0.5, ls=':', 
             yerr= std_testing_targets[1], color='C1', fmt='')
plt.errorbar(ks, mean_testing_targets[2]+0.5, 
             yerr=std_testing_targets[2], color='C1', fmt='')

# Plot validation targets
plt.errorbar(ks, mean_validation_targets[0]+1, ls='--',
             yerr=std_validation_targets[0], color='C2', label='Mean of validation targets', fmt='')
plt.errorbar(ks, mean_validation_targets[1]+1, ls=':', 
             yerr= std_validation_targets[1], color='C2', fmt='')
plt.errorbar(ks, mean_validation_targets[2]+1, 
             yerr=std_validation_targets[2], color='C2', fmt='')

plt.legend()
plt.savefig('../Plots/CrossCheckTargets.png', dpi=600)
plt.show()

# %%
mean_training_features = []
std_training_features = []

mean_testing_features = []
std_testing_features = []

mean_validation_features = []
std_validation_features = []

# Calculate mean and standard deviation for training features
for i in range(len(training_features[0])):  # for all 3-Halo terms
    feature_parameter = np.array(training_features[:,i])
    
    mean = np.mean(feature_parameter, axis=0)  # mean over complete training sample for every k-value
    std = np.std(feature_parameter, axis=0)    # standard deviation over complete training sample for every k-value
    
    mean_training_features.append(mean)
    std_training_features.append(std)

# Calculate mean and standard deviation for testing features
for i in range(len(testing_features[0])):  # for all 3-Halo terms
    feature_parameter = np.array(testing_features[:,i])

    mean = np.mean(feature_parameter, axis=0) # mean over complete testing sample for every k-value
    std = np.std(feature_parameter, axis=0)    # standard deviation over complete testing sample for every k-value

    mean_testing_features.append(mean)
    std_testing_features.append(std)

# Calculate mean and standard deviation for validation features
for i in range(len(validation_features[0])):  # for all 3-Halo terms
    feature_parameter = np.array(validation_features[:,i])

    mean = np.mean(feature_parameter, axis=0)  # mean over complete validation sample for every k-value
    std = np.std(feature_parameter, axis=0)    # standard deviation over complete validation sample for every k-value

    mean_validation_features.append(mean)
    std_validation_features.append(std)




# %%
x_pos = np.arange(len(feature_keys))  # numerische Positionen für die x-Werte
offset = 0.2  # Abstand, um die Features ein bisschen zu verschieben

fig, axes = plt.subplots()

axes.set_xlabel('Features')
axes.set_ylabel('Value')
axes.set_title('Cross Check: Mean of features')

# Optional: Y-Achse logarithmisch
axes.set_xscale('linear')
axes.set_yscale('log')

# Training Features
axes.errorbar(x_pos - offset, mean_training_features, 
              yerr=std_training_features, ls='none', 
              color='C0', label='Mean of training features', capsize=5)

# Testing Features
axes.errorbar(x_pos, mean_testing_features, 
              yerr=std_testing_features, ls='none', 
              color='C1', label='Mean of testing features', capsize=5)

# Validation Features
axes.errorbar(x_pos + offset, mean_validation_features, 
              yerr=std_validation_features, ls='none', 
              color='C2', label='Mean of validation features', capsize=5)

# Setze die Original-Feature-Namen als x-Achsen-Beschriftung
axes.set_xticks(x_pos)
axes.set_xticklabels(['$\\Omega_c$', '$\\Omega_b$', '$h$', '$\\sigma_8$', '$n_s$', '$\\alpha$', '$\\sigma$', '$\log_{10}(M_{th})$', '$\log_{10}(M_{prime})$', '$\\beta$' ])
plt.xticks(rotation=45, ha='right')

# Legende
axes.legend()

plt.tight_layout()
plt.savefig('../Plots/CrossCheckFeatures.png', dpi=600)
plt.show()

# %% [markdown]
# feature_keys = ['Om_c', 'Om_b', 'h', 'sigma_8', 'n_s', 'alpha', 'sigma', 'Mth', 'Mprime', 'beta']

# %% [markdown]
# ## Renaming

# %%
# features
train_targets=training_targets['targets']
test_targets=testing_targets['targets']
val_targets=validation_targets['targets']

# %% [markdown]
# ## Rescaling
# We are rescaling the features by the mean and standard deviation of the training features. Then, they are all scattering by 1 and have a mean value of 0.
# 

# %%
# Calculate mean and standard deviation **only from the training data**
target_processing_vectors = {
    'mean': np.mean(train_targets, axis=1, keepdims=True),
    'sigma': np.std(train_targets, axis=1, keepdims=True)
}


# Ensure that no division by zero occurs
target_processing_vectors['sigma'][target_processing_vectors['sigma'] == 0] = 1  

# Preprocessing function
def preprocessing(targets, processing_vectors):
    return (targets - processing_vectors['mean']) / processing_vectors['sigma']

# Postprocessing function
def postprocessing(targets, processing_vectors):
    return targets * processing_vectors['sigma'] + processing_vectors['mean']

# Apply preprocessing to all data
train_targets_rescaled = preprocessing(train_targets, target_processing_vectors)
test_targets_rescaled = preprocessing(test_targets, target_processing_vectors)
val_targets_rescaled = preprocessing(val_targets, target_processing_vectors)

# JSON-compatible storage of the processing vectors
#serializable_processing_vectors = {key: value.tolist() for key, value in processing_vectors.items()}
#processing_out="../Emulators/NM_processing_vectors.json"
#with open(processing_out, "w") as json_file:
#    json.dump(serializable_processing_vectors, json_file)

# Check shapes after the transformation
print("Train Rescaled Shape:", train_targets_rescaled.shape)
print("Test Rescaled Shape:", test_targets_rescaled.shape)
print("Val Rescaled Shape:", val_targets_rescaled.shape)



# %%
mean_training_targets_rescaled = []
std_training_targets_rescaled = []

mean_testing_targets_rescaled = []
std_testing_targets_rescaled = []

mean_validation_targets_rescaled = []
std_validation_targets_rescaled = []

# Calculate mean and standard deviation for rescaled training targets
for i in range(len(train_targets_rescaled)):  # for all 3-Halo terms
    halo_data_rescaled = np.array(train_targets_rescaled[i])

    mean = np.mean(halo_data_rescaled, axis=0)  # mean over complete rescaled training sample for every k-value
    std = np.std(halo_data_rescaled, axis=0)    # standard deviation over complete rescaled training sample for every k-value

    mean_training_targets_rescaled.append(mean)
    std_training_targets_rescaled.append(std)

# Calculate mean and standard deviation for rescaled testing targets
for i in range(len(test_targets_rescaled)):  # for all 3-Halo terms
    halo_data_rescaled = np.array(test_targets_rescaled[i])

    mean = np.mean(halo_data_rescaled, axis=0)  # mean over complete rescaled testing sample for every k-value
    std = np.std(halo_data_rescaled, axis=0)    # standard deviation over complete rescaled testing sample for every k-value

    mean_testing_targets_rescaled.append(mean)
    std_testing_targets_rescaled.append(std)

# Calculate mean and standard deviation for rescaled validation targets
for i in range(len(val_targets_rescaled)):  # for all 3-Halo terms
    halo_data_rescaled = np.array(val_targets_rescaled[i])

    mean = np.mean(halo_data_rescaled, axis=0)  # mean over complete rescaled validation sample for every k-value
    std = np.std(halo_data_rescaled, axis=0)    # standard deviation over complete rescaled validation sample for every k-value

    mean_validation_targets_rescaled.append(mean)
    std_validation_targets_rescaled.append(std)

# %%
fig, axes=plt.subplots()
#axes.loglog(ks, Pk_lin, color='k', label='linear')
plt.xlabel(r'$k$ [Mpc$^{-1}$]')
plt.ylabel(r'Rescaled $(P(k))$')
plt.title('Rescaled targets')

ax2=axes.twinx()
ax2.plot(np.NaN, np.NaN, ls='--', label='1-halo', color='grey')
ax2.plot(np.NaN, np.NaN, ls=':', label='2-halo', color='grey')
ax2.plot(np.NaN, np.NaN, ls='-', label='total', color='grey')
ax2.set_yticks([])



plt.errorbar(ks,mean_training_targets_rescaled[0], 
             yerr=std_training_targets_rescaled[0], color='C0', label='Mean of training targets')
plt.errorbar(ks,mean_training_targets_rescaled[1], ls=':', 
             yerr=std_training_targets_rescaled[1], color='C0')
plt.errorbar(ks,mean_training_targets_rescaled[2], ls='--', 
             yerr=std_training_targets_rescaled[2], color='C0')

plt.errorbar(ks*1.06,mean_testing_targets_rescaled[0], 
             yerr=std_testing_targets_rescaled[0], color='C1', label='Mean of testing targets')
plt.errorbar(ks*1.06,mean_testing_targets_rescaled[1], ls=':', 
             yerr=std_testing_targets_rescaled[1], color='C1')
plt.errorbar(ks*1.06,mean_testing_targets_rescaled[2], ls='--', 
             yerr=std_testing_targets_rescaled[2], color='C1')

plt.errorbar(ks*1.12,mean_validation_targets_rescaled[0], 
             yerr=std_validation_targets_rescaled[0], color='C2', label='Mean of validation targets')
plt.errorbar(ks*1.12,mean_validation_targets_rescaled[1], ls=':', 
             yerr=std_validation_targets_rescaled[1], color='C2')
plt.errorbar(ks*1.12,mean_validation_targets_rescaled[2], ls='--', 
             yerr=std_validation_targets_rescaled[2], color='C2')

axes.set_ylim(-1.3, 1.3)
plt.xlim(1e-2,100)
plt.ylim(-1.3,1.3)
plt.xscale('log')
plt.legend()
plt.savefig('../Plots/RescalingTargets.png', dpi=600)

# %%
# Calculate mean and standard deviation **only from the training data**
feature_processing_vectors = {
    'mean': np.array(mean_training_features).reshape(1, -1),
    'sigma': np.array(std_training_features).reshape(1, -1)
}



# Ensure that no division by zero occurs
feature_processing_vectors['sigma'][feature_processing_vectors['sigma'] == 0] = 1  

# Preprocessing function
def preprocessing(features, processing_vectors):
    return (features - processing_vectors['mean']) / processing_vectors['sigma']

# Postprocessing function
def postprocessing(features, processing_vectors):
    return features * processing_vectors['sigma'] + processing_vectors['mean']

# Apply preprocessing to all data
train_features_rescaled = preprocessing(training_features, feature_processing_vectors)
test_features_rescaled = preprocessing(testing_features, feature_processing_vectors)
val_features_rescaled = preprocessing(validation_features, feature_processing_vectors)

# JSON-compatible storage of the processing vectors
#serializable_processing_vectors = {key: value.tolist() for key, value in processing_vectors.items()}
#processing_out="../Emulators/NM_processing_vectors.json"
#with open(processing_out, "w") as json_file:
#    json.dump(serializable_processing_vectors, json_file)

# Check shapes after the transformation
print("Train Rescaled Shape:", train_features_rescaled.shape)
print("Test Rescaled Shape:", test_features_rescaled.shape)
print("Val Rescaled Shape:", val_features_rescaled.shape)



# %%


mean_training_features_rescaled = []
std_training_features_rescaled = []

mean_testing_features_rescaled = []
std_testing_features_rescaled = []

mean_validation_features_rescaled = []
std_validation_features_rescaled = []


# Calculate mean and standard deviation for training features
for i in range(len(train_features_rescaled[0])):  # for all 3-Halo terms
    feature_parameter = np.array(train_features_rescaled[:,i])
    
    mean = np.mean(feature_parameter, axis=0)  # mean over complete training sample for every k-value
    std = np.std(feature_parameter, axis=0)    # standard deviation over complete training sample for every k-value
    mean_training_features_rescaled.append(mean)
    std_training_features_rescaled.append(std)

# Calculate mean and standard deviation for testing features
for i in range(len(test_features_rescaled[0])):  # for all 3-Halo terms
    feature_parameter = np.array(test_features_rescaled[:,i])

    mean = np.mean(feature_parameter, axis=0) # mean over complete testing sample for every k-value
    std = np.std(feature_parameter, axis=0)    # standard deviation over complete testing sample for every k-value

    mean_testing_features_rescaled.append(mean)
    std_testing_features_rescaled.append(std)

# Calculate mean and standard deviation for validation features
for i in range(len(val_features_rescaled[0])):  # for all 3-Halo terms
    feature_parameter = np.array(val_features_rescaled[:,i])

    mean = np.mean(feature_parameter, axis=0)  # mean over complete validation sample for every k-value
    std = np.std(feature_parameter, axis=0)    # standard deviation over complete validation sample for every k-value

    mean_validation_features_rescaled.append(mean)
    std_validation_features_rescaled.append(std)

# %%
x_pos = np.arange(len(feature_keys))  # numerische Positionen für die x-Werte
offset = 0.2  # Abstand, um die Features ein bisschen zu verschieben

fig, axes = plt.subplots()

axes.set_xlabel('Features')
axes.set_ylabel('Value')
axes.set_title('Rescaled features')

# Optional: Y-Achse logarithmisch
axes.set_xscale('linear')
axes.set_yscale('linear')

# Training Features
axes.errorbar(x_pos - offset, mean_training_features_rescaled, 
              yerr=std_training_features_rescaled, ls='none', 
              color='C0', label='Mean of training features', capsize=5)

# Testing Features
axes.errorbar(x_pos, mean_testing_features_rescaled, 
              yerr=std_testing_features_rescaled, ls='none', 
              color='C1', label='Mean of testing features', capsize=5)

# Validation Features
axes.errorbar(x_pos + offset, mean_validation_features_rescaled, 
              yerr=std_validation_features_rescaled, ls='none', 
              color='C2', label='Mean of validation features', capsize=5)

# Setze die Original-Feature-Namen als x-Achsen-Beschriftung
axes.set_xticks(x_pos)
axes.set_xticklabels(['$\\Omega_c$', '$\\Omega_b$', '$h$', '$\\sigma_8$', '$n_s$', '$\\alpha$', '$\\sigma$', '$\log_{10}(M_{th})$', '$\log_{10}(M_{prime})$', '$\\beta$' ])
plt.xticks(rotation=45, ha='right')


axes.set_ylim(-1.3, 1.3)
plt.ylim(-1.3,1.3)
# Legende
axes.legend()

plt.tight_layout()
plt.savefig('../Plots/RescalingFeatures.png', dpi=600)
plt.show()

# %% [markdown]
# ## NN training

# %% [markdown]
# ### Define Model

# %% [markdown]
# Define Neural Network with following structure

# %% [markdown]
# ### Set training hyperparameters
# We set some hyperparameters: How many features are processed in one step (batch_size), how fast the gradient descent should happen (learning_Rate), after how many steps without improvement the learning should stop (patience_values) and what is the maximal number of learning steps (max_epochs).
# 
# We also set where to save the emulator

# %%

# %%
#batch_sizes=[1024, 1024, 1024, 1024, 1024, 1024]
#learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
#patience_values=[100, 100, 1000, 1000, 1000, 1000]
#max_epochs=[1000, 1000, 10000, 10000, 10000, 10000]
patience_values=[100, 100, 1000]
max_epochs=[1000, 1000, 10000]
model_filename="../Emulators/NN_GM"

# %%
#restructuring of the arrays
train_targets_rescaled      = np.transpose(train_targets_rescaled, (1, 0, 2))
val_targets_rescaled  = np.transpose(val_targets_rescaled, (1, 0, 2))
test_targets_rescaled = np.transpose(test_targets_rescaled, (1, 0, 2))


# %%
# Parametergrenzen definieren

# Parametergrenzen definieren
LAYER_RANGE = (1, 20)
NEURON_RANGE = (16, 2048)
LEARNING_RATE_RANGE = (1e-5, 1e-2)
ACTIVATIONS = ["relu", "selu", "leaky_relu", "elu", "gelu", "swish", "sigmoid", "tanh"]
OPTIMIZERS = ["adam", "RAdam"]
BATCH_SIZE_RANGE = (16, 2048)
DROPOUT_RANGE = (0.0, 0.5)

log_file = "training_runsv5_1halo.txt"
patience_values = [100, 100, 1000]
max_epochs = [1000, 1000, 10000]
model_filename = "../Emulators/NN_GM"

# Dummy-Platzhalter (ersetzen mit echten Daten)
# train_features_rescaled, train_targets_rescaled, val_features_rescaled, val_targets_rescaled
# N_train, N_val müssen gesetzt sein

def parse_training_log(file_path):
    completed_trials = []
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(r"test_loss=(.*?), validation loss=(.*?), loss=(.*?), val_mae=(.*?), params=(\{.*\})", line)
                if match:
                    val_loss = float(match.group(2))
                    params = ast.literal_eval(match.group(5))
                    params['lr'] = float(params['lr'])
                    completed_trials.append((params, val_loss))
    except FileNotFoundError:
        pass
    return completed_trials

def build_model(n_layers, n_units, activation, input_dim, dropout_rate):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for i in range(n_layers):
        if activation == "leaky_relu":
            model.add(layers.Dense(n_units))
            model.add(layers.LeakyReLU())
        else:
            model.add(layers.Dense(n_units, activation=activation))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(50, activation='linear'))
    return model

class FileAwareSampler(TPESampler):
    def __init__(self, log_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file

    def sample_relative(self, study, trial, search_space):
        existing_trials = parse_training_log(self.log_file)
        for _ in range(100):  # max 100 Versuche
            params = super().sample_relative(study, trial, search_space)
            already_tested = any(all(params.get(k) == v for k, v in t[0].items()) for t in existing_trials)
            if not already_tested:
                return params
        raise RuntimeError("Keine neuen Parameterkombinationen gefunden.")

def objective(trial):
    all_training_loss = []
    all_validation_loss = []

    n_layers = trial.suggest_int("n_layers", *LAYER_RANGE)
    n_units = trial.suggest_int("n_units", *NEURON_RANGE)
    activation = trial.suggest_categorical("activation", ACTIVATIONS)
    learning_rate = trial.suggest_float("lr", *LEARNING_RATE_RANGE, log=True)
    optimizer_choice = trial.suggest_categorical("optimizer", OPTIMIZERS)
    batch_size = trial.suggest_int("batch_size", *BATCH_SIZE_RANGE)
    dropout_rate = trial.suggest_float("dropout_rate", *DROPOUT_RANGE)

    model = build_model(n_layers, n_units, activation, train_features_rescaled.shape[1], dropout_rate)

    # Optimizer wählen
    if optimizer_choice == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_choice == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_choice == "AdamW":
        from tensorflow_addons.optimizers import AdamW
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif optimizer_choice == "RAdam":
        from tensorflow_addons.optimizers import RectifiedAdam
        optimizer = RectifiedAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    print(f"Training model with params: layers={n_layers}, units={n_units}, activation={activation}, optimizer={optimizer_choice}, lr={learning_rate}, batchsize={batch_size}, dropout={dropout_rate}")

    for i in range(len(max_epochs)):
        earlystop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_values[i], restore_best_weights=True, verbose=0
        )
        history = model.fit(
            train_features_rescaled,
            train_targets_rescaled.reshape(N_train, 150)[:, :50],
            epochs=max_epochs[i],
            verbose=0,
            validation_data=(val_features_rescaled, val_targets_rescaled.reshape(N_val, 150)[:, :50]),
            callbacks=[earlystop],
            batch_size=batch_size
        )

        all_training_loss.extend(history.history['loss'])
        all_validation_loss.extend(history.history['val_loss'])

        tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate / (10 ** (i + 1)))
        print(f"Finish Trainingstep {i+1} of {len(max_epochs)} with lr={learning_rate} and {max_epochs[i]} epochs")

    loss, train_mae = model.evaluate(train_features_rescaled, train_targets_rescaled.reshape(N_train, 150)[:, :50], verbose=0)
    val_loss, val_mae = model.evaluate(val_features_rescaled, val_targets_rescaled.reshape(N_val, 150)[:, :50], verbose=0)
    test_loss, test_mae = model.evaluate(test_features_rescaled, test_targets_rescaled.reshape(N_test, 150)[:, :50], verbose=0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(timestamp)

    with open(log_file, "a") as f:
        f.write(f"{timestamp} test_loss={test_loss:.6f}, validation loss={val_loss:.6f}, loss={loss:.6f}, train_mae={train_mae:.6f}, val_mae={val_mae:.6f}, test_mae={test_mae:.6f}, ")
        f.write(f"params={{'n_layers': {n_layers}, 'n_units': {n_units}, 'activation': '{activation}', 'optimizer': '{optimizer_choice}', 'lr': {learning_rate}, 'batchsize': {batch_size}, 'dropout_rate': {dropout_rate}}}\n")

    training_history = {
        'loss': all_training_loss,
        'val_loss': all_validation_loss
    }

    model.save(f'../NN_builds/NN_{timestamp}.keras')

    with open(f'../Traininghist/full_training_history_{timestamp}.json', 'w') as f:
        json.dump(training_history, f)

    return test_loss

# Studie starten
random_seed = random.randint(0, 420)
print('using random seed: ', random_seed)
sampler = FileAwareSampler(log_file=log_file, seed=random_seed)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=1000)

print("Beste Parameter:")
print(study.best_params)

