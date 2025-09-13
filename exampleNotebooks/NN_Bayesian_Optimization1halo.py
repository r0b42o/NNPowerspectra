# Define dataset, mode, subset size
dataset = "1000000_2025-06-11_05-16-31"
powerspectrum_mode = "ll"
training_size = 50000
validation_size = 10000

# Define hyperparameter ranges
LAYER_RANGE = (1, 20)
NEURON_RANGE = (16, 2048)
LEARNING_RATE_RANGE = (1e-5, 1e-2)
ACTIVATIONS = ["relu", "selu", "leaky_relu", "elu", "gelu", "swish", "sigmoid", "tanh"]
OPTIMIZERS = ["adam", "RAdam"]
BATCH_SIZE_RANGE = (16, 2048)
DROPOUT_RANGE = (0.0, 0.5)
patience_values = [100, 100, 1000]
max_epochs = [1000, 1000, 10000]

# Define logfile for Bayesian Optimization
log_file = "training_runsv5_1halo.txt"

# Define study lenght for Bayesian Optimization
bayesian_trials = 1000

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
data_fn = f"{data_dir}GeneratedData_{dataset}.json"

# 2. Relevante Keys für Galaxy-Matter

target_keys = [f'Pk_{powerspectrum_mode}_1h', f'Pk_{powerspectrum_mode}_2h', f'Pk_{powerspectrum_mode}']


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

training_indices = indices[:training_size]
validation_indices = indices[training_size:(training_size+validation_size)]
testing_indices = indices[(training_size+validation_size):]

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
para_fn = f"{data_dir}Parameter_{dataset}.json"

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
# --- Preprocessing & Postprocessing Functions ---
def preprocessing(targets, processing_vectors):
    return (targets - processing_vectors['mean']) / processing_vectors['sigma']

def postprocessing(targets, processing_vectors):
    return targets * processing_vectors['sigma'] + processing_vectors['mean']


# --- Lade die gespeicherten Rescaling-Vektoren ---
processing_in = f"../Data/Rescaling/Processing_vectors_data_{dataset}_{powerspectrum_mode}.json"

with open(processing_in, "r") as json_file:
    loaded_vectors = json.load(json_file)

# Umwandeln von Listen zurück in numpy-Arrays
target_processing_vectors = {
    key: np.array(value) for key, value in loaded_vectors.items()
}


# --- Anwenden auf die Daten ---
train_targets_rescaled = preprocessing(train_targets, target_processing_vectors)
test_targets_rescaled  = preprocessing(test_targets, target_processing_vectors)
val_targets_rescaled   = preprocessing(val_targets, target_processing_vectors)

# --- Check Shapes ---
print("Train Rescaled Shape:", train_targets_rescaled.shape)
print("Test Rescaled Shape:", test_targets_rescaled.shape)
print("Val Rescaled Shape:", val_targets_rescaled.shape)



# --- Preprocessing & Postprocessing ---
def preprocessing(features, processing_vectors):
    return (features - processing_vectors['mean']) / processing_vectors['sigma']

def postprocessing(features, processing_vectors):
    return features * processing_vectors['sigma'] + processing_vectors['mean']


# --- Lade gespeicherte Rescaling-Vektoren ---
processing_in = f"../Data/Rescaling/Processing_vectors_parameter_{dataset}_{powerspectrum_mode}.json"

with open(processing_in, "r") as json_file:
    loaded_vectors = json.load(json_file)

# Umwandeln zurück in numpy Arrays
feature_processing_vectors = {key: np.array(val) for key, val in loaded_vectors.items()}


# --- Anwenden auf die Daten ---
train_features_rescaled = preprocessing(training_features, feature_processing_vectors)
test_features_rescaled  = preprocessing(testing_features, feature_processing_vectors)
val_features_rescaled   = preprocessing(validation_features, feature_processing_vectors)


# --- Check Shapes ---
print("Train Rescaled Shape:", train_features_rescaled.shape)
print("Test Rescaled Shape:", test_features_rescaled.shape)
print("Val Rescaled Shape:", val_features_rescaled.shape)
# %% [markdown]
# ## NN training


# %%
#restructuring of the arrays
train_targets_rescaled      = np.transpose(train_targets_rescaled, (1, 0, 2))
val_targets_rescaled  = np.transpose(val_targets_rescaled, (1, 0, 2))
test_targets_rescaled = np.transpose(test_targets_rescaled, (1, 0, 2))


# %%
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
study.optimize(objective, n_trials=bayesian_trials)

print("Beste Parameter:")
print(study.best_params)

