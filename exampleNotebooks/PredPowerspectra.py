# %% [markdown]
# ## Notebook to analyse prediction performance of NN

# %%
# Define dataset, mode, NN model
dataset = "10000_2025-06-10_23-11-15"
powerspectrum_mode = "ll"
timestamp = '2025-08-27_20-13-31' #define model
processing_vector = "1000000_2025-06-11_05-16-31" #processing vector connected to model


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
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_addons as tfa
import optuna
from tensorflow.keras.layers import LeakyReLU, ELU

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

# %% [markdown]
# ## Read in generated Targets

# %%
np.random.seed(9721)
tf.random.set_seed(9721)

# %%
# 1. Speicherort und Dateiname

data_dir = "../Data/"
data_fn = f"{data_dir}GeneratedData_{dataset}.json"

# 2. Relevante Keys für Galaxy-Matter

target_keys = [f'Pk_{powerspectrum_mode}_1h', f'Pk_{powerspectrum_mode}_2h', f'Pk_{powerspectrum_mode}']
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

# Nach clean_samples erstellen
num_samples = len(clean_samples)
indices = np.arange(num_samples)
N_total = len(clean_samples)

N_test = N_total

print(f"Test: {num_samples}")

# 8. Hilfsfunktion zum Splitten
def split_data(array_dict, split_indices):
    return np.array([data[split_indices] for data in array_dict.values()])

# 9. Targets und Modes für Splits
testing_targets = {
    'modes': split_data(targets_array, indices),
    'targets': split_data(log_transformed_data, indices)
}

# %%
print('number of test targets:', len(testing_targets['modes'][:][0]), '. Should be', N_test)

# %%
N_modes=len(testing_targets['modes'])
print(f"Shape of testing targets: {testing_targets['targets'].shape}. Should be ({N_modes}, {N_test}, 50)")

# %% [markdown]
# ## Read in related features

# %%
# Parameterdatei einlesen
para_fn = f"{data_dir}Parameter_{dataset}.json"

with open(para_fn, "r") as json_file:
    all_parameter_samples = json.load(json_file)

# Parameter entsprechend kept_indices filtern (kept_indices kommt vom Target-Filter)
filtered_parameter_samples = [all_parameter_samples[i] for i in kept_indices]

# Keys für Parameter
feature_keys = ['Om_c', 'Om_b', 'h', 'sigma_8', 'n_s', 'alpha', 'sigma', 'Mth', 'Mprime', 'beta']

# Gefilterte Parameter in NumPy-Arrays umwandeln
feature_array = {
    key: np.array([sample[key] for sample in filtered_parameter_samples])
    for key in feature_keys
}

# Splitten mit denselben Indizes wie bei Targets
testing_features = np.array([feature_array[key][indices] for key in feature_keys]).T

# %%
log_indices = [7, 8]  # Ersetze mit deinen echten Indizes

testing_features[:, log_indices] = np.log10(testing_features[:, log_indices])

print("input min/max:", testing_features.min(), testing_features.max())

# %%
print('number of test features:', len(testing_features), '. Should be', N_test)

# %% [markdown]
# ## Renaming

# %%
# features
test_targets=testing_targets['targets']

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
processing_in = f"../Data/Rescaling/Processing_vectors_data_{processing_vector}_{powerspectrum_mode}.json"

with open(processing_in, "r") as json_file:
    loaded_vectors = json.load(json_file)

# Umwandeln von Listen zurück in numpy-Arrays
target_processing_vectors = {
    key: np.array(value) for key, value in loaded_vectors.items()
}

# --- Anwenden auf die Daten ---
test_targets_rescaled  = preprocessing(test_targets, target_processing_vectors)

# --- Check Shapes ---
print("Test Rescaled Shape:", test_targets_rescaled.shape)

# %%
# --- Preprocessing & Postprocessing ---
def preprocessing(features, processing_vectors):
    return (features - processing_vectors['mean']) / processing_vectors['sigma']

def postprocessing(features, processing_vectors):
    return features * processing_vectors['sigma'] + processing_vectors['mean']


# --- Lade gespeicherte Rescaling-Vektoren ---
processing_in = f"../Data/Rescaling/Processing_vectors_parameter_{processing_vector}_{powerspectrum_mode}.json"

with open(processing_in, "r") as json_file:
    loaded_vectors = json.load(json_file)

# Umwandeln zurück in numpy Arrays
feature_processing_vectors = {key: np.array(val) for key, val in loaded_vectors.items()}


# --- Anwenden auf die Daten ---
test_features_rescaled  = preprocessing(testing_features, feature_processing_vectors)


# --- Check Shapes ---
print("Test Rescaled Shape:", test_features_rescaled.shape)


# %%
#restructuring of the arrays
test_targets_rescaled = np.transpose(test_targets_rescaled, (1, 0, 2))


# %%
#load

model = keras.models.load_model(
    f'../NN_builds/NN_{timestamp}.keras',
    compile=False
)
model.compile(optimizer=RectifiedAdam(), loss='mse')

# %%
#predicted data
emulated_testing = np.array(model.predict(test_features_rescaled, batch_size=N_total))

pred_targets= emulated_testing.reshape(N_test, 3, 50).transpose(1, 0, 2)
postprocessed_pred_targets= postprocessing(pred_targets, target_processing_vectors)


# %%
import numpy as np
import os
import json
from datetime import datetime

# Die passenden Keys auswählen
keys_map = {
    "ss": ["Pk_ss_1h", "Pk_ss_2h", "Pk_ss"],
    "sl": ["Pk_sl_1h", "Pk_sl_2h", "Pk_sl"],
    "ll": ["Pk_ll_1h", "Pk_ll_2h", "Pk_ll"],
}

keys = keys_map[powerspectrum_mode]

# Dimensionen prüfen
num_components, N_total, block_size = postprocessed_pred_targets.shape
assert num_components == 3, "Es werden genau 3 Komponenten (1h, 2h, total) erwartet"
assert block_size == 50, "Jedes Spektrum sollte 50 Werte enthalten"

data_array = []
for i in range(N_total):
    # für dieses Sample die drei Spektren extrahieren
    blocks = [np.exp(postprocessed_pred_targets[j, i, :]) for j in range(3)]
    data_dict = {key: block.tolist() for key, block in zip(keys, blocks)}
    data_array.append(data_dict)

# JSON speichern
os.makedirs(data_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
predictedfile = f"{data_dir}PredictedData_{powerspectrum_mode}_{N_total}_{timestamp}.json"

with open(predictedfile, "w") as f:
    json.dump(data_array, f, indent=4)

print(f"NN-Powerspektren gespeichert in: {predictedfile}")



