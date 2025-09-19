
# %%
# Define dataset, mode, NN model
parameter = "10000_2025-06-10_23-11-15"
powerspectrum_mode = "ss"
timestamp = '2025-07-20_20-46-07' #define model
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

# %% [markdown]
# ## Read in related features

# %%
# Parameterdatei einlesen
data_dir = "../Data/"
para_fn = f"{data_dir}Parameter_{parameter}.json"

with open(para_fn, "r") as json_file:
    all_parameter_samples = json.load(json_file)

# Keys für Parameter
feature_keys = ['Om_c', 'Om_b', 'h', 'sigma_8', 'n_s',
                'alpha', 'sigma', 'Mth', 'Mprime', 'beta']

# Direkt in ein 2D-Array umwandeln: (Anzahl Samples) x (Anzahl Parameter)
testing_features = np.array([
    [sample[key] for key in feature_keys] 
    for sample in all_parameter_samples
])

N_total = len(testing_features)


# %%
log_indices = [7, 8]  # Ersetze mit deinen echten Indizes

testing_features[:, log_indices] = np.log10(testing_features[:, log_indices])

print("input min/max:", testing_features.min(), testing_features.max())

# %%
print('number of test features:', len(testing_features), '. Should be', N_total)


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
#load

model = keras.models.load_model(
    f'../NN_builds/NN_{timestamp}.keras',
    compile=False
)
model.compile(optimizer=RectifiedAdam(), loss='mse')

# %%
#predicted data
emulated_testing = np.array(model.predict(test_features_rescaled, batch_size=N_total))

pred_targets= emulated_testing.reshape(N_total, 3, 50).transpose(1, 0, 2)
postprocessed_pred_targets= postprocessing(pred_targets, target_processing_vectors)


# %%

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



