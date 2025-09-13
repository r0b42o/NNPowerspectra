# Define dataset size
N_total = 50000
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

# %%
import pyccl as ccl
import g3lhalo
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# %%
#%pip install pyDOE

# %%
# checking that we are using a GPU
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
print('using', device, 'device \n')

# %%
# setting the seed for reproducibility
np.random.seed(9721)
tf.random.set_seed(9721)

# %% [markdown]
# ## Calculating dataset

# %%
num_params = 10

data_array = np.empty((N_total, num_params), dtype=object)


# Anzahl der Parameter (Dimensionen des LHS)
num_cosmo_params = len(param_intervals)

# Generiere Latin Hypercube Samples (LHS)
lhs_samples = lhs(num_cosmo_params, samples=N_total)

# Skaliere die LHS-Samples auf die jeweiligen Intervalle
cosmo_samples = []
start = os.times()
for sample in lhs_samples:
    cosmo = {
        key: param_intervals[key][0] + (param_intervals[key][1] - param_intervals[key][0]) * sample[i]
        for i, key in enumerate(param_intervals)
    }
    cosmo_samples.append(cosmo)
end = os.times()

user_time = end.user - start.user
system_time = end.system - start.system
total_cpu_time = user_time + system_time

print(f"CPU-Zeit: user = {user_time:.4f}s, system = {system_time:.4f}s, total = {total_cpu_time:.4f}s")



# %% [markdown]
# ## Saving related parameter

# %%
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_dir = "../Data/"
para_fn = f"{data_dir}/Parameter_{N_total}_{timestamp}.json"
# Speichern der Daten
with open(para_fn, "w") as json_file:
    json.dump(cosmo_samples, json_file, indent=4)



