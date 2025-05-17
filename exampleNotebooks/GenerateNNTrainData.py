# %% [markdown]
# ## This Notebook calculates the powerspektrum for various diffrent cosmological starting parameters

# %%
import socket
socket.gethostname()

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
N_total = 50000
num_params = 10

data_array = np.empty((N_total, num_params), dtype=object)

Mth_min = 1e10
Mth_max = 1e15
# Definiere die Intervalle für die kosmologischen Parameter
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

# Anzahl der Parameter (Dimensionen des LHS)
num_cosmo_params = len(param_intervals)

# Generiere Latin Hypercube Samples (LHS)
lhs_samples = lhs(num_cosmo_params, samples=N_total)

# Skaliere die LHS-Samples auf die jeweiligen Intervalle
cosmo_samples = []
for sample in lhs_samples:
    cosmo = {
        key: param_intervals[key][0] + (param_intervals[key][1] - param_intervals[key][0]) * sample[i]
        for i, key in enumerate(param_intervals)
    }
    cosmo_samples.append(cosmo)

# Halo mass function
hmf = ccl.halos.MassFuncSheth99()

# Halo bias
hbf = ccl.halos.HaloBiasSheth01()

# Concentration mass Relation
cmfunc = ccl.halos.ConcentrationDuffy08()

# Korrelation zwischen Galaxienpopulationen
A = 0  # 0 ==> Keine Korrelation zwischen Populationen
epsilon = 0

# Skalierung des Halo-Profils
flens = 1  # 1 ==> Gleiches Profil wie dunkle Materie

data_array = []

Pk_ss_1h = [1] * 50  # Platzhalter für die 50 Werte
Pk_ss_2h = [2] * 50
Pk_ss = [3] * 50
Pk_sl_1h = [4] * 50
Pk_sl_2h = [5] * 50
Pk_sl = [6] * 50
Pk_ll_1h = [7] * 50
Pk_ll_2h = [8] * 50
Pk_ll = [9] * 50
Pk_lin = [10] * 50

# Hauptschleife für Simulationen
for i in range(N_total):
    # Kosmologische Parameter aus LHS entnehmen
    cosmo = cosmo_samples[i]
    cosmo_array = np.array(list(cosmo.values()))

    # Nur die 5 kosmologischen Parameter für das Modell extrahieren
    cosmo_subset = {
        key: cosmo[key] for key in ['Om_c', 'Om_b', 'h', 'sigma_8', 'n_s']
    }

    # HOD-Parameter extrahieren
    alpha = cosmo['alpha']
    sigma = cosmo['sigma']
    Mth = cosmo['Mth']
    Mprime = cosmo['Mprime']
    beta = cosmo['beta']


    # HOD abrufen
    hod_cen, hod_sat = g3lhalo.HOD_Zheng(alpha, Mth, sigma, Mprime, beta)

    # Modell definieren
    model = g3lhalo.halomodel(verbose=True, cosmo=cosmo_subset, hmfunc=hmf, hbfunc=hbf, cmfunc=cmfunc)
    model.set_hods(hod_cen, hod_sat, A=A, epsilon=epsilon, flens1=flens, flens2=flens)

    # Berechnungen durchführen
    ks = np.geomspace(1e-2, 1e2)
    z = 0

    # Lineares Materie-Leistungsspektrum
    Pk_lin = model.pk_lin(ks, z)

    # Materie-Materie Leistungsspektrum
    Pk_ss_1h, Pk_ss_2h, Pk_ss = model.source_source_ps(ks, z)

    # Materie-Galaxie Leistungsspektrum
    Pk_sl_1h, Pk_sl_2h, Pk_sl = model.source_lens_ps(ks, z, type=1)

    # Galaxie-Galaxie Leistungsspektrum
    Pk_ll_1h, Pk_ll_2h, Pk_ll = model.lens_lens_ps(ks, z, type1=1, type2=1)
    

    # Erstelle das Dictionary mit den Daten für das aktuelle Sample
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
    
    # Füge das Dictionary für dieses Sample der Liste hinzu
    data_array.append(data_dict)

    print(f"Run {i+1}/{N_total} completed")


# %% [markdown]
# ## Saving generated data

# %%
# Funktion, die rekursiv alle np.ndarrays in Listen umwandelt
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Umwandlung in eine Python-Liste
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]  # Rekursive Umwandlung der Listenelemente
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}  # Rekursive Umwandlung der Dictionary-Werte
    else:
        return obj  # Für alle anderen Typen keine Änderung

# Konvertiere alle numpy.ndarrays in data_array
data_array_serializable = [convert_to_serializable(sublist) for sublist in data_array]

# Anzahl der generierten Daten
num_data_points = len(data_array_serializable)

# Verzeichnis für die gespeicherten Daten
data_dir = "../Data/"
os.makedirs(data_dir, exist_ok=True)

# Erzeuge einen einzigartigen Dateinamen mit Zeitstempel
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_fn = f"{data_dir}/GeneratedData_{num_data_points}_{timestamp}.json"

# Speichern der Daten
with open(data_fn, "w") as json_file:
    json.dump(data_array_serializable, json_file, indent=4)

print(f"Daten erfolgreich gespeichert in: {data_fn}")


# %% [markdown]
# ## Saving related parameter

# %%
para_fn = f"{data_dir}/Parameter_{num_data_points}_{timestamp}.json"
# Speichern der Daten
with open(para_fn, "w") as json_file:
    json.dump(cosmo_samples, json_file, indent=4)

print(f"Daten erfolgreich gespeichert in: {data_fn}")


