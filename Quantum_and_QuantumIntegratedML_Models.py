#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('python --version')


# In[10]:


get_ipython().system('pip install scikit-learn==1.3.2 --force-reinstall --no-cache-dir --proxy 14.139.134.20:3128')
get_ipython().system('pip install xgboost==1.7.6 --force-reinstall --no-cache-dir --proxy 14.139.134.20:3128')
# !pip install lightgbm==4.0.0 --force-reinstall --no-cache-dir --proxy 14.139.134.20:3128
# !pip install catboost==1.2.2 --force-reinstall --no-cache-dir --proxy 14.139.134.20:3128


# In[11]:


get_ipython().system('pip install --upgrade --force-reinstall --no-cache-dir numpy==1.26.4 scipy==1.13.1 --proxy 14.139.134.20:3128')


# In[21]:


get_ipython().system('pip install matplotlib')


# In[23]:


get_ipython().system('pip install seaborn')


# In[12]:


import numpy as np
import pandas as pd

# Classical ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# External ML libraries
# import xgboost as xgb
# import lightgbm as lgb
#import catboost as cb


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- Load Data ---
df = pd.read_csv("QuantPreprocessed1999_20.csv")
#df = df.head(10000)
Field_list_sort=['Claims Count','Assignee Count','count_inventor','Count of Cited Refs - Patent','Count of Cited Refs - Non-patent','Count of Citing Patents','DWPI Count of Family Members','DWPI Count of Family Countries/Regions','IPC - Current Count','RelatedApplicationCount']
df['High Quality'] = 0  # Initialize all rows with 0
for i in Field_list_sort:
# Step 1: Sort the DataFrame based on 'Claims Count' in descending order
 df = df.sort_values(by=i, ascending=False).reset_index(drop=True)

# Step 2: Calculate the number of rows representing the top 5%
 top_5_percent_index = int(len(df) * 0.05)

# Step 3: Set 'High Quality' to 0 for all rows and 1 for the top 5%
 df.loc[:top_5_percent_index, 'High Quality'] = 1

#print(df)




# In[14]:


feature_names=['grant_lag_days','Claims Count','Assignee Count','count_inventor','Count of Cited Refs - Patent','Count of Cited Refs - Non-patent',
       'Count of Citing Patents','DWPI Count of Family Members','DWPI Count of Family Countries/Regions','RelatedApplicationCount',
        'IPC - Current Count','Sub Class Count','High Quality']#
training_features = df[feature_names]

#df=df[]

#Now delete all the columns for which all rows are having value null
df[feature_names] = df[feature_names].dropna(axis=1, how='all')
print('data shape=',df[feature_names].shape)
# Check if there is any more na
#(df.isnull().sum()*100)/len(df)


# In[15]:


df[feature_names].head()
df_num=df[feature_names]
X=df[['grant_lag_days','Claims Count','Assignee Count','count_inventor','Count of Cited Refs - Patent','Count of Cited Refs - Non-patent',
       'Count of Citing Patents','DWPI Count of Family Members','DWPI Count of Family Countries/Regions','RelatedApplicationCount',
        'IPC - Current Count','Sub Class Count']]#
y=df['High Quality']
# Train test split - 80% training data, 20% validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=100)
# Checking the shape after split
print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print('X_val Shape:', X_val.shape)
print('y_val Shape:', y_val.shape)


# In[16]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


results = {}

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
  #  "SVM": SVC(kernel="rbf", probability=True, random_state=42),
  # "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
  #  "XGBoost": XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="logloss"),
  #  "LightGBM": LGBMClassifier(n_estimators=200, random_state=42),
   # "CatBoost": CatBoostClassifier(iterations=200, verbose=0, random_state=42),
  #  "MLP": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred))


# In[17]:


#!rm -rf __pycache__


# In[18]:


# ===================== 6. Quantum ML (VQC) =====================
#!pip uninstall -y qiskit qiskit-terra qiskit-aer qiskit-ibmq-provider
# !pip install qiskit==1.1.0 qiskit-machine-learning==0.8.1 --force-reinstall --no-cache-dir --proxy 14.139.134.20:3128
# Install required libraries (Qiskit 1.x + ML module)
# !pip install qiskit qiskit-machine-learning  --force-reinstall --no-cache-dir --proxy 14.139.134.20:3128

# Imports
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

from qiskit_machine_learning.algorithms.classifiers import VQC
#from qiskit.utils import algorithm_globals
from qiskit_algorithms.utils import algorithm_globals
#!pip install -q "qiskit==1.1.0" "qiskit-machine-learning==0.8.1" --force-reinstall --no-cache-dir --proxy 14.139.134.20:3128
#!pip install -q qiskit-aer==0.13.0 --force-reinstall --no-cache-dir --proxy 14.139.134.20:3128# Needed for Sampler
get_ipython().system('#pip install -q qiskit-algorithms==0.3.0 --force-reinstall --no-cache-dir --proxy 14.139.134.20:3128')


# In[19]:


# ===================== Data Preprocessing =====================
# Reduce features to 4 dimensions (for 4-qubit circuit)
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_val_scaled)

# ===================== Quantum Circuit =====================
# Feature Map (encodes classical data into quantum state)
feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement="linear")

# Ansatz (variational circuit)
ansatz = TwoLocal(num_qubits=4, rotation_blocks="ry", entanglement_blocks="cz")

# Optimizer
optimizer = COBYLA(maxiter=50)

# ===================== Quantum Classifier =====================
algorithm_globals.random_seed = 42
sampler = Sampler()  # replaces Aer + QuantumInstance in Qiskit 1.x

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler,
)

# Ensure numpy arrays
X_train_pca = X_train_pca if isinstance(X_train_pca, np.ndarray) else X_train_pca.values
X_test_pca = X_test_pca if isinstance(X_test_pca, np.ndarray) else X_test_pca.values
y_train = y_train.values.ravel() if hasattr(y_train, "values") else y_train
y_test = y_val.values.ravel() if hasattr(y_val, "values") else y_val

print("Training Quantum VQC...")
vqc.fit(X_train_pca, y_train)

# Predictions
y_pred_q = vqc.predict(X_test_pca)
results["Quantum_VQC"] = accuracy_score(y_val, y_pred_q)

print("Quantum VQC Results:")
print(classification_report(y_val, y_pred_q))


# In[24]:


from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
def draw_roc(actual, probs):
    fpr, tpr, thresholds = roc_curve(actual, probs,drop_intermediate = False)
    auc_score = roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds


# In[25]:


fpr, tpr, thresholds = draw_roc(y_val, y_pred_q)


# In[26]:


# ===================== Quantum-Integrated AdaBoost =====================
# Installs (uncomment if you haven't installed Qiskit 1.x in this session)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Qiskit imports (1.x modern API)
from qiskit.primitives import Sampler
from qiskit_algorithms.utils import algorithm_globals
#from qiskit_machine_learning.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
#from qiskit.fidelity import ComputeUncompute # Corrected import location for ComputeUncompute

# ---- Parameters you can tweak ----
n_landmarks = min(100, max(10, X_train_scaled.shape[0]//5))  # number of landmarks for feature construction
feature_map_reps = 2
quantum_shots = 1024
ada_n_estimators = 50
random_state = 42
# ----------------------------------

# Ensure numpy arrays
X_train_np = X_train_scaled if isinstance(X_train_scaled, np.ndarray) else X_train_scaled.toarray() if hasattr(X_train_scaled, "toarray") else np.asarray(X_train_scaled)
X_test_np  = X_val_scaled  if isinstance(X_val_scaled,  np.ndarray) else X_val_scaled.toarray()  if hasattr(X_val_scaled, "toarray")  else np.asarray(X_val_scaled)
y_train_np = y_train.values.ravel() if hasattr(y_train, "values") else np.asarray(y_train).ravel()
y_test_np  = y_val.values.ravel()  if hasattr(y_val,  "values") else np.asarray(y_val).ravel() # Use y_val for testing

# 1) PCA -> 4D (quantum feature dimension)
pca = PCA(n_components=4, random_state=random_state)
X_train_pca = pca.fit_transform(X_train_np)
X_test_pca  = pca.transform(X_test_np)

# 2) Build quantum kernel
algorithm_globals.random_seed = random_state
sampler = Sampler(options={"shots": quantum_shots})

# Create the fidelity mechanism with a sampler
fidelity = ComputeUncompute(sampler=sampler)

# Construct the quantum kernel
feature_map = ZZFeatureMap(feature_dimension=4, reps=feature_map_reps, entanglement="linear")
qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)


# 3) Choose landmarks (representative training points)
rng = np.random.default_rng(random_state)
train_idx = np.arange(X_train_pca.shape[0])
rng.shuffle(train_idx)
landmark_idx = train_idx[:n_landmarks]
X_landmarks = X_train_pca[landmark_idx]

print(f"Using {n_landmarks} landmarks (indices sample) for quantum feature construction.")

# 4) Compute kernel features: K(X, landmarks)
print("Evaluating quantum kernel: train vs landmarks ...")
K_train_features = qkernel.evaluate(X_train_pca, X_landmarks)   # shape (n_train, n_landmarks)

print("Evaluating quantum kernel: test vs landmarks ...")
K_test_features  = qkernel.evaluate(X_test_pca, X_landmarks)    # shape (n_test, n_landmarks)

# 5) Train AdaBoost on quantum features (weak learners = decision stumps)
base_stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
ada = AdaBoostClassifier(estimator=base_stump, n_estimators=ada_n_estimators, random_state=random_state)
print("Training AdaBoost on quantum features...")
ada.fit(K_train_features, y_train_np)

# 6) Predict & evaluate
y_pred_qada = ada.predict(K_test_features)
acc_qada = accuracy_score(y_test_np, y_pred_qada)
print("Quantum-Integrated AdaBoost Accuracy:", acc_qada)
print(classification_report(y_test_np, y_pred_qada))

# Save to results dict
try:
    results["Quantum_AdaBoost"] = acc_qada
except NameError:
    results = {"Quantum_AdaBoost": acc_qada}


# In[27]:


fpr, tpr, thresholds = draw_roc(y_val, y_pred_q)


# 

# In[ ]:




