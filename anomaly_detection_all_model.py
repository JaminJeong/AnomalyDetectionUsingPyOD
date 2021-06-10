from __future__ import division
from __future__ import print_function

import os
import sys
import pprint 

## temporary solution for relative imports in case pyod is not installed
## if pyod is installed, no need to use the following line
#sys.path.append(
#    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sos import SOS
from pyod.models.lscp import LSCP
from pyod.models.cof import COF
from pyod.models.sod import SOD

from pyod.models.vae import VAE
from pyod.models.auto_encoder import AutoEncoder

from sklearn.model_selection import train_test_split

from save_graph import GenGraph, BasicGenGraph
import pandas as pd
from pathlib import Path
import argparse

def load_csvfiles(fpath):
    try:
        if not Path(fpath).exists():
            print(f"file path is not available : {fpath}")
        df = pd.read_csv(fpath, delimiter=',')
    except Exception as e:
        print('fail to load file {} caused by {}'.format(fpath, e))
    finally:
        pass

    return df

def make_data_sampling(data, window_size):
    n_samples = len(data)
    if n_samples < window_size:
        print("window size is too long!!")
        return None

    return [data[i:i + window_size] \
        for i in range(n_samples - window_size + 1)]


parser = argparse.ArgumentParser(prog="AnomalyDetectionForMIT-BIHUsingPyod",
                                    description="AnomalyDetectionForMIT-BIHUsingPyod", add_help=True)
parser.add_argument('-w', '--WINDOW_SIZE', help='data window size.')
parser.add_argument('-c', '--CONTAMINATION', help='contamination.')
args = parser.parse_args()
 

# TODO: add neural networks, LOCI, SOS, COF, SOD

# Define the number of inliers and outliers
window_size = int(args.WINDOW_SIZE)
contamination = float(args.CONTAMINATION)
outliers_fraction = contamination # 0.25
clusters_separation = [0]

# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                 LOF(n_neighbors=50)]

df_data = load_csvfiles("./data/daily-min-temperatures.csv")
pprint.pprint(f"df_data.head() : {df_data.head()}")
df_data_train, df_data_test = train_test_split(np.array(df_data["Temp"]), test_size=0.2, shuffle=False)

graph_path = Path('./graph')
graph_path.mkdir(exist_ok=True, parents=True)

GenGraph('train').draw(Path('./graph'), df_data_train[::30])
GenGraph('test').draw(Path('./graph'), df_data_test[::30])

#df_data_train = np.array(list(map(lambda x : float(x), df_data_train)))
# sys.exit(0)

random_state = 42
# Define nine outlier detection tools to be compared
classifiers = {
    'Angle-based Outlier Detector (ABOD)':
        ABOD(contamination=outliers_fraction),
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state, n_clusters=15),
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=35),
                       contamination=outliers_fraction,
                       random_state=random_state),
    'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction,
                                random_state=random_state),
    'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
    'Average KNN': KNN(method='mean',
                       contamination=outliers_fraction),
     'Median KNN': KNN(method='median',
                       contamination=outliers_fraction),
    'Local Outlier Factor (LOF)':
        LOF(n_neighbors=35, contamination=outliers_fraction),
     'Local Correlation Integral (LOCI)':
         LOCI(contamination=outliers_fraction),
    'Minimum Covariance Determinant (MCD)': MCD(
        contamination=outliers_fraction, random_state=random_state),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    'Principal Component Analysis (PCA)': PCA(
       contamination=outliers_fraction, random_state=random_state),
    'Stochastic Outlier Selection (SOS)': SOS(
        contamination=outliers_fraction),
    'Locally Selective Combination (LSCP)': LSCP(
        detector_list, contamination=outliers_fraction,
        random_state=random_state),
    'Connectivity-Based Outlier Factor (COF)':
        COF(n_neighbors=35, contamination=outliers_fraction),
    'Subspace Outlier Detection (SOD)':
        SOD(contamination=outliers_fraction),

    'Variational Auto Encoder':
        VAE(epochs=30, contamination=contamination, gamma=0.8, capacity=0.2, batch_size = window_size),
    'Auto Encoder':
        AutoEncoder(epochs=30, contamination=contamination, batch_size = window_size),
}

# Show all detectors
for i, clf in enumerate(classifiers.keys()):
    print('Model', i + 1, clf)

# fit the data and tag outliers
df_data_train = np.array(make_data_sampling(df_data_train, window_size))
pprint.pprint(f"df_data_train.shape : {df_data_train.shape}")
df_data_test = np.array(make_data_sampling(df_data_test, window_size))
pprint.pprint(f"df_data_test.shape : {df_data_test.shape}")

# Fit the models with the generated data and
# compare model performances
for i, (clf_name, clf) in enumerate(classifiers.items()):
    title_name = f"{clf_name}"
    print()
    print(i + 1, 'fitting : ', title_name)
    try:
        clf.fit(df_data_train)
        y_pred = clf.predict(df_data_test)
        data = {}
        #pprint.pprint(f"df_data_test : {df_data_test}")
        data['real_value'] = df_data_test[:, -1]
        data['anomaly_label'] = y_pred
        BasicGenGraph(title_name).draw(graph_path, data)
    except Exception as err:
        #print(err)
        pass
