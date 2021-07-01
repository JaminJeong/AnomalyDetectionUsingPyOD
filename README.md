# AnomalyDetectionUsingPyOD

## Install 
```bash
pip install pyod
```
* https://github.com/yzhao062/pyod#installation

## How To Run 
```bash
(pyod) $ python anomaly_detection_all_model.py -w 30 -c 0.25
```

## File Tree
```bash 
$ tree
.
├── README.md
├── anomaly_detection_all_model.py
├── data
│   └── daily-min-temperatures.csv
├── graph
│   ├── Angle-based Outlier Detector (ABOD).png
│   ├── Average KNN.png
│   ├── Cluster-based Local Outlier Factor (CBLOF).png
│   ├── Feature Bagging.png
│   ├── Histogram-base Outlier Detection (HBOS).png
│   ├── Isolation Forest.png
│   ├── K Nearest Neighbors (KNN).png
│   ├── Local Outlier Factor (LOF).png
│   ├── Median KNN.png
│   ├── test.png
│   └── train.png
└── save_graph.py
```

## Result

![Angle-based Outlier Detector (ABOD)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Angle-based%20Outlier%20Detector%20(ABOD).png)
![Average KNN](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Average%20KNN.png)
![Cluster-based Local Outlier Factor (CBLOF)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Cluster-based%20Local%20Outlier%20Factor%20(CBLOF).png)
![Connectivity-Based Outlier Factor (COF)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Connectivity-Based%20Outlier%20Factor%20(COF).png)
![Feature Bagging](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Feature%20Bagging.png)
![Histogram-base Outlier Detection (HBOS)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Histogram-base%20Outlier%20Detection%20(HBOS).png)
![Isolation Forest](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Isolation%20Forest.png)
![K Nearest Neighbors (KNN)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/K%20Nearest%20Neighbors%20(KNN).png)
![Local Outlier Factor (LOF)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Local%20Outlier%20Factor%20(LOF).png)
![Locally Selective Combination (LSCP)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Locally%20Selective%20Combination%20(LSCP).png)
![Median KNN](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Median%20KNN.png)
![Minimum Covariance Determinant (MCD)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Minimum%20Covariance%20Determinant%20(MCD).png)
![One-class SVM (OCSVM)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/One-class%20SVM%20(OCSVM).png)
![Principal Component Analysis (PCA)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Principal%20Component%20Analysis%20(PCA).png)
![Stochastic Outlier Selection (SOS)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Stochastic%20Outlier%20Selection%20(SOS).png)
![Subspace Outlier Detection (SOD)](https://github.com/JaminJeong/AnomalyDetectionUsingPyOD/blob/main/graph/Subspace%20Outlier%20Detection%20(SOD).png)

## Reference 
* data : https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
* pyod : https://github.com/yzhao062/pyod
* https://machinelearningmastery.com/time-series-datasets-for-machine-learning/
