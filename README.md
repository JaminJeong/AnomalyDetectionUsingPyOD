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

## Reference 
* data : https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
* pyod : https://github.com/yzhao062/pyod
