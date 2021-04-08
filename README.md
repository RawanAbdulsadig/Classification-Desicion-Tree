# Classification-Desicion-Tree
A classification decision tree model implemented from scratch is described and its performance is statistically compared to python's sklearn library implementation "DesicionTreeClassifier". A project done at the end of the "Programming for Data Scientists" module (MSc. Data Science @ Lancaster University)

The comparison was done based on the performance of the models on the problem to classifying cancer as malignant or benign using the breast cancer wisconsin data-
set. It was found that both implementations produced almost
the same accuracy, F1-score, precision and recall as the minimum number of leaf samples required for a split
changed, while in terms of fitting and prediction time, the implemented model failed to beat sklearn's optimized
computational time. In contrast, the implemented model was able to be almost half the size of sklearn's model
in all times (in Bytes).
