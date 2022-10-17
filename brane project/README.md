# Implementing Prediction of Survivors in Titanic Disaster on Brane

[Brane](https://wiki.enablingpersonalizedinterventions.nl/user-guide/welcome.html), a programmable application orchestration framework, manages and con-
trols the utilization of multiple k8s clusters. In this project, we participated in a
Kaggle competition2 to predict the survivors of Titanic disaster based on the fea-
tures of passengers through using various classifier. We first implemented EDA
to analyze the importance of features and completed pre-processing and nor-
malization. After training with processed data, the performance of prediction is
evaluated by calculating the value of precision, recall and F1-score. With the aim
of testing whether Brane manages and executes the pipeline of machine learning
or not, the whole process is deployed and ran on a remote k8s cluster, which
are controlled by Brane (i.e., meet the requirements of Bonus part). At last,
the implementation of Github Actions also verified the feasibility of automatic
building the Brane’s dependencies and testing the performance of execution.
