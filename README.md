# RNN as models of human cognition on mental health and disorder


## Objective:
1. Analyze individual differences of longitudinal smartphone based data (multi-task cognitive behavioral battery) focusing on performance, reaction time and errors 
1. Train individual RNNs on datasets and find optimal hyperparameter sets (grid search)
2. Evaluate internal representations in terms of robustness, topological structure and representational similarity over training and data collection time
3. Investigate sensible environmental factors (task complexity, task quantity, model hyperparameters) on internal representations
6. Compare individuals within and between, derive mechanistic understanding of computations and highlight differences between cognitive profiles
7. Compare the internal representational metrics to brain embedded graphs (fMRI based)


## Hierarchy of actions
0. preprocessing & dataAugmentation.py for data preperation
1. training.py & hyperparameterTuning.py for training models
2. hyperparameterOverview.py & hyperparameterOverview_overlays.py for performance and topological marker visualization
3. singleNetworkAnalysis.py & multipleNetworkAnalysis.py - analyzing models hidden structure and task representation + saving representational network metrics
4. cognitiveErrorAnalysis.py & cognitiveTaskAccuracy.py - analyzing individual cognitive behavior


## Docker Flow
- Dockerfile defines dependencies
- docker-build.yaml builds the .dockerignore reduced image whenever repo is pushed to github
- latest image is pushed to docker hub (actions with secrets)
- from VM manually pull image from hub
- initalize container with fine-tuned hps defined by config files w. different shell scripts


