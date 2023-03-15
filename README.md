
# Pseudo-Labels Regularization for Unbiased Classification: A Strong, Simple and Competitive Baseline

# Introduction
This repository contains official PyTorch ClusterFix implementation. Our proposal is an effective unsupervised debiasing framework which uses a cluster-membership loss-based re-weighting strategy to mitigate bias signals.

# Overview
In the first step, feature embeddings are extracted through a self-supervised or pretrained CNN to obtain pseudo labels. More in details: 
- i) We used a pretrained network for features extraction
- ii) We applied k-means to obtain pseudo-labels

In a second step, the pre-trained model is fine-tuned using a multi-head approach: one head is used to calculate the loss on the target (true) classes while the others on the pseudo-labels:
- i) We used both target and pseudo label cross-entropy losses in the optimization process
- ii) The importance weight of each cluster is computed by the average of the both target and pseudo losses values of the elements it contains. The cluster weight then is updated at each iteration by averaging the previous step value with a momentum.
- iii) For each sample, we modulate the target-loss value with the weight of the cluster to which the sample belongs

# Setup
- Clone repository
```
git clone https://github.com/ICCV11624/ICCV11624.git
```
- Install Dependecies
```
pip install -r requirements.txt
```
# Get Data
- Download Waterbirds from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) in $root/data/
- Download CelebA from [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) in $root/data/

# Usage
We provide two simple scripts to replicate our experiments. Default configurations for the ClusterFix experiments are in the configurations/ directory

### Set configuration files
- set dataset folder "data_root" (es. "$root/data/" for CelebA or "$root/data/waterbird_complete95_forest2water2/ for Waterbirds"
- set "checkpoint_root" for logging (es. "$root/experiments")
- available targets "target_attr" for CelebA [Double_Chin, Wearing_Necklace, Chubby, Pale_Skin, Receding_Hairline, Wearing_Hat, Bangs Big_Lips, Brown_Hair, Wavy_Hair] 
- available targets "target_attr" for Waterbirds [Object, Place]

### Evaluation
- We provide models weights for all targets in the (CelebA, Waterbirds)
- to evaluate our method you can run the eval.py script and change the configuration file in eval.py
```
eval.py ".configutations/eval_cfix_waterbirds"
```
### Training 
- We provide models weights of the pre-trained networks used for clustering
- You can launch the run.sh script 

```
run.py ".configutations/run_cfix_waterbirds"
```
