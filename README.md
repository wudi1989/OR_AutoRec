# OR-AutoRec model
This is the implementions for the paper work of "OR-AutoRec: An Outlier-Resilient Autoencoder-based Recommendation model"
# Brief Introduction
This paper proposes an outlier-resilient autoencoder-based recommendation model, termed OR-AutoRec. Its main idea is to incorporate the Cauchy Loss into an autoencoder to measure the discrepancy between the observed user behaviour data and the predicted ones. As such, OR-AutoRec is resilient to outliers owing to the robustness of Cauchy Loss. By conducting extensive experiments on five benchmark datasets, we demonstrate that: 1) our OR-AutoRec is much more robust to outliers than original autoencoder-based model, and 2) our OR-AutoRec achieves significantly better prediction accuracy than both DNN-based and non-DNN-based state-of-the-art models.
# Enviroment Requirement
1. python = 3.6
2. tensorflow = 1.17.0

# Dataset
The MovieLens_100k (ml100k) dataset contains rating data for multiple movies from multiple users.
Important information:  
uses = 943  
items = 1682  
ratings = 100000  
density = 6.3%   
The main parameters of the dataset are provided below.

# Parameters setting
1. learning_rate = 0.001
2. lambda_value = 1
3. gamma_value = 4
4. epochs = 500
5. batch_size = 512
