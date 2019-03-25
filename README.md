# Machine Learning from Andrew Ng

## About this course
Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI.

This course provides a broad introduction to machine learning, datamining, and statistical pattern recognition. Topics include:
- Supervised learning (parametric/non-parametric algorithms, support vector machines, kernels, neural networks).
- Unsupervised learning (clustering, dimensionality reduction, recommender systems, deep learning).
- Best practices in machine learning (bias/variance theory; innovation process in machine learning and AI). The course will also draw from numerous case studies and applications, so that you'll also learn how to apply learning algorithms to building smart robots (perception, control), text understanding (web search, anti-spam), computer vision, medical informatics, audio, database mining, and other areas.

## Tought by
Taught by:  Andrew Ng, CEO/Founder Landing AI; Co-founder, Coursera; Adjunct Professor, Stanford University; formerly Chief Scientist, Baidu and founding lead of Google Brain

# Notes

## Week 1

## Week 2

## Week 3

## Week 4

## Week 5

## Week 6 - Evaluating a learning algorithm
- Learning curves
- How to build learning algorithms
- Spam Classifier design
- Error Analysis
- Skewed classes (Precision and Recall, F1 score)
- Data for machine learning

## Week 7
- Support Vector machine SVM (aka Large Margin Classifier) - used for classification problems.
- SVM optimization for linear regression. Hard to use for logistic reg.
- SVM Parameters:
  - Choice of parameter C (1/lambda) - controls penalty for misclassified training examples.
    - Large C - Lower bias, high variance.  
    - Small C - Higher bias, low variance.
  - Choice of parameter sigma^2 - how fast similarity metric decreases.
    - Large sigma - features vary more smoothly. Higher bias, lower variance.  
    - Small sigma - features vary less smoothly. Lower bias, higher variance.  
  - Choice of kernel (similarity). No kernel = linear kernel.  
- Use SVM software package (liblinear, libsvm, ...) to solve for params theta  
- Kernels (SVM Kernel must satisfy Mercer's Theorem):
  - Linear kernel: n - large, m - small. Risk of overfitting.
  - Gaussian kernel: n - small, m - large. Requires feature scaling!  
  - Polynomial kernel (rare). Params: const + degree of polynomial.
  - Esoteric kernels: string/chi-square/histogram intersection
- What kernel to choose?  
  Choose whatever performs best on the cross-validation data.
- Multi-class classification?  
  You one-vs-all algo.
- Logistic regression vs SVMs?
  - If n is large relative to m (ex. n = 10.000, m = 10...1000) - use LR or SVM without kernel ("linear kernel")
  - If n is small, m is intermediate (ex. n = 1...1.000, m = 10...10.000) - use SVM with Gaussian kernel.
  - If n is small, m is large (ex. n = 1...1.000, m = 50.000+) - create/add more features, then use LR or SVM without kernel.
  - NOTES
    - LR ~= SVM without kernel.
    - Neural Networks works well, but maybe slower to train.

## Week 8
- Unsupervised Clusterization Algorithm: K-Means
  - K-Means: Init; Repeat: Cluster assignment, Move Centroids
  - How to pick initial params - choose random.
  - Number of random initializatino - 50-1000
  - How to choose K? "Elbow" method and manually.
- Dimensionality reduction: PCA
  - Data preprocessing: feature scaling / mean normalization
  - Compute covariance matrix Sigma (svd - singular value decomposition, eig)
    svd = eig if applied to covariance matrix
  - Compute "eighvectors"
  - Reconstruction
  - How to choose number of principal components? 85-99% variance is retained.
  - In reality data is reduced in 5-10 times
  - Do not use as overfitting prevention
  - Good for: data compression, learning speedup, step for visualization.

## Week 9
- Anomaly Detection Algorithm
  - Density Estimation
  - Multivariate Gaussian Distribution
- Collaborative Filtering for Rating Prediction / Recommendation System:
  - Algorithm:
    1. Initialize with random values
    2. Minimize using gradient descent
    3. Predict rating
  - Low Rank Matrix Factorization (Low-rank approximation)
    - X * Theta'


# Overview

## Type of machine learning problems:
- Classification
- Prediction
- Clusterization
- Anomaly Detection
  Examples:
  - Failing Servers on Network
  - Defective product
- Recommendation System
  Examples:
  - Rating prediction
  - Find related products
  



## Model Selection (h)
To pick model (d - degree of polynomial):
1. Measure hypotesis on cross-validation set.
2. Use Test set to test error

## Auto Lambda Selection
Define J train, cv, test
1. Choose range of lambdas [0.01...10.24]
2. Minimize J for each lambda, find thetha_c
3. Use cross-validation set in Jcv calculation
4. Pick Jcv with the lowest value
5. Calculate Test error, Jtest (thetha_c)

## Ways to improve Learning Algorithm
- Get more training examples  (fixes high variance)
- Try smaller sets of features  (fixes high variance)
- Try getting additional features (fixes high bias)
- Try adding polynomial features  (fixes high bias)
- Try decreasing lambda (fixes high bias)
- Try increasing lambda  (fixes high variance)

## Neural networks and underfitting/overfitting
Small network - underfitting
Large network - overfitting (use larger lambda to prevent)

## How to debug Learning Algorithm
- Learning Curve

## Excercies
- Build numbers recognition systems
- Spam classifier using SVM

## Links
* Course on [Coursera](https://www.coursera.org/learn/machine-learning/)
