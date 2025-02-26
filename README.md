# Machine Learning Study Guidebook

This guidebook is a comprehensive reference covering key topics in machine learning—from foundational methods to advanced ensemble techniques. The content is organized into two parts:

- **Part 1: Foundations & Core Methods**
  - Linear Regression
  - Logistic Regression
  - Bias-Variance Trade-off
  - Key Performance Indicators (KPIs)
  - Validation & Cross-Validation

- **Part 2: Advanced Topics & Ensemble Methods**
  - Fighting High Variance (Overfitting)
  - K-Nearest Neighbors (KNN)
  - Bayesian Classifiers
  - Decision Trees & Random Forests

Each section includes detailed concepts, key equations, quiz highlights, essay prompts, and an expanded glossary with short, clear definitions. Additionally, relevant diagrams and plots are provided to illustrate key ideas.

---

## Table of Contents

- [Part 1: Foundations & Core Methods](#part-1-foundations--core-methods)
  - [1. Linear Regression](#linear-regression)
  - [2. Logistic Regression](#logistic-regression)
  - [3. Bias-Variance Trade-off](#bias-variance-trade-off)
  - [4. Key Performance Indicators (KPIs)](#ml-key-performance-indicators)
  - [5. Validation & Cross-Validation](#validation--cross-validation)
- [Part 2: Advanced Topics & Ensemble Methods](#part-2-advanced-topics--ensemble-methods)
  - [6. Fighting High Variance](#fighting-high-variance)
  - [7. K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [8. Bayesian Classifiers](#bayesian-classifiers)
  - [9. Decision Trees & Random Forests](#decision-trees--random-forests)
  
---

## Part 1: Foundations & Core Methods

### Linear Regression

**Overview:**  
Linear Regression is a supervised learning technique for modeling the relationship between a numeric target and one or more features by fitting a linear equation. For a single variable, the model is:

$$
\hat{y} = w_0 + w_1 x
$$

In the multivariable case, it is:

$$
\hat{y} = w_0 + w_1 x_1 + \dots + w_p x_p
$$

The method minimizes the **Mean Squared Error (MSE)** between predictions and true values, finding the best-fit line via least squares.

![Linear Regression Fit](https://upload.wikimedia.org/wikipedia/commons/e/ed/Residuals_for_Linear_Regression_Fit.png)  
*Figure: A scatterplot with a red best-fit line and residuals shown as vertical lines.*

**Key Concepts:**

- **Linear Hypothesis:** The assumption of a linear relationship between features and target.
- **Parameters/Weights:**  
  - **$$\(w_0\) (Intercept/Bias)$$:** The predicted value when all features are zero.  
  - **$$\(w_1, \dots, w_p\)$$:** The slopes showing how each feature affects the prediction.
- **Cost Function (MSE):**
    **$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \bigl(y_i-\hat{y}_i\bigr)^2$$**
- **Training vs. Test Error:** Training error is measured on the training data; generalization (test) error shows performance on unseen data.
- **Outliers:** Outliers can disproportionately affect MSE, skewing the model.

**Glossary:**

- **Dependent variable (Target):** The output \(y\) the model aims to predict.
- **Independent variable (Feature):** The input \(x\) used for prediction.
- **Coefficient (Weight):** Parameter \(w_i\) indicating the effect of a feature on \(y\).
- **Intercept (Bias):** Constant \(w_0\) representing the output when all features are zero.
- **Least Squares:** Method to fit the model by minimizing the sum of squared residuals.
- **Residual (Error):** The difference $$\(y - \hat{y}\)$$.
- **Underfitting:** When a model is too simple, leading to high bias.

[![Watch the video](https://img.youtube.com/vi/3g-e2aiRfbU/maxresdefault.jpg)](https://youtu.be/3g-e2aiRfbU)

### [An Intuitive Introduction to Linear Regression](https://youtu.be/3g-e2aiRfbU)
---

### Logistic Regression

**Overview:**  
Logistic Regression is used for binary classification. It models the probability of a positive class by applying the sigmoid function to a linear combination of features:

$$p(x) = \frac{1}{1 + e^{-(w_0 + w_1 x_1 + \dots + w_p x_p)}}$$

A probability above a chosen threshold (typically 0.5) indicates the positive class.

![Logistic Regression Curve](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Exam_pass_logistic_curve.svg/1280px-Exam_pass_logistic_curve.svg.png)  
*Figure: The sigmoid function mapping inputs to probabilities.*

**Key Concepts:**

- **Sigmoid Function:** Transforms any real value into a value between 0 and 1.
- **Binary Classification:** Predicts one of two classes.
- **Odds Ratio:** Exponentiating coefficients gives the factor change in odds for a unit change in a feature.
- **Maximum Likelihood Estimation:** Training method to optimize the model parameters.

**Glossary:**

- **Sigmoid/Logistic Function:**  
  $$\sigma(z)=\frac{1}{1+e^{-z}}$$  
  mapping linear outputs to probabilities.
- **Odds & Log-Odds:**  
  Odds are given by  
  $$\frac{p}{1-p}$$  
  and log-odds are the natural logarithm of the odds.
- **Binary Classification:** Predicting two possible outcomes.
- **Decision Threshold:** The cutoff (often 0.5) used to classify the probability output.
- **Odds Ratio:** $$\(e^{w_i}\)$$; quantifies the change in odds per unit increase in feature $$\(i\)$$.
- **Multinomial Logistic Regression:** Extends logistic regression to multi-class problems.


[![Watch the video](https://img.youtube.com/vi/EKm0spFxFG4/maxresdefault.jpg)](https://youtu.be/EKm0spFxFG4)

### [An Quick Intro to Logistic Regression](https://youtu.be/EKm0spFxFG4)


---

### Bias-Variance Trade-off

**Overview:**  
The Bias-Variance Trade-off describes the balance between the error from erroneous assumptions (bias) and the error from sensitivity to small data fluctuations (variance). An optimal model minimizes overall error by finding a balance between underfitting and overfitting.

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

![Bias–Variance Trade-off](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/1024px-Bias_and_variance_contributing_to_total_error.svg.png)  
*Figure: Illustration of bias and variance contributions to total error.*

**Glossary:**

- **Bias:** Error due to simplifying assumptions; high bias leads to underfitting.
- **Variance:** Error due to sensitivity to training data; high variance leads to overfitting.
- **Underfitting:** When a model is too simple, resulting in high bias.
- **Overfitting:** When a model is too complex, resulting in high variance.
- **Model Complexity:** Degree of flexibility in a model.
- **Regularization:** Methods to constrain a model, reducing variance.

---

### ML Key Performance Indicators

**Overview:**  
Key Performance Indicators (KPIs) are metrics used to evaluate classification and regression models. For classification, these include accuracy, precision, recall, F1 score, specificity, and ROC AUC. For regression, common KPIs are Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared.

**Classification Metrics:**

- **Confusion Matrix:**  

  |                      | Predicted Positive | Predicted Negative |
  |----------------------|--------------------|--------------------|
  | **Actual Positive**  | True Positive (TP) | False Negative (FN)|
  | **Actual Negative**  | False Positive (FP)| True Negative (TN) |

- **Accuracy:**  
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precision:**  
  $$\text{Precision} = \frac{TP}{TP+FP}$$

- **Recall (Sensitivity):**  
  $$\text{Recall} = \frac{TP}{TP+FN}$$

- **F1 Score:**  
  $$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Specificity:**  
  $$\text{Specificity} = \frac{TN}{TN+FP}$$

- **ROC AUC:** The area under the Receiver Operating Characteristic curve, summarizing the trade-off between sensitivity and specificity.

**Regression Metrics:**

- **Mean Squared Error (MSE):**  
  $$\text{MSE} = \frac{1}{n}\sum (y - \hat{y})^2$$

- **Root Mean Squared Error (RMSE):**  
  $$\text{RMSE} = \sqrt{\text{MSE}}$$

- **Mean Absolute Error (MAE):**  
  $$\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|$$

- **R-squared:** Proportion of variance in the target explained by the model.

**Glossary:**

- **True Positive/Negative (TP/TN):** Correctly predicted instances.
- **False Positive/Negative (FP/FN):** Misclassified instances.
- **Precision:** The proportion of positive predictions that are correct.
- **Recall:** The proportion of actual positives that are correctly predicted.
- **F1 Score:** Harmonic mean of precision and recall.
- **Specificity:** True negative rate.
- **ROC AUC:** A summary measure of a classifier’s ability to distinguish between classes.
- **MSE, RMSE, MAE:** Different ways to quantify error in regression.
- **R-squared:** The fraction of variance in the target explained by the model.

---

### Validation & Cross-Validation

**Overview:**  
Validation is critical to ensure a model’s generalizability. Data is typically divided into training, validation, and test sets. Cross-validation, such as k-fold cross-validation, systematically rotates the validation set to produce a robust estimate of model performance without overfitting to a single split.

![K-Fold Cross-Validation](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/K-fold_cross_validation_EN.svg/1024px-K-fold_cross_validation_EN.svg.png)  
*Figure: Illustration of k-fold cross-validation where the dataset is partitioned into k subsets.*

**Glossary:**

- **Training Set:** Data used to learn model parameters.
- **Validation Set:** Data used for tuning hyperparameters and model selection.
- **Test Set:** A final hold-out set used only for final performance evaluation.
- **Hold-Out Validation:** A single split of data into training and test sets.
- **k-Fold Cross-Validation:** The data is divided into k parts and the model is trained and validated k times.
- **Hyperparameter Tuning:** Adjusting model settings (e.g., learning rate, tree depth) based on validation performance.
- **Overfitting/Underfitting:** When validation performance indicates that the model is either too complex or too simple relative to the data.

[![Watch the video](https://img.youtube.com/vi/fSytzGwwBVw/maxresdefault.jpg)](https://youtu.be/fSytzGwwBVw)

### [Cross Validation](https://youtu.be/fSytzGwwBVw)

---

## Part 2: Advanced Topics & Ensemble Methods

### Fighting High Variance

**Overview:**  
High variance (overfitting) occurs when a model fits training data too closely, capturing noise. Strategies to fight high variance include regularization, ensemble methods, early stopping, feature selection, and increasing training data.

**Techniques:**

- **Regularization:**  
  - **L2 (Ridge):** Adds $$\lambda \sum w_i^2$$ to the loss, encouraging small weights.
  - **L1 (Lasso):** Adds $$\lambda \sum |w_i|$$, promoting sparsity by zeroing out some weights.
- **Ensemble Methods:**  
  - **Bagging:** Trains multiple models on bootstrap samples and averages their predictions.
- **Early Stopping:** Monitors validation error during training to stop before overfitting.
- **Feature Selection/Dimensionality Reduction:** Reduces the number of input variables to lower model complexity.
- **Increase Training Data:** More data typically reduces variance.

**Glossary:**

- **Regularization:** Methods to prevent overfitting by penalizing complexity.
- **Lasso (L1) and Ridge (L2):** Types of regularization that shrink coefficients; Lasso can set some coefficients to zero.
- **Bagging:** Ensemble method using bootstrap samples to train multiple models.
- **Early Stopping:** Halting training based on validation error trends.
- **Bias–Variance Trade-off:** Balancing model complexity to avoid too much bias (underfitting) or variance (overfitting).



[![Watch the video](https://img.youtube.com/vi/aBgMRXSqd04/maxresdefault.jpg)](https://youtu.be/aBgMRXSqd04)

### [L1 Vs L2 Regularzation Methods](https://youtu.be/aBgMRXSqd04)

---
[![Watch the video](https://img.youtube.com/vi/tjy0yL1rRRU/maxresdefault.jpg)](https://youtu.be/tjy0yL1rRRU)

### [Bagging vs Boosting](https://youtu.be/tjy0yL1rRRU)

---

### K-Nearest Neighbors (KNN)

**Overview:**  
KNN is an instance-based learning algorithm used for both classification (by majority voting) and regression (by averaging). It predicts new examples by finding the k closest training instances according to a chosen distance metric.

**Key Points:**

- **Distance Metrics:**  
  - **Euclidean Distance:** Straight-line distance.
  - **Manhattan Distance:** Sum of absolute differences.
  - **Cosine Similarity:** Measures the cosine of the angle between two vectors.
- **Parameter \(k\):** Choosing \(k\) balances bias and variance.
- **Normalization:** Essential to scale features for distance calculations.
- **Curse of Dimensionality:** In high dimensions, distances become less meaningful.

**Glossary:**

- **Instance-Based Learning (Lazy Learning):** Methods that defer generalization until a query is made.
- **Distance Metric:** Function to measure similarity (e.g., Euclidean, Manhattan).
- **\(k\) (Number of Neighbors):** The count of nearest data points used for predictions.
- **Weighted KNN:** A variant where nearer neighbors have more influence.
- **Curse of Dimensionality:** Phenomenon where high-dimensional spaces dilute the significance of distance.

[![Watch the video](https://img.youtube.com/vi/b6uHw7QW_n4/maxresdefault.jpg)](https://youtu.be/b6uHw7QW_n4)

### [K nearest Neighbors](https://youtu.be/b6uHw7QW_n4)
---

### Bayesian Classifiers

**Overview:**  
Bayesian classifiers use Bayes’ Theorem to compute the probability of each class given the features. The most common form is the Naive Bayes classifier, which assumes features are conditionally independent given the class.

$$
P(C \mid X) \propto P(C) \prod_{i=1}^n P(X_i \mid C)
$$

**Key Points:**

- **Naive Bayes:** Fast and effective for many applications despite its simplifying assumptions.
- **Smoothing (Laplace Smoothing):** Prevents zero probabilities when a feature value has not been seen in training.
- **Application in Text Classification:** Often uses a "bag of words" representation.

**Glossary:**

- **Bayes’ Theorem:** \(P(A|B)=\frac{P(B|A)P(A)}{P(B)}\); fundamental to updating probabilities.
- **Prior Probability:** \(P(C)\), the initial probability of a class.
- **Likelihood:** \(P(X|C)\), the probability of the data given the class.
- **Posterior Probability:** \(P(C|X)\), the probability of the class after observing data.
- **Naive Bayes:** A Bayesian classifier assuming feature independence.
- **Conditional Independence:** When the occurrence of one feature is independent of another given the class.
- **Laplace Smoothing:** Technique to prevent zero likelihood by adding a small constant to counts.


---

### Decision Trees & Random Forests

**Overview:**  
Decision Trees split data recursively based on feature tests to predict outcomes, producing a flowchart-like structure of decisions. Random Forests combine multiple trees trained on random subsets of data and features to improve stability and reduce variance.

**Key Points:**

- **Decision Trees:**  
  - Learn by recursively partitioning data using criteria like information gain or Gini impurity.
  - Are highly interpretable and can handle both categorical and numerical data.
- **Pruning:**  
  - Prevents overfitting by trimming unnecessary branches.
- **Random Forests:**  
  - Use bagging and random feature selection to generate a set of uncorrelated trees whose predictions are aggregated.
  - Often provide robust performance and feature importance measures.

![Decision Tree Example](https://github.com/user-attachments/assets/ffada2e4-eeac-4c18-945d-49abb7118930)
  
*Figure: A decision tree example for the Iris dataset.*

![Random Forest Illustration](https://github.com/user-attachments/assets/032b91d7-7bb0-4978-8973-e6edbe75c89c)

*Figure: Illustration showing multiple decision trees in a Random Forest with random feature selection.*

**Glossary:**

- **Decision Tree:** A model that uses a series of questions (splits) to reach a decision.
- **Node / Leaf / Branch:** Components of a tree; nodes are decision points, leaves are final predictions, and branches are paths connecting nodes.
- **Impurity Measures:** Metrics such as Gini impurity or entropy used to determine the quality of a split.
- **Information Gain:** Reduction in entropy achieved by a split.
- **Pruning:** The process of reducing tree size to avoid overfitting.
- **Bagging (Bootstrap Aggregating):** Ensemble method that trains multiple models on randomly sampled subsets.
- **Random Forest:** An ensemble of decision trees with randomized feature selection and bagging to reduce variance.
- **Feature Importance:** A measure of a feature's contribution to model predictions, aggregated across trees.
- **Out-of-Bag (OOB) Error:** An internal error estimate in Random Forests using data not sampled for each tree.

---

## Final Notes

This README.md file contains a complete, detailed study guide for machine learning. It includes theoretical foundations, practical considerations, quiz/essay question highlights, and comprehensive glossaries for each section. Visual aids such as diagrams and flowcharts help illustrate complex concepts.

Feel free to customize further (e.g., add additional images or links to resources) as needed for your studies.

Happy learning!



#About Project itself
```
# CS-188.-Introduction-to-Artificial-Intelligence-Assignment5
This project predicts soccer player market values using machine learning. Data from Transfermarkt includes performance metrics and market values. A Perceptron algorithm classifies players based on these features. The model, trained with Stochastic Gradient Descent, aims to enhance prediction accuracy and address class imbalances.

Soccer Player Market Value Prediction
Project Overview
This project aims to predict the market value of soccer players using machine learning techniques. The project uses a dataset of soccer players' performance metrics and applies a Perceptron algorithm for classification.

Dataset
The dataset used for this project was obtained from Transfermarkt, a free global website tracking soccer players' data. It includes:

Player market values
Player appearances
Player performance metrics (goals, assists, minutes played, yellow cards, red cards)
Data Preprocessing
The data was cleaned and preprocessed to handle missing values and duplicates.
Numeric values such as goals, assists, and minutes played were aggregated and averaged.
Market values were converted to binary classes for initial classification and later to multi-class labels for a more detailed classification.
Model Selection
The Perceptron algorithm was chosen for its simplicity and effectiveness in handling linear separable data. It was trained using Stochastic Gradient Descent (SGD) due to the large size of the dataset.

Model Training
The dataset was split into training and testing sets, ensuring a balanced distribution of market values.
The model was trained using the fit function from the sklearn library.
Evaluation metrics such as Accuracy, Precision, Recall, and ROC AUC scores were used to assess the model's performance.
Model Evaluation
The model achieved high accuracy but lower recall, indicating a good precision but a need for improvement in identifying true positives.
ROC AUC scores suggested the model's moderate ability to distinguish between classes.
Future Improvements
Improve feature selection to capture better correlations.
Address class imbalance using advanced techniques.
Implement more sophisticated models for better performance.
Consider using metrics like F1 score for balanced evaluation.
Code and Resources
The complete Jupyter notebook with the code can be found here.

Author
Shoval Benzer
