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
