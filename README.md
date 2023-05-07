# Churn Rate Prediction on Telco Dataset
This is a machine learning project that aims to predict the churn rate of a telecommunications company's customers.

## Dataset
The dataset used in this project is the Telco Customer Churn dataset from Kaggle, which contains information about customers who have either churned or stayed with the company. The dataset contains 21 features including customer demographics, account information, and services subscribed to.

## Approach
The project uses a supervised learning approach and various classification algorithms were tested including logistic regression, decision trees, random forests, and support vector machines. Hyperparameter tuning was performed on the best performing model using grid search.

## Evaluation Metrics
The model was evaluated using accuracy, precision, recall, F1 score, and ROC AUC. Since the dataset is imbalanced, with only about 26% of the customers churning, the models were also evaluated using the area under the precision-recall curve (PR AUC) to give more weight to the minority class.

## Results
The best performing model was a random forest classifier with an accuracy of 80%, F1 score of 0.58, and PR AUC of 0.57.

## Files
telco_churn.ipynb: Jupyter notebook containing the code for the project
telco_data.csv: dataset used in the project
README.md: this readme file
Dependencies
The project requires the following Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost.

## Acknowledgements
The dataset used in this project was obtained from Kaggle: https://www.kaggle.com/blastchar/telco-customer-churn

## References
Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems. O'Reilly Media, Inc.
Brownlee, J. (2020). How to Develop a Baseline Model for Telecommunications Churn. Machine Learning Mastery. https://machinelearningmastery.com/develop-baseline-model-performance-for-telecom-churn/
