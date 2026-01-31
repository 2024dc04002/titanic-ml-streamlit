\# Titanic Survival Prediction using Machine Learning



\## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a passenger survived the Titanic disaster based on demographic and travel information. The project also includes deployment of the models using a Streamlit web application.



---



\## b. Dataset Description

The dataset used is the Titanic dataset obtained from Kaggle. It contains information about passengers such as age, gender, ticket class, fare, and embarkation port.  

The target variable is \*\*Survived\*\*, where:

\- 0 = Did not survive  

\- 1 = Survived  



The dataset contains 891 instances and 12 original features. After preprocessing, 7 useful features were used for model training.



---



\## c. Models Used and Performance Comparison



The following machine learning models were implemented and evaluated:

\- Logistic Regression  

\- Decision Tree Classifier  

\- K-Nearest Neighbors (KNN)  

\- Naive Bayes (Gaussian)  

\- Random Forest (Ensemble)  

\- XGBoost (Ensemble)  



\### Performance Comparison Table



| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |

|-------|----------|-----|-----------|--------|----------|-----|

| Logistic Regression | 0.7989 | 0.8798 | 0.7714 | 0.7297 | 0.7500 | 0.5826 |

| Decision Tree | 0.7877 | 0.7831 | 0.7500 | 0.7297 | 0.7397 | 0.5607 |

| KNN | 0.7039 | 0.7644 | 0.6780 | 0.5405 | 0.6015 | 0.3767 |

| Naive Bayes | 0.7710 | 0.8601 | 0.7260 | 0.7162 | 0.7211 | 0.5268 |

| Random Forest | 0.8212 | 0.8918 | 0.8000 | 0.7568 | 0.7778 | 0.6291 |

| XGBoost | 0.7877 | 0.8677 | 0.7500 | 0.7297 | 0.7397 | 0.5607 |



---



\## d. Observations



| Model | Observation |

|-------|-------------|

| Logistic Regression | Performed well with balanced precision and recall and showed strong baseline performance. |

| Decision Tree | Produced reasonable accuracy but is prone to overfitting compared to ensemble models. |

| KNN | Gave lower performance due to sensitivity to feature scaling and choice of K value. |

| Naive Bayes | Showed fast training and decent performance but assumes feature independence. |

| Random Forest | Achieved the best overall performance due to ensemble learning and reduced overfitting. |

| XGBoost | Performed competitively and handled complex relationships effectively. |



---



\## Streamlit Application

A Streamlit web application was developed that allows users to:

\- Select a machine learning model  

\- View classification report  

\- View confusion matrix  



---



\## How to Run



1\. Install required libraries:
pip install -r requirements.txt




2\. Run the Streamlit app:

streamlit run app.py

