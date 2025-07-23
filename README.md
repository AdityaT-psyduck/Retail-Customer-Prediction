E-commerce Customer Value Prediction (RFM-NN)
=============================================

Table of Contents
-----------------

1.  [Project Overview](https://www.google.com/search?q=#project-overview)
    
2.  [Data Source](https://www.google.com/search?q=#data-source)
    
3.  [Methodology](https://www.google.com/search?q=#methodology)
    
    *   [Data Cleaning & Preprocessing](https://www.google.com/search?q=#data-cleaning--preprocessing)
        
    *   [RFM & Feature Engineering](https://www.google.com/search?q=#rfm--feature-engineering)
        
    *   [Target Variable Definition](https://www.google.com/search?q=#target-variable-definition)
        
    *   [Exploratory Data Analysis (EDA)](https://www.google.com/search?q=#exploratory-data-analysis-eda)
        
    *   [Model Training (Neural Network)](https://www.google.com/search?q=#model-training-neural-network)
        
    *   [Prediction & Evaluation](https://www.google.com/search?q=#prediction--evaluation)
        
4.  [Key Insights & Results](https://www.google.com/search?q=#key-insights--results)
    
5.  [How to Run the Code](https://www.google.com/search?q=#how-to-run-the-code)
    
6.  [Requirements](https://www.google.com/search?q=#requirements)
    
7.  [Future Work](https://www.google.com/search?q=#future-work)
    
8.  [License](https://www.google.com/search?q=#license)
    

1\. Project Overview
--------------------

This project aims to build a predictive model to identify **High-Value Customers** within an online retail dataset. By leveraging RFM (Recency, Frequency, Monetary) analysis and a Neural Network, the goal is to provide e-commerce businesses with actionable insights to target their most profitable customers effectively, optimize marketing strategies, and enhance customer retention efforts.

The project encompasses data cleaning, comprehensive feature engineering based on transactional data, exploratory data analysis to understand customer behavior, training a deep learning model, and robust evaluation of its performance.

2\. Data Source
---------------

The dataset used for this project is Online Retail.xlsx - Online Retail.csv, which contains transactional data from a UK-based online retail store.

*   **File Name:** Online Retail.xlsx (coverted to Online Retail.csv for the use)
    
*   **Description:** Contains all transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based online retail store.
    

3\. Methodology
---------------

The project follows a standard machine learning workflow:

### Data Cleaning & Preprocessing

*   Calculated TotalPrice for each transaction line.
    
*   Handled missing CustomerID values by removing relevant rows.
    
*   Removed return/cancellation entries (transactions with non-positive quantities).
    
*   Converted InvoiceDate to datetime objects.
    
*   One-Hot Encoding for categorical features (e.g., Country).
    
*   Feature Scaling using StandardScaler for all numerical features to prepare data for the Neural Network.
    

### RFM & Feature Engineering

Transactional data was aggregated to the customer level to derive essential RFM metrics and other behavioral features:

*   **Recency:** Days since the last purchase.
    
*   **Frequency:** Number of unique invoices/purchase days.
    
*   **Monetary:** Total spend and average order value.
    
*   **Additional Features:** Number of items purchased, average unit price, number of unique products, customer lifespan, purchase frequency per lifespan, and unique day frequency per lifespan.
    

### Target Variable Definition

A customer was defined as "High-Value" if their TotalSpend was above the **80th percentile** of all customers. This created a binary target variable (is\_high\_value: 1 for high-value, 0 otherwise). The original TotalSpend column was then dropped to prevent data leakage.

### Exploratory Data Analysis (EDA)

Visualizations were generated to understand the distribution of the target variable and key features, as well as their relationship with the "high-value" status:

*   Distribution of High-Value vs. Not High-Value customers.
    
*   Histograms of RFM features (Recency, Frequency, Monetary) by target class.
    
*   Box plots of RFM features by target class.
    
*   Pairplots of key RFM features.
    
*   Top 10 countries by the count of high-value customers.
    
*   Insights were extracted regarding class imbalance, and the typical characteristics of high-value customers (e.g., lower Recency, higher Frequency, higher Monetary values).
    

### Model Training (Neural Network)

A Sequential Neural Network model was built using TensorFlow/Keras:

*   **Architecture:** Multiple Dense layers with relu activation for hidden layers and a sigmoid activation for the output layer (binary classification).
    
*   **Optimizer:** Adam with a learning rate of 0.001.
    
*   **Loss Function:** binary\_crossentropy.
    
*   **Metrics:** Accuracy and AUC (Area Under the Receiver Operating Characteristic curve).
    
*   The model was trained for 10 epochs with a batch size of 64.
    

### Prediction & Evaluation

The trained model's performance was evaluated on the held-out test set:

*   **Classification Report:** Providing Precision, Recall, F1-score for both classes.
    
*   **Confusion Matrix:** Visualizing True Positives, False Positives, True Negatives, and False Negatives.
    
*   **ROC AUC Score & Curve:** Assessing the model's ability to distinguish between classes across various thresholds.
    
*   **Precision-Recall Curve & Average Precision Score:** Crucial for imbalanced datasets, showing the trade-off between precision and recall.
    
*   **Distribution of Predicted Probabilities:** Histograms of predicted probabilities for actual high-value and non-high-value customers, indicating model confidence and separation.
    

4\. Key Insights & Results
--------------------------

*   **Class Imbalance:** The project dataset exhibits class imbalance, with high-value customers being a minority class. This highlights the importance of metrics like AUC and Precision-Recall.
    
*   **RFM Importance:** High-value customers are strongly characterized by lower Recency (recent purchases), higher Frequency (more frequent purchases), and higher Monetary value (more spending).
    
*   **Geographical Concentration:** A significant portion of customers, including high-value ones, are from the United Kingdom.
    
*   **Model Performance:** The Neural Network demonstrates good capability in distinguishing high-value customers, indicated by strong ROC AUC and Precision-Recall scores, even with class imbalance. The probability distributions show reasonable separation between the predicted scores of the two classes.
    

5\. How to Run the Code
-----------------------

1.  **Download the Dataset:** Ensure you have the Online Retail.xlsx (convert it into a csv file before use)in the same directory as your Python script or Jupyter Notebook.
    
2.  Bashpip install pandas numpy matplotlib seaborn scikit-learn tensorflow
    
3.  **Run the Notebook/Script:**
    
    *   If using a Jupyter Notebook, copy the provided code blocks into separate cells and run them sequentially.
        
    *   Bashpython your\_project\_script\_name.py
        

6\. Requirements
----------------

*   Python 3.x, pandas, numpy, matplotlib, seaborn, scikit-learn, TensorFlow
    
*   (Optional but recommended: Jupyter Notebook for interactive execution)
    

7\. Future Work
---------------

*   **Advanced Imbalance Handling:** Explore techniques like SMOTE, ADASYN, or class weighting during model training to further address class imbalance.
    
*   **Hyperparameter Tuning:** Systematically tune the Neural Network's architecture (number of layers, neurons), learning rate, and regularization techniques for optimal performance.
    
*   **Other Models:** Experiment with other machine learning models (e.g., Gradient Boosting, Random Forest) to compare performance.
    
*   **Customer Lifetime Value (CLTV) Prediction:** Extend the project to predict actual CLTV rather than just a high-value segment.
    
*   **Time-Series Features:** Incorporate more time-series based features if granular transaction timestamps allow (e.g., time between purchases).
    

Further Contributions are Highly appreciated, feel free to commit changes
