### **Analysis of Sales Prediction**  
This project involves analyzing and predicting product sales based on various marketing strategies and channels using statistical and machine learning techniques. The goal is to determine the impact of different factors on sales and develop models to accurately forecast product sales.

---

## **Table of Contents**
1. [Introduction](#introduction)  
2. [Dataset Description](#dataset-description)  
3. [Project Workflow](#project-workflow)  
4. [Analysis and Insights](#analysis-and-insights)  
5. [Models Implemented](#models-implemented)  
6. [Results and Evaluation](#results-and-evaluation)  
7. [Installation and Usage](#installation-and-usage)  
8. [Conclusion](#conclusion)  

---

## **Introduction**
This project is a comprehensive analysis of sales data to predict product sales based on the expenditure on various marketing channels. By exploring relationships between factors like TV, Google Ads, and Social Media marketing, we aim to:
- Understand the significant predictors of product sales.
- Develop regression and classification models for sales prediction.
- Identify strategies for maximizing marketing efficiency.

---

## **Dataset Description**
The dataset contains information on:  
- **Independent Variables (Marketing Channels):**
  - TV Advertising Expenditure  
  - Billboards Advertising Expenditure  
  - Google Ads Expenditure  
  - Social Media Marketing Expenditure  
  - Influencer Marketing Expenditure  
  - Affiliate Marketing Expenditure  
- **Target Variable:**  
  - **Product_Sold**: Number of products sold during the marketing campaign.

---

## **Project Workflow**
1. **Exploratory Data Analysis (EDA):**
   - Summary statistics and data visualization.
   - Correlation analysis between marketing channels and sales.

2. **Model Development:**
   - Fitting a multiple linear regression model to analyze relationships.
   - Checking assumptions: residual analysis, multicollinearity (VIF).

3. **Machine Learning Models:**
   - Implemented K-Nearest Neighbors (KNN), Random Forest, and Logistic Regression.
   - Train-Test split for model evaluation.

4. **Subset Selection:**
   - Performed feature selection to identify the best predictors of sales.

5. **Evaluation Metrics:**
   - Accuracy, R-squared, and residual plots for evaluation.

---

## **Analysis and Insights**
- **TV Advertising and Google Ads** were found to be the most significant predictors of sales.
- High multicollinearity was observed among some marketing channels.
- A combination of traditional and digital marketing channels improves sales efficiency.

---

## **Models Implemented**
1. **Linear Regression:**
   - Analyzed the contribution of each marketing channel to sales.
   - Residual analysis and multicollinearity checks.

2. **K-Nearest Neighbors (KNN):**
   - Implemented to predict sales based on neighborhood patterns.

3. **Random Forest:**
   - Used for robust prediction and handling non-linear relationships.

4. **Logistic Regression:**
   - Employed for classification tasks.

---

## **Results and Evaluation**
- **Linear Regression:** Achieved an R-squared value of ~0.85, indicating a strong fit.
- **KNN Accuracy:** ~78% accuracy in predicting sales.
- **Random Forest Performance:** Outperformed other models in handling complex relationships.
- **Feature Importance:** TV and Google Ads were consistently ranked as significant predictors.

---

## **Installation and Usage**
### **Prerequisites**
- Install R and the following libraries:  
  - `caret`  
  - `ggplot2`  
  - `corrplot`  
  - `class`  
  - `randomForest`  

### **Steps:**
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/sales-prediction-analysis.git
   cd sales-prediction-analysis
   ```
2. Load the dataset:  
   Ensure the `Advertising_Data.csv` file is in the working directory.
3. Run the script:  
   Use an R environment or RStudio to execute the script step by step.

---

## **Conclusion**
This project highlights the importance of analyzing marketing channels to maximize sales. The developed models provide actionable insights for optimizing marketing strategies and demonstrate the power of data-driven decision-making in sales forecasting.

