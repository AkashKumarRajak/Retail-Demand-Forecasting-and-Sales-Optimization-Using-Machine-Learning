# Retail Demand Forecasting and Sales Optimization Using Machine Learning

Forecasting weekly sales in retail using advanced machine learning techniques to optimize inventory, reduce overstock/understock issues, and improve promotional effectiveness.

---

## About the Project

Retailers often face issues in forecasting product demand across stores and departments, which leads to inefficiencies in inventory management and lost sales opportunities. This project leverages historical sales, store metadata, and promotional features to build accurate demand forecasting models using Random Forest and XGBoost.

---

## Project Structure

Retail-Demand-Forecasting<br>
│
├── Retail Demand Forecasting and Sales Optimization Using Machine Learning.ipynb<br>
├── train.csv<br>
├── test.csv<br>
├── features.csv<br>
├── stores.csv<br>
├── sampleSubmission.csv<br>
├── random_forest_model.pkl<br>
├── random_forest_sales_model.pkl<br>
├── xgboost_sales_model.pkl<br>
└── README.md<br>


---

## Problem Statement

Retail businesses struggle to accurately predict demand, especially during promotions or holidays. Incorrect forecasts lead to overstocking, understocking, or missed sales. The objective is to predict **weekly sales at the department-store level** using historical data and associated features.

---

## Objectives

- Predict future weekly sales for each department of each store.
- Identify the most important factors impacting sales (e.g., markdowns, holidays, store type).
- Build and compare robust machine learning models to find the most accurate forecasting solution.
- Reduce error margins to assist in better operational decision-making.

---

## Data Overview

1. **train.csv** – Historical data from 2010–2012 (Store, Dept, Date, Weekly_Sales, IsHoliday).
2. **test.csv** – Test set without Weekly_Sales, used for prediction.
3. **features.csv** – External factors like Temperature, Fuel_Price, MarkDowns, CPI, Unemployment.
4. **stores.csv** – Metadata about each store (Type, Size).
5. **sampleSubmission.csv** – Format required for final predictions submission.

---

## Modeling Approach

- Merge all data sources (train/test, features, stores) based on common keys.
- Perform exploratory data analysis (EDA) to understand patterns and distributions.
- Engineer features such as `Month`, `Year`, `IsHoliday`, `Markdown impact`, etc.
- Normalize and encode data for modeling.
- Split training data into validation sets for model testing.
- Compare model performance using cross-validation.
- Serialize best models for reuse on test data.

---

## Tools and Libraries

- **Language**: Python 3.x  
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Model Saving**: Joblib  
- **IDE**: Jupyter Notebook

---

## Model Used

| Model Name         | Description                                 | Evaluation Metrics |
|--------------------|---------------------------------------------|--------------------|
| Random Forest       | Ensemble tree-based model (baseline)       | RMSE, R² Score     |
| XGBoost Regressor   | Boosted trees for high accuracy            | RMSE, R² Score     |

Models are saved as:

- `random_forest_model.pkl`
- `random_forest_sales_model.pkl`
- `xgboost_sales_model.pkl`

---

## Key Steps in Notebook

1. **Data Cleaning & Merging**
   - Combined multiple datasets using Store and Date keys
   - Handled missing values and encoded holidays

2. **Feature Engineering**
   - Created features: `Week`, `Month`, `IsHoliday`, `Markdown_Total`
   - One-hot encoded categorical variables

3. **EDA (Exploratory Data Analysis)**
   - Visualized sales patterns
   - Identified trends around holidays and markdowns

4. **Model Training**
   - Used Random Forest and XGBoost with tuned hyperparameters
   - Evaluated using cross-validation

5. **Final Prediction**
   - Predictions on test data
   - Submission file generated using model output

---
## Business Impact
- This machine learning–based forecasting system provides real business value by empowering retail decision-makers with accurate and actionable insights:

1. **Inventory & Stock Optimization**
   - Enables more accurate stock forecasting, reducing both overstocking and stockouts.
   - Supports optimized warehouse-to-store allocation, ensuring the right inventory reaches the right locations with minimal cost.
   - Reduces waste and improves inventory turnover, especially for perishable or seasonal items.

2. **Operational Efficiency**
   - Improves demand visibility across stores and departments, aiding in supply chain planning, staffing, and store readiness.
   - Assists in automating replenishment planning, saving manual effort and increasing consistency.

3. **Promotion & Sales Performance Insights**
   - Helps in promotion planning by identifying how markdowns and holidays affect sales.
   - Enables tracking of campaign effectiveness, informing future marketing strategies.
   - Enhances sales predictions around seasonal or event-driven demand surges.

4. **Revenue Uplift & Cost Reduction**
   - Leads to improved sales forecasting accuracy, resulting in better financial planning.
   - Reduces revenue leakage by minimizing missed sales due to stockouts or poor promotion timing.

5. **Improved Customer Satisfaction**
   - Ensures better product availability, especially during peak seasons or campaigns.
   - Contributes to enhanced customer experience and brand loyalty.

---

## Conclusion

This project demonstrates the power of machine learning in solving retail demand forecasting problems. By incorporating a variety of external and store-level features, the models achieve strong predictive accuracy. XGBoost emerged as the most effective model, offering insights into the factors that drive sales and allowing businesses to take informed decisions.

---

## Results

| Model              | RMSE     | R² Score |
|-------------------|----------|----------|
| Random Forest      | 2150.73  | 0.872    |
| XGBoost (Final)    | **1923.65**  | **0.901**  |

> The XGBoost model was the best performer, reducing prediction error significantly and improving generalization across stores.

---

## Contact
Akash Kumar
- Email: akashkumarrajak200@gmail.com
- LinkedIn: [linkedin.com/in/akash-kumar-786](https://www.linkedin.com/in/akash-kumar-rajak-22a98623b/)
- GitHub: https://github.com/AkashKumarRajak
  


