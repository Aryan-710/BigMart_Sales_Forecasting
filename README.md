# 🛒 Big Mart Sales Prediction
A machine learning project that predicts item-level sales across Big Mart outlets using historical sales data. Built end-to-end --> from data cleaning and feature engineering to model training and a deployed Streamlit web app.
---
## 🔍 What's Been Done
**Exploratory Data Analysis** - Analyzed sales patterns across product types, outlet sizes, outlet types, and establishment years. Handled missing values in `Item_Weight` (interpolation) and `Outlet_Size` (mode imputation by outlet type). Treated zero values in `Item_Visibility` as missing and interpolated them.
**Feature Engineering** - Standardized `Item_Fat_Content` labels (e.g. `low fat`, `LF` → `LF`). Extracted item category from `Item_Identifier` prefix. Converted `Outlet_Establishment_Year` into `Outlet_age` (years since establishment). Dropped low-importance features identified via XGBoost feature importance: `Item_Visibility`, `Item_Weight`, `Item_Type`, `Outlet_Location_Type`, `Item_Identifier`, `Item_Fat_Content`.
**Model Training & Evaluation** - Compared Random Forest and XGBoost RF Regressor using 5-fold cross-validation (R2 scoring). Final model trained on 5 features: `Item_MRP`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Type`, and `Outlet_age`. Model evaluated using Mean Absolute Error (MAE ≈ ₹714), R², RMSE, and an Actual vs Predicted scatter plot for a complete performance assessment.
**Explainable AI (SHAP)** - Integrated SHAP value visualizations into the Streamlit app, allowing users to see which features drove each individual prediction, not just the final number.
**Feature Importance Dashboard** - Added an interactive visual breakdown in the Streamlit app showing how much each input (`Item_MRP`, outlet type, outlet age, etc.) contributes to the final prediction.
**Confidence Intervals** - Replaced fixed MAE-based ranges with proper prediction intervals using quantile regression for more statistically grounded uncertainty estimates.
**Deployment** - Model serialized with Joblib and deployed as an interactive Streamlit web app with a clean dark UI.
🚀 **Live App:** [Click here](https://bigmartsalesforecasting-ylr7vqjqjvyypnb7sm23ve.streamlit.app/)
---
## 🧠 Model Features
| Feature | Description |
|---|---|
| `Item_MRP` | Maximum Retail Price of the product |
| `Outlet_Identifier` | Which outlet the item is sold at |
| `Outlet_Size` | Size of the outlet (Small / Medium / High) |
| `Outlet_Type` | Type of outlet (Grocery Store / Supermarket) |
| `Outlet_age` | Years since the outlet was established |
---
## 📁 Project Structure
```
├── Big_Mart_Sales_Prediction.ipynb   # EDA, feature engineering, model training
├── app.py                            # Streamlit web application
├── bigmart_model                     # Serialized trained model (joblib)
├── requirements.txt                  # Python dependencies
└── README.md
```
---
## 🛠 Tech Stack
Python, XGBoost, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Joblib, SHAP
