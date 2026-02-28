# 🛒 Big Mart Sales Prediction

A machine learning project that predicts item-level sales across Big Mart outlets using historical sales data. Built end-to-end --> from data cleaning and feature engineering to model training and a deployed Streamlit web app.

---

## 🔍 What's Been Done

**Exploratory Data Analysis** - Analyzed sales patterns across product types, outlet sizes, outlet types, and establishment years. Handled missing values in `Item_Weight` (interpolation) and `Outlet_Size` (mode imputation by outlet type). Treated zero values in `Item_Visibility` as missing and interpolated them.

**Feature Engineering** - Standardized `Item_Fat_Content` labels (e.g. `low fat`, `LF` → `LF`). Extracted item category from `Item_Identifier` prefix. Converted `Outlet_Establishment_Year` into `Outlet_age` (years since establishment). Dropped low-importance features identified via XGBoost feature importance: `Item_Visibility`, `Item_Weight`, `Item_Type`, `Outlet_Location_Type`, `Item_Identifier`, `Item_Fat_Content`.

**Model Training & Evaluation** - Compared Random Forest and XGBoost RF Regressor using 5-fold cross-validation (R² scoring). Final model trained on 5 features: `Item_MRP`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Type`, and `Outlet_age`. Model evaluated using Mean Absolute Error (MAE ≈ ₹714).

**Deployment** - Model serialized with Joblib and deployed as an interactive Streamlit web app with a clean dark UI.

🚀 **Live App:** [Click here](https://bigmartsalesforecasting-ckdpugmbarpap4yaajwdfe.streamlit.app/)

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

## 🔮 Upcoming Improvements

- **Explainable AI (SHAP)** - Adding SHAP value visualizations to the Streamlit app so users can see *which features drove each prediction*, not just the number itself
- **Better Evaluation** - Adding R², RMSE, and an Actual vs Predicted scatter plot to the notebook for a more complete model assessment
- **Feature Importance Dashboard** - Visual breakdown of how much each input (MRP, outlet type, age etc.) contributes to the final prediction in streamlit
- **Confidence Intervals** - Moving from a fixed MAE-based range to proper prediction intervals using quantile regression

---

## 🛠 Tech Stack

Python, XGBoost, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Joblib
