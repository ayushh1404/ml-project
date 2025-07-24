# 🧠 ML Project Pipeline - Auto Classification & Regression

This is a dynamic end-to-end machine learning pipeline that **automatically detects whether the problem is classification or regression**, based on the dataset's target variable, and trains the best model accordingly.

It includes:
- Clean modular design
- Model selection with hyperparameter tuning
- Logging & exception handling
- Artifacts saving (trained model, data splits, etc.)

---

## 🚀 Features

✅ Automatic classification/regression selection  
✅ Model selection with GridSearchCV  
✅ Cleaned pipeline with logging & error handling  
✅ Easily extendable to any structured CSV dataset  
✅ Future-ready structure for deployment (API & UI ready)

---

## 🧠 Models Used

### Classification:
- Random Forest
- Decision Tree
- Gradient Boosting
- Logistic Regression
- KNN
- XGBoost
- CatBoost
- AdaBoost

### Regression:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost
- CatBoost
- Decision Tree Regressor
- AdaBoost Regressor

---

## 🔁 Pipeline Flow

1. **Data Ingestion**  
   Reads CSV and splits into train/test → `artifacts/train.csv`, `test.csv`

2. **Data Transformation**  
   - Handles scaling, encoding, missing values
   - Returns NumPy arrays

3. **Model Trainer (Dynamic)**  
   - Detects whether the target is categorical or continuous
   - Selects classification or regression path
   - Performs GridSearchCV
   - Saves best model → `artifacts/model.pkl`

4. **Prediction Ready**  
   You can now load the model and serve it via API or Web UI.

---

## ⚙️ How It Works

### Auto-detection of task:

'''python
if is_classification_task(y_train):
    use classification trainer
else:
    use regression trainer
