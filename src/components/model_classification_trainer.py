import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models  # assumes utils works for classifiers

@dataclass
class ModelTrainerClassificationConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainerClassification:
    def __init__(self):
        self.model_trainer_config = ModelTrainerClassificationConfig()

    def initiate_model_trainer_classification(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "KNN Classifier": KNeighborsClassifier()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 10, 15]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 5, 10]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 150]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0]
                },
                "CatBoost Classifier": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100]
                },
                "XGBoost Classifier": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [50, 100]
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                "KNN Classifier": {
                    'n_neighbors': [3, 5, 7]
                }
            }

            model_report, best_model= evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
        

            if best_model_score < 0.6:
                raise CustomException("No good classification model found")

            logging.info(f"Best model: {best_model_name} with accuracy: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_pred = best_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, y_pred)
            return final_accuracy

        except Exception as e:
            raise CustomException(e, sys)
