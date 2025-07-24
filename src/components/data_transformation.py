import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        """
        Creates a preprocessing pipeline for numerical and categorical features.
        """
        try:
            logging.info("Creating preprocessing pipelines")

            # For example usage; actual features should be passed dynamically or extracted
            numerical_features = ['numerical_col1', 'numerical_col2']
            categorical_features = ['categorical_col1', 'categorical_col2']

            # Numerical pipeline: median imputation + standard scaling
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline: mode imputation + one-hot + standard scaling
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ("scaler", StandardScaler())
            ])

            # Combine both
            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            target_column = 'target'  # Change this to your actual label column
            input_features_train = train_df.drop(columns=[target_column])
            target_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column])
            target_test = test_df[target_column]

        # Infer column types
            numerical_features = input_features_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_features = input_features_train.select_dtypes(include=["object", "category"]).columns.tolist()

            preprocessor = ColumnTransformer(transformers=[
                ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numerical_features),

            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ("scaler", StandardScaler())
            ]), categorical_features)
            ])

            input_feature_train_arr = preprocessor.fit_transform(input_features_train)
            input_feature_test_arr = preprocessor.transform(input_features_test)

             # ✅ Save the preprocessor for reuse
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessing object saved successfully")

            train_arr = np.hstack((input_feature_train_arr, target_train.values.reshape(-1, 1)))
            test_arr = np.hstack((input_feature_test_arr, target_test.values.reshape(-1, 1)))

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
