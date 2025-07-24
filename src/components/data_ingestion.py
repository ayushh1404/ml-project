import os
import sys
import pandas as pd
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_classification_trainer import ModelTrainerClassification
from src.components.model_classification_trainer import ModelTrainerClassificationConfig




@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw dataset, splits it into train and test sets, and saves them into the artifacts folder.
        Returns:
            Tuple containing the file paths of train and test datasets.
        """
        logging.info("Starting data ingestion process.")

        try:
            # Step 1: Read dataset
            raw_file_path = os.path.join('Notebook\data\heart.csv')
            logging.info(f"Reading data from: {raw_file_path}")
            df = pd.read_csv('notebook\data\heart.csv')
            logging.info(f"Dataset shape: {df.shape}")

            # Step 2: Ensure artifacts folder exists
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Step 3: Save raw data
            df.to_csv(self.config.raw_data_path, index=False , header=True)
            logging.info(f"Raw data saved at: {self.config.raw_data_path}")

            # Step 4: Split data into train/test
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into training and testing sets.")

            # Step 5: Save train and test datasets
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Training data saved at: {self.config.train_data_path}")
            logging.info(f"Testing data saved at: {self.config.test_data_path}")

            logging.info("Data ingestion completed successfully.")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error("Error during data ingestion.")
            raise CustomException(e, sys)
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_df,test_df= ingestion.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_df,test_df)

    modeltrainer=ModelTrainerClassification()
    print(modeltrainer.initiate_model_trainer_classification(train_arr,test_arr))
