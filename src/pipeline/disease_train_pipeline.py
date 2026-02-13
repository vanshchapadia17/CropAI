import sys
from src.exception import CustomException
from src.logger import logging
from src.components.disease_ingestion import DiseaseIngestion
from src.components.disease_transformation import DiseaseTransformation
from src.components.disease_trainer import DiseaseTrainer

class DiseaseTrainPipeline:
    def __init__(self):
        self.data_ingestion = DiseaseIngestion()
        self.data_transformation = DiseaseTransformation()
        self.model_trainer = DiseaseTrainer()

    def run_pipeline(self):
        try:
            logging.info("--- Starting the Disease Detection Training Pipeline ---")

            # 1. Data Ingestion
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()

            # 2. Data Transformation
            X_train_img, y_train_cat, X_test_img, y_test_cat, num_classes = \
                self.data_transformation.initiate_data_transformation(train_path, test_path)

            # 3. Model Training
            model_path = self.model_trainer.initiate_model_trainer(
                X_train_img,
                y_train_cat,
                X_test_img,
                y_test_cat,
                num_classes
            )

            logging.info(f"--- Disease Pipeline Completed Successfully! Model saved at: {model_path} ---")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DiseaseTrainPipeline()
    obj.run_pipeline()
