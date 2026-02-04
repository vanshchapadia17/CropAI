import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        # Initializing the components
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("--- Starting the CropAI Training Pipeline ---")

            # 1. Data Ingestion
            # This fetches raw images and splits them into Train/Test CSVs
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()

            # 2. Data Transformation
            # This resizes images and returns numpy arrays ready for EfficientNet
            X_train_img, y_train_cat, X_test_img, y_test_cat,num_classes = \
                self.data_transformation.initiate_data_transformation(train_path, test_path)

            # 3. Model Training
            # This trains the model and saves the final model.h5
            model_path = self.model_trainer.initiate_model_trainer(
                X_train_img, 
                y_train_cat, 
                X_test_img, 
                y_test_cat, 
                num_classes
            )
            
            logging.info(f"--- Pipeline Completed Successfully! Model saved at: {model_path} ---")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # This allows you to run 'python src/pipeline/train_pipeline.py' from your terminal
    obj = TrainPipeline()
    obj.run_pipeline()