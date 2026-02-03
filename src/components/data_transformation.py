import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    
    preprocessor_obj_file_path = os.path.join('data', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        For CropAI, we only want to keep 'path' and 'croplabel'.
        Everything else (Unnamed: 0, crop name) is dropped.
        '''
        try:
            # We use 'passthrough' for the columns we want to keep
            # and let ColumnTransformer drop the rest automatically
            keep_columns = ['path', 'croplabel']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("keep_columns", "passthrough", keep_columns)
                ],
                remainder="drop" # This drops Unnamed: 0 and 'crop' name
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data successfully")
            logging.info("Obtaining preprocessing object to drop unnecessary columns")

            preprocessing_obj = self.get_data_transformer_object()

            # Transform the dataframes
            # Note: For CNNs, we usually keep them as DataFrames/Arrays of paths
            train_data = preprocessing_obj.fit_transform(train_df)
            test_data = preprocessing_obj.transform(test_df)

            logging.info("Column dropping completed. Saving preprocessing object.")

            # Save the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_data,
                test_data,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)