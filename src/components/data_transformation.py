import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from src.exception import CustomException
from src.logger import logging

#image sizing 
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.utils import save_object

@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join('data', "preprocessor.pkl")
    img_size: int = 224

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
        
    def load_and_preprocess_images(self, path_array):    
        """
        This replaces your notebook 'load_images' function.
        Automates reading, resizing, and EfficientNet preprocessing.
        """
        try:
            images = []
            for path in path_array:
                img = cv2.imread(path)
                if img is None:
                    logging.warning(f"Image not found at: {path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.data_transformation_config.img_size, self.data_transformation_config.img_size))
                img = preprocess_input(img)
                images.append(img)

            return np.array(images)
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

            #image resizing
            # DYNAMIC CLASS DETECTION
            num_classes = len(train_df['croplabel'].unique())
            logging.info(f"Detected {num_classes} unique classes.")

            train_paths = train_df['path'].values
            train_labels = train_df['croplabel'].values

            test_paths = test_df['path'].values
            test_labels = test_df['croplabel'].values

            logging.info("image preprocessing start!")
            X_train_img = self.load_and_preprocess_images(train_paths)
            X_test_img = self.load_and_preprocess_images(test_paths)
            
            #encoding is used to convert the labels 0,1,2,3,4 into [] , if 0 = [1,0,0,0,0 ]
            logging.info("Converting labels to categorical format")
            y_train_cat = to_categorical(train_labels, num_classes=self.data_transformation_config.num_classes)
            y_test_cat = to_categorical(test_labels, num_classes=self.data_transformation_config.num_classes)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={
                    "img_size": self.data_transformation_config.img_size, 
                    "num_classes": num_classes
                    }
            )

            logging.info("Data Transformation complete.")

            return (
                X_train_img,
                y_train_cat,
                X_test_img,
                y_test_cat,
                self.data_transformation_config.preprocessor_obj_file_path,
                num_classes
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
''' if __name__=="__main__":
    # This ensures that when you run THIS file, the logging starts
    logging.info("Testing the logger from Data Transformation")
'''