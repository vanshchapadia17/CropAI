import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import cv2
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore

from src.utils import save_object

@dataclass
class DiseaseTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "disease_preprocessor.pkl")
    img_size: int = 224

class DiseaseTransformation:
    def __init__(self):
        self.data_transformation_config = DiseaseTransformationConfig()

    def load_and_preprocess_images(self, path_array):
        try:
            images = []
            for path in path_array:
                img = cv2.imread(path)
                if img is None or len(img.shape) != 3:
                    logging.warning(f"Image not found or invalid at: {path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.data_transformation_config.img_size,
                                       self.data_transformation_config.img_size),
                                       interpolation=cv2.INTER_AREA)
                img = preprocess_input(img)
                images.append(img)

            return np.array(images)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read disease train and test data successfully")

            num_classes = len(train_df['diseaselabel'].unique())
            logging.info(f"Detected {num_classes} unique disease classes")

            train_paths = train_df['path'].values
            train_labels = train_df['diseaselabel'].values

            test_paths = test_df['path'].values
            test_labels = test_df['diseaselabel'].values

            logging.info("Disease image preprocessing start!")
            X_train_img = self.load_and_preprocess_images(train_paths)
            X_test_img = self.load_and_preprocess_images(test_paths)

            logging.info("Converting disease labels to categorical format")
            y_train_cat = to_categorical(train_labels, num_classes=num_classes)
            y_test_cat = to_categorical(test_labels, num_classes=num_classes)

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={
                    "img_size": self.data_transformation_config.img_size,
                    "num_classes": num_classes
                }
            )

            logging.info("Disease data transformation complete.")

            return (
                X_train_img,
                y_train_cat,
                X_test_img,
                y_test_cat,
                num_classes
            )

        except Exception as e:
            raise CustomException(e, sys)
