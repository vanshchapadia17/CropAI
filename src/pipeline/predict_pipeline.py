import sys
import os
import pandas as pd
import numpy as np
import cv2
from src.exception import CustomException
from src.utils import load_object
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, image_path):
        try:
            # 1. Define paths to our saved artifacts
            model_path = os.path.join("data", "model.h5")
            preprocessor_path = os.path.join("data", "preprocessor.pkl")

            # 2. Load the model and the preprocessing settings
            model = load_model(model_path)
            # This is the dictionary we saved: {"img_size": 224, "num_classes": 5}
            preprocess_dict = load_object(file_path=preprocessor_path)

            # 3. Process the new image (The user's upload)
            img_size = preprocess_dict["img_size"]
            
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = preprocess_input(img)
            
            # Expand dimensions because model expects (batch_size, height, width, channels)
            # This turns (224, 224, 3) into (1, 224, 224, 3)
            img = np.expand_dims(img, axis=0)

            # 4. Get Prediction
            preds = model.predict(img)
            
            # np.argmax picks the index of the highest probability
            # e.g., if [0.1, 0.1, 0.7, 0.1], it picks index 2
            result = np.argmax(preds, axis=1)
            
            return result[0]

        except Exception as e:
            raise CustomException(e, sys)