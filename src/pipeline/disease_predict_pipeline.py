import sys
import os
import numpy as np
import cv2
from src.exception import CustomException
from src.utils import load_object
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore

class DiseasePredictPipeline:
    def __init__(self):
        pass

    def predict(self, image_path):
        try:
            model_path = os.path.join("artifacts", "disease_model.h5")
            preprocessor_path = os.path.join("artifacts", "disease_preprocessor.pkl")

            model = load_model(model_path, compile=False)
            preprocess_dict = load_object(file_path=preprocessor_path)

            img_size = preprocess_dict["img_size"]

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = preprocess_input(img)

            img = np.expand_dims(img, axis=0)

            preds = model.predict(img)
            result = np.argmax(preds, axis=1)
            confidence = float(np.max(preds) * 100)

            return result[0], confidence

        except Exception as e:
            raise CustomException(e, sys)
