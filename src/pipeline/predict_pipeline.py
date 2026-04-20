import sys
import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

from src.exception import CustomException
from src.utils import load_object


def _build_inference_model(num_classes: int) -> Model:
    # Mirror the trainer's inference-time graph (no augmentation). imagenet init
    # is required — EfficientNetB0's internal Normalization layer only holds
    # correct mean/variance when loaded from imagenet; the trained weights
    # overlay the fine-tuned values on top.
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)


def _load_weights_npy(model: Model, path: str) -> None:
    # Positional weight load — trainer and predict build the same architecture,
    # so model.weights lines up one-to-one with the saved array.
    data = np.load(path, allow_pickle=True)
    for var, arr in zip(model.weights, data):
        var.assign(arr)


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, image_path):
        try:
            model_path = os.path.join("artifacts", "model.npy")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            preprocess_dict = load_object(file_path=preprocessor_path)
            img_size = preprocess_dict["img_size"]
            num_classes = preprocess_dict["num_classes"]

            model = _build_inference_model(num_classes)
            _load_weights_npy(model, model_path)

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
