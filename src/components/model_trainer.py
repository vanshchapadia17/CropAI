import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    
    trained_model_file_path = os.path.join("artifacts", "model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_img, y_train_cat, X_test_img, y_test_cat, num_classes):
        try:

            logging.info("Applying training-time augmentation")

            # Fresh EarlyStopping per phase — reusing one instance across the
            # frozen-base fit and the fine-tune fit carries phase-1's best
            # val_accuracy into phase 2 and causes premature stop + weight
            # rollback that erases fine-tuning.
            def _make_early_stop():
                return tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=4,
                    restore_best_weights=True,
                )

            X_train_img = tf.convert_to_tensor(X_train_img, dtype=tf.float32)
            X_test_img  = tf.convert_to_tensor(X_test_img, dtype=tf.float32)

            logging.info("Building the EfficientNetB0 model architecture")

            # 1. Load the Base Model
            base_model = EfficientNetB0(
                weights="imagenet",
                include_top=False, # We don't want the 1000-class ImageNet top
                input_shape=(224, 224, 3)
            )

            # 2. Freeze the base layers (Transfer Learning)
            base_model.trainable = False

            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomBrightness(0.2),
            ])

            # 3. Add custom layers for CropAI
            inputs = base_model.input
            x = data_augmentation(inputs)
            x = base_model(x, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs, outputs)

            # 4. Compile the model
            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )

            logging.info("Starting initial model training (Frozen Base)")

            # 5. Training (Initial)
            model.fit(
                X_train_img,
                y_train_cat,
                validation_data=(X_test_img, y_test_cat),
                epochs=5,
                batch_size=10,
                callbacks=[_make_early_stop()]
            )

            # 6. Fine-Tuning (Unfreezing)
            logging.info("Starting fine-tuning (Unfreezing base model)")
            for layer in base_model.layers[:-50]:
                layer.trainable = False
            for layer in base_model.layers[-50:]:
                layer.trainable = True
            
            # Recompile with a much lower learning rate for fine-tuning
            model.compile(
                optimizer=Adam(learning_rate=5e-6),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )

            class_weight = {
                            0: 1.0,
                            1: 1.0,
                            2: 1.8,
                            3: 1.2,
                            4: 1.0
                            }
            model.fit(
                X_train_img,      
                y_train_cat,
                validation_data=(X_test_img, y_test_cat),
                epochs=10,
                batch_size=16,
                class_weight=class_weight,
                callbacks=[_make_early_stop()]
            )

            logging.info("Generating evaluation metrics")

            # Predict probabilities
            y_pred_probs = model.predict(X_test_img)

            # Convert probabilities to class labels
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_test_cat, axis=1)

            
            # Classification report
            report = classification_report(
                y_true,
                y_pred,
                digits=4
            )

            logging.info("Classification Report:\n" + report)
            print("\nClassification Report:\n", report)

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            logging.info(f"Confusion Matrix:\n{cm}")
            print("\nConfusion Matrix:\n", cm)

            os.makedirs("artifacts", exist_ok=True)

            with open("artifacts/metrics.txt", "w") as f:
                f.write("Classification Report:\n")
                f.write(report) 
                f.write("\n\nConfusion Matrix:\n")
                f.write(str(cm))


            logging.info(f"Model training completed. Saving model to: {self.model_trainer_config.trained_model_file_path}")

            # 7. Save the final model
            # TF 2.12 can't JSON-serialize EfficientNetB0's internal Normalization
            # layer, so full model.save() to .h5 fails. Save weights only from a
            # rebuilt inference-time graph (no augmentation) so the predict
            # pipeline — which builds the same topology — can load them back.
            inf_inputs = tf.keras.Input(shape=(224, 224, 3))
            y = base_model(inf_inputs, training=False)
            y = GlobalAveragePooling2D()(y)
            y = Dropout(0.2)(y)
            inf_outputs = Dense(num_classes, activation="softmax")(y)
            inference_model = Model(inf_inputs, inf_outputs)
            inference_model.layers[-1].set_weights(model.layers[-1].get_weights())

            inference_model.save_weights(self.model_trainer_config.trained_model_file_path)

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)