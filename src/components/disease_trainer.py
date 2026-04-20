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

@dataclass
class DiseaseTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "disease_model.npy")

class DiseaseTrainer:
    def __init__(self):
        self.model_trainer_config = DiseaseTrainerConfig()

    def initiate_model_trainer(self, X_train_img, y_train_cat, X_test_img, y_test_cat, num_classes):
        try:
            logging.info("Applying training-time augmentation for disease model")

            # Fresh EarlyStopping per phase — reusing one instance across the
            # frozen-base fit and the fine-tune fit carries phase-1's best
            # val_accuracy into phase 2 and causes premature stop + weight
            # rollback that erases fine-tuning.
            def _make_early_stop():
                return EarlyStopping(
                    monitor="val_accuracy",
                    patience=4,
                    restore_best_weights=True,
                )

            X_train_img = tf.convert_to_tensor(X_train_img, dtype=tf.float32)
            X_test_img = tf.convert_to_tensor(X_test_img, dtype=tf.float32)

            logging.info("Building the EfficientNetB0 disease model architecture")

            base_model = EfficientNetB0(
                weights="imagenet",
                include_top=False,
                input_shape=(224, 224, 3)
            )

            base_model.trainable = False

            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomBrightness(0.2),
            ])

            inputs = base_model.input
            x = data_augmentation(inputs)
            x = base_model(x, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs, outputs)

            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )

            logging.info("Starting initial disease model training (Frozen Base)")

            model.fit(
                X_train_img,
                y_train_cat,
                validation_data=(X_test_img, y_test_cat),
                epochs=5,
                batch_size=16,
                callbacks=[_make_early_stop()]
            )

            # Fine-Tuning
            logging.info("Starting fine-tuning (Unfreezing base model)")
            for layer in base_model.layers[:-50]:
                layer.trainable = False
            for layer in base_model.layers[-50:]:
                layer.trainable = True

            model.compile(
                optimizer=Adam(learning_rate=5e-6),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )

            # Class weights: sugarcane classes (~500 imgs) get higher weight than rice classes (~1500 imgs)
            # Alphabetical order: Bacterialblight(0), Blast(1), Brownspot(2), Healthy(3),
            # Mosaic(4), RedRot(5), Rust(6), Tungro(7), Yellow(8)
            class_weight = {
                0: 1.0,   # Bacterialblight (~1584)
                1: 1.0,   # Blast (~1440)
                2: 1.0,   # Brownspot (~1600)
                3: 2.5,   # Healthy (~522)
                4: 2.8,   # Mosaic (~462)
                5: 2.5,   # RedRot (~518)
                6: 2.5,   # Rust (~514)
                7: 1.2,   # Tungro (~1308)
                8: 2.6,   # Yellow (~505)
            }

            model.fit(
                X_train_img,
                y_train_cat,
                validation_data=(X_test_img, y_test_cat),
                epochs=10,
                batch_size=32,
                class_weight=class_weight,
                callbacks=[_make_early_stop()]
            )

            logging.info("Generating disease model evaluation metrics")

            y_pred_probs = model.predict(X_test_img)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_test_cat, axis=1)

            report = classification_report(
                y_true,
                y_pred,
                digits=4
            )

            logging.info("Disease Classification Report:\n" + report)
            print("\nDisease Classification Report:\n", report)

            cm = confusion_matrix(y_true, y_pred)
            logging.info(f"Disease Confusion Matrix:\n{cm}")
            print("\nDisease Confusion Matrix:\n", cm)

            os.makedirs("artifacts", exist_ok=True)

            with open("artifacts/disease_metrics.txt", "w") as f:
                f.write("Disease Classification Report:\n")
                f.write(report)
                f.write("\n\nConfusion Matrix:\n")
                f.write(str(cm))

            logging.info(f"Disease model training completed. Saving model to: {self.model_trainer_config.trained_model_file_path}")

            # TF 2.12's Keras H5 / SavedModel serializers both try to JSON-encode
            # layer configs, which breaks on the EagerTensors inside
            # EfficientNetB0's Normalization layer. Sidestep Keras serialization
            # entirely: rebuild an inference-only graph (no augmentation) and
            # dump its weights as a positional numpy object-array. The predict
            # pipeline builds the same architecture and assigns by position.
            inf_inputs = tf.keras.Input(shape=(224, 224, 3))
            y = base_model(inf_inputs, training=False)
            y = GlobalAveragePooling2D()(y)
            y = Dropout(0.2)(y)
            inf_outputs = Dense(num_classes, activation="softmax")(y)
            inference_model = Model(inf_inputs, inf_outputs)
            inference_model.layers[-1].set_weights(model.layers[-1].get_weights())

            weights = [w.numpy() for w in inference_model.weights]
            np.save(
                self.model_trainer_config.trained_model_file_path,
                np.array(weights, dtype=object),
                allow_pickle=True,
            )

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
