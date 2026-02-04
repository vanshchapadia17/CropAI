import os
import sys
from dataclasses import dataclass

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    
    trained_model_file_path = os.path.join("data", "model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, train_label, test_array, test_label, num_classes):
        try:
            logging.info("Building the EfficientNetB0 model architecture")

            # 1. Load the Base Model
            base_model = EfficientNetB0(
                weights="imagenet",
                include_top=False, # We don't want the 1000-class ImageNet top
                input_shape=(224, 224, 3)
            )

            # 2. Freeze the base layers (Transfer Learning)
            base_model.trainable = False

            # 3. Add custom layers for CropAI
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.4)(x) # Help prevent overfitting on small datasets
            output = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=base_model.input, outputs=output)

            # 4. Compile the model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            logging.info("Starting initial model training (Frozen Base)")
            
            # 5. Training (Initial)
            model.fit(
                train_array,
                train_label,
                validation_data=(X_test_img, y_test_cat),
                epochs=10,
                batch_size=16 
            )

            # 6. Fine-Tuning (Unfreezing)
            logging.info("Starting fine-tuning (Unfreezing base model)")
            base_model.trainable = True
            
            # Recompile with a much lower learning rate for fine-tuning
            model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            model.fit(
                train_array,
                train_label,
                validation_data=(X_test_img, y_test_cat),
                epochs=5,
                batch_size=16
            )

            logging.info(f"Model training completed. Saving model to: {self.model_trainer_config.trained_model_file_path}")

            # 7. Save the final model
            # Note: We use model.save() because h5 files are better saved this way than dill
            model.save(self.model_trainer_config.trained_model_file_path)

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)