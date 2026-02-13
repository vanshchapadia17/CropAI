import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DiseaseIngestionConfig:
    train_data_path: str = os.path.join('data', "train2.csv")
    test_data_path: str = os.path.join('data', "test2.csv")
    raw_data_path: str = os.path.join('data', "raw2.csv")

class DiseaseIngestion:
    def __init__(self):
        self.ingestion_config = DiseaseIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the disease data ingestion component")
        try:
            # Build dataset from folder structure: data/RD/* and data/SD/*
            data_dirs = [
                os.path.join('data', 'RD'),
                os.path.join('data', 'SD')
            ]

            records = []
            for data_dir in data_dirs:
                if not os.path.exists(data_dir):
                    logging.warning(f"Directory not found: {data_dir}")
                    continue
                for disease_folder in sorted(os.listdir(data_dir)):
                    disease_path = os.path.join(data_dir, disease_folder)
                    if not os.path.isdir(disease_path):
                        continue
                    for img_file in os.listdir(disease_path):
                        img_path = os.path.join(disease_path, img_file)
                        if os.path.isfile(img_path):
                            records.append({
                                'path': img_path,
                                'disease': disease_folder
                            })

            df = pd.DataFrame(records)
            logging.info(f"Built disease dataset with {len(df)} images across {df['disease'].nunique()} classes")

            # Encode disease labels as integers (alphabetical order)
            label_map = {name: idx for idx, name in enumerate(sorted(df['disease'].unique()))}
            df['diseaselabel'] = df['disease'].map(label_map)
            logging.info(f"Label mapping: {label_map}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df['diseaselabel']
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Disease data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DiseaseIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
