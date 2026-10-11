import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import standardscaler,robustscaler,functiontransformer
from sklearn.pipeline import Pipeline

from src.constant import *
from src.exception import customException
from src.logger import logging
from src.constant import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    artfact_dir = os.path.join(artifact_folder)
    transformed_train_file_path = os.path.join(artfact_dir, "train.npy")
    transformed_test_file_path = os.path.join(artfact_dir, "test.npy")
    transformed_object_file_path = os.path.join(artfact_dir, "preprocessor.pkl")

class DataTransformation:
    def __init__(self,feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_data(feature_store_file_path:str) -> pd.DataFrame:
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"Good/Bad":TARGET_COLUMN}, inplace=True)
            return data

        except Exception as e:
            raise customException(e, sys)

    def get_data_transformer_object(self):
        try:
            imputer_step = SimpleImputer(strategy="constant", fill_value=0)
            scaler_step  = ('scaler',robustscaler())

            preprocessor = Pipeline(
                steps=[
                    imputer_step,
                    scaler_step
                ]
            )
            return preprocessor
        except Exception as e:
            raise customException(e, sys)   

    def initiate_data_transformation(self):
        logging.info("Entered the initiated_data_transformation method of DataTransformation class")
        try:
            data_frame = self.get_data(feature_store_filePath = self.feature_store_file_path)

            x= data_frame.drop(columns=[TARGET_COLUMN])
            y = np.where(data_frame[TARGET_COLUMN] == -1, 1, 0)

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            preprocessor = self.get_data_transformer_object()
            X_train_scaled = preprocessor.fit_transform(X_train)
            x_test_scaled = preprocessor.transform(X_test)

            prepocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(prepocessor_path), exist_ok=True)
            self.utils.save_object(file_path=prepocessor_path, obj=preprocessor)

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c[x_test_scaled,np.array(y_test)]

            return (train_arr,test_arr,prepocessor_path)

        except Exception as e:
            raise customException(e,sys) from e