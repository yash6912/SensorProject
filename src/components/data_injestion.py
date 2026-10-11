import sys 
import os
import numpy as np 
import pandas as pd
from pymongo import MongoClient
from zipfile import path
from src.constant import *
from src.exception import customException
from src.logger import logging
from src.constant import *
from src.utils.main_utils import main_utils
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)


class dataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = main_utils()


    """
                Export a MongoDB collection as a pandas DataFrame.
        
                Args:
                    database_name (str): The name of the MongoDB database.
                    collection_name (str): The name of the MongoDB collection.
                Returns:
                    pd.DataFrame: The exported data as a pandas DataFrame.
    """

    def export_collection_as_dataframe(self, database_name: str, collection_name: str):

        try:
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list:
                df = df.drop(columns=["_id"])

            df.replace("na", np.NAN, inplace=True)
            return df
        except Exception as e:
            raise customException(e, sys)

    def export_data_into_feature_store_file_path(self) -> pd.DataFrame:
        try:
            logging.info("Exporting data from MongoDB to feature store")
            raw_file_path = self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path, exist_ok=True)

            sensor_data = self.export_collection_as_dataframe(
                database_name=MONGO_DATABASE_NAME, collection_name=MONGO_COLLECTION_NAME
            )

            logging.info(f"Saving exported data into feature store file path :{raw_file_path}")
            feature_store_file_path = os.path.join(raw_file_path, "wafer_fault.csv")
            sensor_data.to_csv(feature_store_file_path, index=False, header=True)
            return feature_store_file_path
        except Exception as e:
            raise customException(e, sys) 

    def initiate_data_ingestion(self) -> path:
        logging.info("Entered initiated_data_ingestion method of dataIngestion class")
        try:
            feature_store_file_path = self.export_data_into_feature_store_file_path()

            logging.info("got the data from mongoDB")
            logging.info("Exited and initiated_data_ingestion method of dataIngestion class")

            return feature_store_file_path
        
        except Exception as e:
            raise customException(e, sys) from e