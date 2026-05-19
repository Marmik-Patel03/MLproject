import sys
from  dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            Numerical_Features = ['reading_score', 'writing_score']
            Categorical_Features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Numerical columns Standard Scaling Completed")

            logging.info("Categorical columns encoding Completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,Numerical_Features),
                    ("cat_pipeline",cat_pipeline,Categorical_Features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data completed")

            logging.info("Obtaining Preprocessing Object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "maths_score"
            numerical_columns = ['reading_score', 'writing_score']
        except:
            pass
