from project import logger
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from project.entity.config_entity import DataTransformationConfig





class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.columns_to_drop = config.columns_to_drop
        self.target_column = config.target_column
        self.label_encoders = {}
        self.categorical_columns = config.categorical_columns
        self.numeric_columns = config.numeric_columns
        self.test_size = config.test_size
        self.random_state = config.random_state
        

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.copy()

            data.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')
            
            for column in self.categorical_columns:
                if column in data.columns:
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column].astype(str))
                    self.label_encoders[column] = le            
            
            
            
            os.makedirs(os.path.dirname(self.config.label_encoder), exist_ok=True)
            joblib.dump(self.label_encoders, self.config.label_encoder)
            logger.info(f"Saved label encoders to {self.config.label_encoder}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {str(e)}")
            raise e

    def train_test_splitting(self):
        try:
            logger.info(f"Loading data from {self.config.data_path}")
            data = pd.read_csv(self.config.data_path)
            
            data = self.preprocess_data(data)
            data = data.dropna()
            
            
            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]
            
            smote = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_data = X_resampled.copy()
            resampled_data[self.target_column] = y_resampled
            
            train, test = train_test_split(resampled_data, test_size= self.test_size, random_state=self.random_state)
            
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            logger.info("Split data into training and test sets")
            logger.info(f"Training data shape: {train.shape}")
            logger.info(f"Test data shape: {test.shape}")

            return train, test
            
        except Exception as e:
            logger.error(f"Error in train_test_splitting: {e}")
            raise e
    
    def preprocess_features(self, train, test):
        try:
            numerical_columns = self.numeric_columns
            categorical_columns = self.categorical_columns

            if self.target_column in categorical_columns:
                categorical_columns.remove(self.target_column)

            logger.info(f"Numerical columns: {list(numerical_columns)}")
            logger.info(f"Categorical columns: {list(categorical_columns)}")

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns)
                ],
                remainder="passthrough"
            )

            train_x = train.drop(columns=[self.target_column])
            test_x = test.drop(columns=[self.target_column])
            train_y = train[self.target_column]
            test_y = test[self.target_column]

            train_processed = preprocessor.fit_transform(train_x)
            test_processed = preprocessor.transform(test_x)

            train_y = train_y.values.reshape(-1, 1)
            test_y = test_y.values.reshape(-1, 1)

            train_combined = np.hstack((train_processed, train_y))
            test_combined = np.hstack((test_processed, test_y))

            joblib.dump(preprocessor, self.config.preprocessor_path)
            logger.info(f"Preprocessor saved at {self.config.preprocessor_path}")

            train_processed_path = os.path.join(self.config.root_dir, "train_processed.npy")
            test_processed_path = os.path.join(self.config.root_dir, "test_processed.npy")
            
            np.save(train_processed_path, train_combined)
            np.save(test_processed_path, test_combined)

            logger.info("Preprocessed train and test data saved successfully.")
            return train_processed, test_processed

        except Exception as e:
            logger.error(f"Error in preprocess_features: {e}")
            raise e