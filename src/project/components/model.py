import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import urllib.request as request
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from project import logger
from project.utils.common import *
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from project.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    def download_file(self):
        if not os.path.exists(self.config.zip_file):
            filename, headers = request.urlretrieve(
            url = self.config.source_url,
            filename = self.config.zip_file)
            logger.info(f'{filename} downlod with following information: \n{headers}')
        else:
            logger.info(f"File already exixts of the size: {get_size(Path(self.config.zip_file))}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_file
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)





class DataValidation:
    def __init__(self, config):
        self.config = config

    def validation(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_file)
            all_cols = list(data.columns)
            expected_cols = set(self.config.all_schema.keys())  

            validation_status = set(all_cols).issubset(expected_cols)

            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation_status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.columns_to_drop = config.columns_to_drop
        self.target_column = config.target_column.lower()
        self.label_encoders = {}
        
        self.categorical_columns = [col.lower() for col in config.categorical_columns]
        self.numerical_columns = [col.lower() for col in config.numerical_columns]
        

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.copy()

            data.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')
            
            data.columns = [col.lower() for col in data.columns]
                        
            
            for column in self.categorical_columns:
                col_matches = [col for col in data.columns if col.lower() == column.lower()]
                if col_matches:
                    actual_col = col_matches[0]
                    le = LabelEncoder()
                    data[actual_col] = le.fit_transform(data[actual_col].astype(str))
                    self.label_encoders[actual_col] = le
            
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
            
            train, test = train_test_split(resampled_data, test_size= self.config.test_size, random_state=self.config.random_state)
            
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
            numerical_columns = self.numerical_columns
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
        




class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

        # Initialize MLflow tracking
        os.environ['MLFLOW_TRACKING_USERNAME'] = "JavithNaseem-J"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = ""
        
        dagshub.init(repo_owner="JavithNaseem-J", repo_name="Flight-Fare-Price-Prediction")
        mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/Flight-Fare-Price-Prediction.mlflow")
        mlflow.set_experiment("Flight-Fare-Price-Prediction")

    def train(self):
        

        # Validate file paths
        if not os.path.exists(self.config.train_preprocess):
            logger.error(f"Train preprocessed file not found at: {self.config.train_preprocess}")
            raise FileNotFoundError("Train preprocessed file not found")
        if not os.path.exists(self.config.test_preprocess):
            logger.error(f"Test preprocessed file not found at: {self.config.test_preprocess}")
            raise FileNotFoundError("Test preprocessed file not found")

        # Load preprocessed data
        train_data = np.load(self.config.train_preprocess, allow_pickle=True)
        test_data = np.load(self.config.test_preprocess, allow_pickle=True)

        logger.info(f'Loaded train and test data')
        logger.info(f'Train data shape: {train_data.shape}')
        logger.info(f'Test data shape: {test_data.shape}')

        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]

        logger.info(f"Training data shape: X={train_x.shape}, y={train_y.shape}")
        logger.info(f"Testing data shape: X={test_x.shape}, y={test_y.shape}")

        # Initialize MLflow experiment
        mlflow.set_experiment("Credit-Fraud-Detection")
        mlflow.xgboost.autolog()  
        with mlflow.start_run(run_name="RandomizedSearchCV_Tuning"):
            mlflow.set_tag("run_type", "hyperparameter_tuning")
            mlflow.set_tag("model", "XGBClassifier")

            logger.info('Initializing Randomized Search')

            xgb_model = XGBClassifier(
                objective='binary:logistic',
                verbosity=0,
                eval_metric='logloss'
            )

            param_dist = self.config.random_search_params

            logger.info('>>>>>>>>>> ......Performing Randomized Search - this may take some time...... <<<<<<<<<')


            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_dist,
                n_iter=self.config.n_iter,
                cv=self.config.cv_folds,
                scoring='accuracy',
                verbose=1,
                n_jobs=self.config.n_jobs,
                return_train_score=True
            )
            random_search.fit(train_x, train_y)

            for i, (params, mean_score, std_score) in enumerate(
                zip(
                    random_search.cv_results_["params"],
                    random_search.cv_results_["mean_test_score"],
                    random_search.cv_results_["std_test_score"]
                )
            ):
                with mlflow.start_run(nested=True, run_name=f"Trial_{i+1}"):
                    mlflow.set_tag("trial_number", i + 1)
                    mlflow.log_params(params)
                    mlflow.log_metric("mean_accuracy", mean_score)
                    mlflow.log_metric("std_accuracy", std_score)  
                    logger.info(f"Trial {i+1}: params={params}, mean_accuracy={mean_score:.4f}, std_accuracy={std_score:.4f}")


            best_model = random_search.best_estimator_
            mlflow.xgboost.log_model(
                xgb_model=best_model,
                artifact_path="xgboost_model",
                registered_model_name="XGBClassifier_CreditFraud"
            )
            logger.info("Best model logged to MLflow")

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            joblib.dump(random_search, model_path)
            logger.info(f'Model saved locally at {model_path}')
    



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        

    def evaluation(self):
        # Validate file paths
        if not os.path.exists(self.config.test_path):
            raise FileNotFoundError(f"Test data file not found at {self.config.test_path}")
        if not os.path.exists(self.config.preprocess_path):
            raise FileNotFoundError(f"Preprocessor file not found at {self.config.preprocess_path}")
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found at {self.config.model_path}")


        # Load preprocessor and model
        logger.info("Loading preprocessor and model...")
        preprocessor = joblib.load(self.config.preprocess_path)
        model = joblib.load(self.config.model_path)

        # Extract best estimator if model is RandomizedSearchCV
        if hasattr(model, 'best_estimator_'):
            logger.info("Model is a RandomizedSearchCV object, extracting best estimator...")
            best_params = model.best_params_
            model = model.best_estimator_
        else:
            best_params = model.get_params()
            logger.info("Model is a direct estimator, using its parameters...")


        # Load test and train data
        test_data = pd.read_csv(self.config.test_path)
        target_column = self.config.target_column.lower()

        if target_column not in test_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in test data.")

        # Prepare test and train data
        test_x = test_data.drop(columns=[target_column])
        test_y = test_data[target_column]

        test_x_preprocessed = preprocessor.transform(test_x)

        # Make predictions
        test_predictions = model.predict(test_x_preprocessed)

        # Get predicted probabilities for ROC
        test_probabilities = model.predict_proba(test_x_preprocessed)[:, 1]
        # Calculate metrics
        metrics = {
                "test_accuracy": accuracy_score(test_y, test_predictions),
                "test_precision_weighted": precision_score(test_y, test_predictions, average='weighted'),
                "test_recall_weighted": recall_score(test_y, test_predictions, average='weighted'),
                "test_f1_weighted": f1_score(test_y, test_predictions, average='weighted'),
                "test_auc": auc(*roc_curve(test_y, test_probabilities)[:2])
        }

        
        logger.info(f"Model evaluation metrics: {metrics}")

        # Log confusion matrix
        cm = confusion_matrix(test_y, test_predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        cm_path = Path(self.config.root_dir)/"cm.png"
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved and logged at {cm_path}")

        # Log ROC curve
        fpr, tpr, _ = roc_curve(test_y, test_probabilities)
        roc_auc = metrics["test_auc"]
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--") 
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        roc_path = Path(self.config.root_dir)/"roc.png"
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curve saved at {roc_path}")


        # Save and log metrics
        metrics_file = Path(self.config.root_dir) / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        logger.info(f"Metrics saved to {metrics_file}")


        return metrics