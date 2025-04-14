import os
import joblib   
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from project import logger
from project.entity.config_entity import ModelTrainerConfig






class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

        mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/E2E-Credit-Fraud-Detection.mlflow")
        mlflow.set_experiment("E2E-Credit-Fraud-Detection")

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