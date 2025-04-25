import os
import joblib   
import numpy as np
import mlflow
import dagshub
import mlflow.xgboost
from xgboost import XGBClassifier
from project import logger
from project.entity.config_entity import ModelTrainerConfig
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna




class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        dagshub.init(repo_owner="JavithNaseem-J", repo_name="E2E-Credit-Fraud-Detection")
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

        # Optuna Optimization
        mlflow.xgboost.autolog()
        with mlflow.start_run(run_name="Optuna_HPO") as parent_run:
            parent_run_id = parent_run.info.run_id
            mlflow.set_tag("run_type", "hyperparameter_tuning")
            mlflow.set_tag("model", "XGBClassifier")

            def get_search_space(trial):
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                    "gamma": trial.suggest_float("gamma", 0.0, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                }

            def objective(trial):
                with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                    mlflow.set_tag("mlflow.parentRunId", parent_run_id)
                    mlflow.set_tag("trial_number", trial.number)

                    params = get_search_space(trial)

                    model = XGBClassifier(
                        objective='binary:logistic',
                        verbosity=0,
                        eval_metric='logloss',
                        use_label_encoder=False,
                        **params
                    )

                    cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, train_x, train_y, scoring=self.config.scoring, cv=cv, n_jobs=self.config.n_jobs)

                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()

                    mlflow.log_params(params)
                    mlflow.log_metric("cv_mean_accuracy", mean_score)
                    mlflow.log_metric("cv_std_accuracy", std_score)

                    logger.info(f"Trial {trial.number}: cv_mean_accuracy={mean_score:.4f} (+/- {std_score:.4f}), params={params}")

                    return mean_score

            logger.info('>>>>>>>>>> Starting Optuna Study <<<<<<<<<')

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.config.n_iter)

            logger.info(f"Best trial found: {study.best_trial.params} with accuracy {study.best_trial.value:.4f}")

            # Retrain best model on full training data
            best_params = study.best_trial.params
            best_model = XGBClassifier(
                objective='binary:logistic',
                verbosity=0,
                eval_metric='logloss',
                use_label_encoder=False,
                **best_params
            )
            best_model.fit(train_x, train_y)

            mlflow.xgboost.log_model(
                xgb_model=best_model,
                artifact_path="xgboost_model",
                registered_model_name="XGBClassifier_CreditFraud_Optuna"
            )
            logger.info("Best model logged to MLflow")

            # Save the model locally
            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            joblib.dump(best_model, model_path)
            logger.info(f'Best model saved locally at {model_path}')