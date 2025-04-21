import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from project import logger
from project.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from pathlib import Path



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        

    def evaluation(self):
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

        if hasattr(model, 'best_estimator_'):
            logger.info("Model is a RandomizedSearchCV object, extracting best estimator...")
            best_params = model.best_params_
            model = model.best_estimator_
        else:
            best_params = model.get_params()
            logger.info("Model is a direct estimator, using its parameters...")


        test_data = pd.read_csv(self.config.test_path)
        target_column = self.config.target_column

        if target_column not in test_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in test data.")

        test_x = test_data.drop(columns=[target_column])
        test_y = test_data[target_column]

        test_x_preprocessed = preprocessor.transform(test_x)

        test_predictions = model.predict(test_x_preprocessed)

        test_probabilities = model.predict_proba(test_x_preprocessed)[:, 1]

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