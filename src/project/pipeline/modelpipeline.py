from project import logger
from project.config.config import ConfigurationManager
from project.components.model import DataIngestion, DataValidation, DataTransformation, ModelTrainer, ModelEvaluation



class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()




class DataValidationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(data_validation_config)
        data_validation.validation()




class DataTransformationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train, test = data_transformation.train_test_splitting()
        train_processed, test_processed = data_transformation.preprocess_features(train, test)




class ModelTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_training_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()

        



class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluation()