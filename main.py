import os
from project import logger
from project.pipeline.stage_01 import DataIngestionPipeline
from project.pipeline.stage_02 import DataValidationPipeline
from project.pipeline.stage_03 import DataTransformationPipeline
from project.pipeline.stage_04 import ModelTrainingPipeline
from project.pipeline.stage_05 import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion"

if __name__ == "__main__":
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.run()
    logger.info(f"<<<<< stage {STAGE_NAME} completed >>>>>>>")



STAGE_NAME = "Data Validation"

if __name__ == "__main__":
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
    data_validation = DataValidationPipeline()
    data_validation.run()
    logger.info(f"<<<<< stage {STAGE_NAME} completed >>>>>>>")




STAGE_NAME = "Data Transformation"


if __name__ == "__main__":
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
    data_transformation = DataTransformationPipeline()
    data_transformation.run()
    logger.info(f"<<<<< stage {STAGE_NAME} completed >>>>>>>")




STAGE_NAME = "Model Training"

if __name__ == "__main__":
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
    model_training = ModelTrainingPipeline()
    model_training.run()
    logger.info(f"<<<<< stage {STAGE_NAME} completed >>>>>>>")



STAGE_NAME = "Model Evaluation"

if __name__ == "__main__":
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.run()
    logger.info(f"<<<<< stage {STAGE_NAME} completed >>>>>>>")