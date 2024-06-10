# -*- coding: utf-8 -*-
"""This module is responsible for the evaluation stage of the pipeline. """
from se489_group_project import logger
from se489_group_project.components.model_evalution_mlflow import Evaluation
from se489_group_project.config.configuration import ConfigurationManager

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    """
    Class to run the evaluation stage of the pipeline
    """

    def __init__(self):
        """
        Initialization of the EvaluationPipeline class.
        """
        pass

    def main(self):
        """
        Main method to run the evaluation stage of the pipeline.
        Responsible for getting the evaluation configuration,
        running the evaluation, saving the score and logging
        the evaluation into mlflow.
        """

        # Initialize the configuration manager
        config = ConfigurationManager()
        # Get the evaluation configuration
        eval_config = config.get_evaluation_config()
        # Initialize
        evaluation = Evaluation(eval_config)
        # Run the evaluation
        evaluation.evaluation()
        # Save the score
        evaluation.save_score()
        # evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f"Evaluation stage started")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"Evaluation stage completed\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
