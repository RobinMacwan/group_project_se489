from se489_group_project.config.configuration import ConfigurationManager
from se489_group_project.components.model_evalution_mlflow import Evaluation
from se489_group_project import logger



STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        # evaluation.log_into_mlflow()




if __name__ == '__main__':
    try:
        logger.info(f"Evaluation stage started")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"Evaluation stage completed\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
            