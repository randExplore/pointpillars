from config.kitti_config import cfg
from utils.training import Train
from eval.evaluate import Evaluation
from utils.testing import InferDataset


def main():

    # Dataset will be prepared for the first time to run, then it will be fast to directly start training
    trainer = Train(cfg)
    trainer.train()

    # Evaluate the training results by calculating mAP metrics based on official evaluation method
    evaluator = Evaluation(cfg)
    evaluator.evaluate()

    # Perform the test for one data from the validation dataset
    inference = InferDataset(cfg)
    inference()


if __name__ == "__main__":
    main()
