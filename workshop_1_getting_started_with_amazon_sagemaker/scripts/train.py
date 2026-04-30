import argparse
import logging
import os
import sys
from typing import Tuple, Dict

import numpy as np
from datasets import load_from_disk, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# Set up logging
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parses the command-line arguments passed by SageMaker or the user.

    Returns:
        argparse.Namespace: Parsed arguments containing hyperparameters and directory paths.
    """
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model ID to load from HuggingFace Hub",
    )
    parser.add_argument("--learning_rate", type=str, default="5e-5")
    parser.add_argument("--fp16", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output_data"),
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )
    parser.add_argument(
        "--n_gpus", type=str, default=os.environ.get("SM_NUM_GPUS", "0")
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "./data/train"),
    )
    parser.add_argument(
        "--test_dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", "./data/test")
    )

    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    """
    Main function to execute the training script.
    Loads datasets, configures the HuggingFace Trainer, and runs training and evaluation.
    """
    args = parse_args()

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading datasets...")
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f"Loaded train_dataset length: {len(train_dataset)}")
    logger.info(f"Loaded test_dataset length: {len(test_dataset)}")

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Computes evaluation metrics.

        Args:
            eval_pred (Tuple[np.ndarray, np.ndarray]): Predictions and labels.

        Returns:
            Dict[str, float]: Dictionary containing computed metrics.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Prepare model labels - useful in inference API
    labels = train_dataset.features["labels"].names
    num_labels = len(labels)
    label2id: Dict[str, str] = {}
    id2label: Dict[str, str] = {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    logger.info(f"Downloading/Loading model: {args.model_id}")
    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True
        if get_last_checkpoint(args.output_dir) is not None
        else False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # train model
    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint is not None:
        logger.info("***** Continue training from checkpoint *****")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("***** Starting new training *****")
        trainer.train()

    # evaluate model
    logger.info("***** Evaluating model *****")
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print("***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")
            print(f"{key} = {value}\n")

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    logger.info("Saving the model...")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
