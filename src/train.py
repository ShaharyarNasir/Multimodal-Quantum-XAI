import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import mean_squared_error
import logging
import numpy as np
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions, multioutput='raw_values')
    return {
        "mse_num_qubits": mse[0],
        "mse_max_time": mse[1],
        "mse_num_points": mse[2],
        "mse_epochs": mse[3],
        "mse_decoherence": mse[4]
    }


def train_model():
    try:
        data_dir = os.path.join("data", "processed_dataset")
        model_dir = os.path.join("data", "model")
        log_dir = os.path.join("data", "logs")

        logger.info("Loading preprocessed dataset...")
        dataset = load_from_disk(data_dir)
        logger.info(f"Loaded {len(dataset)} samples")

        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        logger.info("Initializing DistilBERT model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=5
        )

        training_args = TrainingArguments(
            output_dir=model_dir,
            num_train_epochs=10,  # Increased from 3
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="mse_num_qubits",
            logging_dir=log_dir,
            logging_steps=10,
        )

        logger.info("Starting training...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()

        logger.info("Saving model...")
        model.save_pretrained(model_dir)
        return model
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    train_model()
