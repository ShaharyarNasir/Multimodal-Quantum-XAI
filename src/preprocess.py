import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def preprocess_data(csv_path="data/queries.csv"):
    try:
        logger.info("Loading CSV...")
        # Load CSV with robust parsing
        df = pd.read_csv(csv_path, on_bad_lines='skip', quotechar='"')
        logger.info(f"Loaded {len(df)} rows from {csv_path}")

        # Validate columns
        expected_columns = ["query", "num_qubits", "max_time",
                            "num_points", "epochs", "decoherence_rate"]
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(
                f"CSV missing columns. Expected: {expected_columns}")

        # Force numeric columns and handle NaNs
        label_columns = ["num_qubits", "max_time",
                         "num_points", "epochs", "decoherence_rate"]
        defaults = {"num_qubits": 2, "max_time": 10,
                    "num_points": 100, "epochs": 1000, "decoherence_rate": 0.0}
        for col in label_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                logger.warning(
                    f"Found NaN in {col}. Filling with default: {defaults[col]}")
                df[col].fillna(defaults[col], inplace=True)
        # Drop rows with missing queries
        df.dropna(subset=["query"], inplace=True)
        logger.info(f"After cleaning, {len(df)} rows remain")

        # Initialize tokenizer
        logger.info("Initializing DistilBERT tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")

        # Tokenization function
        def tokenize_function(examples):
            # Reduced for speed
            return tokenizer(examples["query"], padding="max_length", truncation=True, max_length=16)

        # Convert to HF Dataset and tokenize
        logger.info("Converting to Dataset and tokenizing...")
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(tokenize_function, batched=True,
                              batch_size=200)  # Increased for speed

        # Prepare labels
        def format_labels(examples):
            examples["labels"] = [
                float(examples["num_qubits"]),
                float(examples["max_time"]),
                float(examples["num_points"]),
                float(examples["epochs"]),
                float(examples["decoherence_rate"])
            ]
            return examples

        logger.info("Formatting labels...")
        # Single-pass for labels
        dataset = dataset.map(format_labels, batched=False)

        # Set format for PyTorch
        dataset.set_format("torch", columns=[
                           "input_ids", "attention_mask", "labels"])
        logger.info(f"Processed dataset with {len(dataset)} samples")

        # Save outputs
        logger.info("Saving dataset and tokenizer...")
        dataset.save_to_disk("data/processed_dataset")
        tokenizer.save_pretrained("data/tokenizer")
        logger.info("Preprocessing complete!")
        return dataset, tokenizer, df
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    dataset, tokenizer, df = preprocess_data()
    logger.info(f"Saved dataset with {len(dataset)} samples")
