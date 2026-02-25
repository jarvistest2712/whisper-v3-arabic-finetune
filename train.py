import os
import torch
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- CONFIGURATION ---
MODEL_PATH = "./whisper-v3-large-local"  # Local model path
DATASET_PATH = "./arabic_dataset_hf"     # Local dataset path (merged output)
OUTPUT_DIR = "./whisper-v3-arabic-lora-output"
BATCH_SIZE = 32                          # Per-GPU batch size
GRADIENT_ACCUMULATION = 1                
LEARNING_RATE = 1e-4
MAX_STEPS = 100000                       
WARMUP_STEPS = 1000
NUM_PROC = 16                            

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load Dataset
    logger.info(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    
    # 2. Cleanup Extra Columns (Important for Merger compatibility)
    # The merger script adds 'metadata', 'audio', and 'text'. 
    # We must keep only 'audio' and 'text' for the prepare_dataset phase.
    # Everything else (like metadata) must go now to prevent trainer errors.
    initial_cols = dataset.column_names if not hasattr(dataset, "keys") else dataset[next(iter(dataset.keys()))].column_names
    logger.info(f"Detected columns: {initial_cols}")
    
    cols_to_drop = [c for c in initial_cols if c not in ["audio", "text"]]
    if cols_to_drop:
        logger.info(f"Removing non-essential columns: {cols_to_drop}")
        dataset = dataset.remove_columns(cols_to_drop)

    # 3. Train/Test Split
    logger.info("Splitting dataset...")
    dataset = dataset.train_test_split(test_size=0.01, seed=42)
    
    # Ensure 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # 4. Load Processor and Model
    logger.info("Loading Whisper assets...")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="Arabic", task="transcribe")
    
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map="auto"
    )

    # 5. PEFT/LoRA Setup
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)

    # 6. Data Preparation (Strict Cleaning)
    def prepare_dataset(batch):
        audio = batch["audio"]
        # Create input_features
        batch["input_features"] = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        # Create labels from the 'text' column produced by the merger
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    logger.info("Converting audio to features and tokenizing text...")
    # remove_columns here ensures only 'input_features' and 'labels' remain
    dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset["train"].column_names, 
        num_proc=NUM_PROC,
        load_from_cache_file=False,
        desc="Final feature extraction"
    )

    # 7. Data Collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 8. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        bf16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=2000,
        eval_steps=2000,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=8,
    )

    # 9. Start Training
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    logger.info("All systems nominal. Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
