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
DATASET_PATH = "./arabic_dataset_hf"     # Local dataset path
OUTPUT_DIR = "./whisper-v3-arabic-lora-output"
BATCH_SIZE = 32                          # Per-GPU batch size (L40S has 46GB VRAM)
GRADIENT_ACCUMULATION = 1                # Increase if you face OOM
LEARNING_RATE = 1e-4
MAX_STEPS = 100000                       # Total training steps
WARMUP_STEPS = 1000
NUM_PROC = 16                            # Multiprocessing for data preparation

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load and Split Dataset
    logger.info("Loading dataset from disk...")
    # load_from_disk uses memory mapping, 755GB RAM is plenty for 1.4M rows
    dataset = load_from_disk(DATASET_PATH)
    
    logger.info("Splitting dataset into train/test...")
    dataset = dataset.train_test_split(test_size=0.01, seed=42) # 1% test is ~14k samples, enough for eval
    
    # Ensure 16kHz
    logger.info("Casting audio column to 16kHz...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # 2. Load Processor and Model
    logger.info(f"Loading processor and model from {MODEL_PATH}...")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="Arabic", task="transcribe")
    
    # L40S supports BF16 which is more stable than FP16
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map="auto"
    )

    # 3. PEFT/LoRA Setup
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 4. Data Preparation
    def prepare_dataset(batch):
        # Transcribe audio to input features
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # Tokenize target text
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    logger.info("Preprocessing dataset (mapping)...")
    # Using num_proc for faster execution
    dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset.column_names["train"], 
        num_proc=NUM_PROC
    )

    # 5. Data Collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding with -100 to ignore loss
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 6. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        bf16=True,               # Use BF16 for L40S
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
        dataloader_num_workers=8, # Faster data loading
    )

    # 7. Start Training
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    logger.info("Starting training session...")
    trainer.train()

if __name__ == "__main__":
    main()
