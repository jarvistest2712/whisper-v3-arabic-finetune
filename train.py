import os
import torch
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import PartialState

# --- CONFIGURATION (ULTRA-SLIM VERSION) ---
MODEL_PATH = "./whisper-v3-large-local"  # Local model path
DATASET_PATH = "./merged_whisper_dataset_processed_448" 
OUTPUT_DIR = "./whisper-v3-arabic-lora-output"
BATCH_SIZE = 32                          # Per-GPU batch size
GRADIENT_ACCUMULATION = 1                
LEARNING_RATE = 1e-4
MAX_STEPS = 100000                       
WARMUP_STEPS = 1000

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load Pre-processed Dataset
    logger.info(f"Loading pre-processed dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    
    logger.info(f"Dataset columns: {dataset.column_names}")

    logger.info("Splitting dataset into train/test...")
    dataset = dataset.train_test_split(test_size=0.01, seed=42)

    # 2. Load Processor and Model
    logger.info("Loading Whisper assets...")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="Arabic", task="transcribe")
    
    # Distributed training fix: Use PartialState to get current device index
    device_string = PartialState().process_index
    
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        # device_map="auto" is NOT allowed in distributed training
        device_map={"": device_string} 
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

    # 4. Data Collator
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

    # 5. Training Arguments
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
        ddp_find_unused_parameters=False, # Optimization for DDP
    )

    # 6. Start Training
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    logger.info("Starting distributed training...")
    trainer.train()

if __name__ == "__main__":
    main()
