import torch
import re
import os
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from peft import PeftModel
from datasets import load_from_disk
from jiwer import wer
from tqdm import tqdm

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WER_Tester")

# --- SETTINGS ---
BASE_MODEL = "./whisper-v3-large-local"
CHECKPOINT = "./whisper-v3-arabic-lora-bf16-output/checkpoint-25000"
DATASET_PATH = "./merged_whisper_dataset_processed_448" # Ensure this has the 'test' split with 'audio' and 'text'
SAMPLE_COUNT = 100 # Adjust number of samples to test

# --- 1. ARABIC NORMALIZATION ---
def normalize_arabic(text):
    if not text: return ""
    # Remove diacritics
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # Standardize Alifs
    text = re.sub(r"[أإآ]", "ا", text)
    # Standardize Yaa
    text = re.sub(r"ى", "ي", text)
    # Standardize Te Marbuta
    text = re.sub(r"ة", "ه", text)
    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    # 2. Load Model and Pipeline
    logger.info("Loading Whisper v3-Large in BF16 mode...")
    if not os.path.exists(BASE_MODEL):
        logger.error(f"Base model not found at {BASE_MODEL}")
        return

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    logger.info(f"Injecting LoRA Adapter from: {CHECKPOINT}")
    if not os.path.exists(CHECKPOINT):
        logger.error(f"Checkpoint not found at {CHECKPOINT}")
        return
        
    model = PeftModel.from_pretrained(model, CHECKPOINT)
    model = model.merge_and_unload() # Permanently merge for faster inference

    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.bfloat16,
        device="cuda:0"
    )

    # 3. Load Test Data
    logger.info("Loading dataset from disk...")
    dataset = load_from_disk(DATASET_PATH)

    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        if "test" in dataset:
            test_data = dataset["test"]
        else:
            logger.warning("'test' split not found, using 'train' for evaluation.")
            test_data = dataset[next(iter(dataset.keys()))]
    else:
        test_data = dataset

    # Select samples
    test_data = test_data.select(range(min(SAMPLE_COUNT, len(test_data))))

    # 4. Inference Loop
    references = []
    predictions = []

    print(f"\n{'='*20} ARABIC WER EVALUATION STARTED {'='*20}\n")

    for i, row in enumerate(tqdm(test_data)):
        audio_sample = row["audio"]
        
        # Run Inference
        # Force Arabic language and transcribe task
        result = pipe(
            audio_sample,
            generate_kwargs={
                "language": "arabic", 
                "task": "transcribe", 
                "num_beams": 5
            }
        )
        
        # Normalize both reference and prediction
        # Try 'text' column, fallback to 'sentence'
        ref_text = row.get("text", row.get("sentence", ""))
        ref = normalize_arabic(ref_text)
        pred = normalize_arabic(result["text"])
        
        references.append(ref)
        predictions.append(pred)
        
        # Log first 5 results for visual check
        if i < 5:
            print(f"\nSample #{i+1}")
            print(f"REF : {ref}")
            print(f"PRED: {pred}")
            print("-" * 20)

    # 5. Calculate Final WER
    if not references:
        logger.error("No data processed. Evaluation failed.")
        return

    final_wer = 100 * wer(references, predictions)
    
    print(f"\n{'='*50}")
    print(f"TOTAL SAMPLES TESTED: {len(references)}")
    print(f"FINAL WORD ERROR RATE (WER): %{final_wer:.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
