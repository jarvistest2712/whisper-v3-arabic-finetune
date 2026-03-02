import torch
import re
import os
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
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
DATASET_PATH = "./merged_whisper_dataset_processed_448"
SAMPLE_COUNT = 100 

# --- 1. ARABIC NORMALIZATION ---
def normalize_arabic(text):
    if not text: return ""
    # Remove diacritics
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # Standardize Alifs (أ، إ، آ -> ا)
    text = re.sub(r"[أإآ]", "ا", text)
    # Standardize Yaa (ى -> ي)
    text = re.sub(r"ى", "ي", text)
    # Standardize Te Marbuta (ة -> ه)
    text = re.sub(r"ة", "ه", text)
    # Clean redundant whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 # Using FP16 for better stability during inference on L40S

    # 2. Load Model and Processor
    logger.info("Loading base model...")
    if not os.path.exists(BASE_MODEL):
        logger.error(f"Base model not found at {BASE_MODEL}")
        return

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True
    )

    logger.info(f"Merging LoRA Adapter from: {CHECKPOINT}")
    if not os.path.exists(CHECKPOINT):
        logger.error(f"Checkpoint not found at {CHECKPOINT}")
        return
        
    # Crucial for preventing 'vectorized_gather_kernel' index errors:
    model = PeftModel.from_pretrained(model, CHECKPOINT)
    model = model.merge_and_unload() 
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    # 3. Load Test Data
    logger.info("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)

    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        test_data = dataset["test"] if "test" in dataset else dataset[next(iter(dataset.keys()))]
    else:
        test_data = dataset

    test_data = test_data.select(range(min(SAMPLE_COUNT, len(test_data))))

    # 4. Preparation for Generation
    # Force Arabic language IDs to prevent cross-language index errors
    forced_decoder_ids = processor.get_forced_decoder_ids(language="arabic", task="transcribe")

    references = []
    predictions = []

    print(f"\n{'='*20} ARABIC WER EVALUATION (ROBUST MODE) {'='*20}\n")

    for i, row in enumerate(tqdm(test_data)):
        try:
            # Process audio input
            audio_array = row["audio"]["array"]
            input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device).to(torch_dtype)

            # 5. Manual Generation (More stable than pipeline for Whisper v3)
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=225,
                    num_beams=1,
                    use_cache=True,
                    return_timestamps=False # CRITICAL: Disabling this fixes the gather kernel error
                )
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Normalize and store
            ref_text = row.get("text", row.get("sentence", ""))
            ref = normalize_arabic(ref_text)
            pred = normalize_arabic(transcription)
            
            references.append(ref)
            predictions.append(pred)
            
            if i < 5:
                print(f"\nSample #{i+1}")
                print(f"REF : {ref}")
                print(f"PRED: {pred}")
                print("-" * 20)

        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            continue

    # 6. Final Report
    if not references:
        logger.error("No samples were successfully processed.")
        return

    final_wer = 100 * wer(references, predictions)
    
    print(f"\n{'='*50}")
    print(f"TOTAL SAMPLES PROCESSED: {len(references)}")
    print(f"FINAL WORD ERROR RATE (WER): %{final_wer:.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
