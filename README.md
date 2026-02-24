# Whisper v3-Large Arabic Fine-Tuning (LoRA)

This repository provides a high-performance training script for fine-tuning OpenAI's **Whisper v3-Large** specifically for Arabic dialects using **PEFT/LoRA**.

## Features
- **Optimized for Multi-GPU:** Designed to run on 4x NVIDIA L40S (or similar) using Hugging Face `accelerate`.
- **Memory Efficient:** Uses 8-bit quantization and LoRA to fit the large model comfortably while maintaining performance.
- **High-Performance Training:** Leverages `BF16` precision and multi-process data loading.
- **Dialect Support:** Configured to handle large-scale, multi-dialectal datasets (1.4M+ samples).

## Requirements
- Python 3.10+
- 4x GPUs (e.g., L40S 46GB)
- ~512GB+ System RAM (for large dataset memory mapping)

### Installation
```bash
pip install transformers datasets peft accelerate bitsandbytes sentencepiece librosa soundfile
```

## Setup
1. Place your model in `./whisper-v3-large-local`.
2. Place your Hugging Face formatted dataset in `./arabic_dataset_hf`.
3. Configure `accelerate`:
   ```bash
   accelerate config
   ```
   (Select Multi-GPU, 4 devices, and BF16 precision).

## Usage
Start the training across all 4 GPUs:
```bash
accelerate launch train.py
```

## Configuration
Adjust the following parameters in `train.py` based on your specific setup:
- `BATCH_SIZE`: Set to 32 per GPU for L40S.
- `NUM_PROC`: Number of CPU cores for data preprocessing.
- `MAX_STEPS`: Total training steps for 1.4M rows.

## License
MIT
