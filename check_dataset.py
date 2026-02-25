import os
import logging
from datasets import load_from_disk

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetChecker")

DATASET_PATH = "./arabic_dataset_hf" # Dataset yolunuzu buraya yazın

def main():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset yolu bulunamadı: {DATASET_PATH}")
        return

    logger.info(f"Dataset yükleniyor: {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    
    # 1. Genel Yapıyı Kontrol Et
    print("\n--- 1. DATASET YAPISI ---")
    print(dataset)
    
    # 2. Kolon İsimlerini Yazdır
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        split_name = next(iter(dataset.keys()))
        cols = dataset[split_name].column_names
        sample_data = dataset[split_name][0]
    else:
        cols = dataset.column_names
        sample_data = dataset[0]
        
    print("\n--- 2. TESPİT EDİLEN KOLONLAR ---")
    print(cols)
    
    # 3. İlk Satırdan Örnek Bastır
    print("\n--- 3. İLK SATIR ÖRNEĞİ (İÇERİK) ---")
    for col in cols:
        content = sample_data[col]
        # Ses verisi çok uzun olduğu için sadece tipini ve uzunluğunu yazdıralım
        if col == "audio":
            print(f"[{col}]: {type(content)} - Örnekleme Hızı: {content.get('sampling_rate')}")
        else:
            print(f"[{col}]: {content}")

    print("\n--- 4. TAVSİYE EDİLEN TEXT_COLUMN ---")
    # Yaygın isimleri kontrol et
    possible_text_cols = ["sentence", "text", "transcription", "normalized_text"]
    found_text_col = [c for c in cols if c in possible_text_cols]
    if found_text_col:
        print(f"Tahmin edilen metin kolonu: '{found_text_col[0]}'")
    else:
        print("UYARI: Standart metin kolonu bulunamadı. Lütfen manuel seçin.")

if __name__ == "__main__":
    main()
