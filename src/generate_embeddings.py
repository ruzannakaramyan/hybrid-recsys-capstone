import os
import argparse
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from dataset import SequentialDataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate LLM Embeddings for Items")
    parser.add_argument("--dataset", type=str, default="industrial_and_scientific")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--batch_size", type=int, default=256, help="Encoding batch size")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    train_file = os.path.join(data_dir, f"train_{args.dataset}_merged.csv")
    out_file = os.path.join(data_dir, f"item_embeddings_{args.dataset}.pt")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer '{args.model}' to {device}...")
    st_model = SentenceTransformer(args.model, device=device)

    print("Building exact item vocabulary...")
    temp_dataset = SequentialDataset(train_file, max_seq_len=2)
    item_vocab = temp_dataset.item_vocab  # mapping: item_str -> int (1 to V)
    vocab_size = len(item_vocab) + 1      # index 0 is padding

    print("Extracting metadata...")
    df = pd.read_csv(train_file)
    
    # We will build a text mapping dictionary for every known item
    item_texts = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing rows"):
        asin = str(row['parent_asin'])
        
        title = str(row['title']) if pd.notna(row['title']) else ""
        desc = str(row['description']) if pd.notna(row['description']) else ""
        combined = (title + " " + desc).replace("nan ", "").strip()
        
        # In case an item is only seen in "history", we could try to look back, 
        # but the merge logic implies metadata was broadcasted.
        if asin not in item_texts or len(combined) > len(item_texts.get(asin, "")):
            item_texts[asin] = combined

    print(f"Extracted metadata for {len(item_texts)} items out of {vocab_size-1} vocab.")

    # We build the ordered list of strings corresponding to exact vocab indices
    # index 0 is padding and stays empty
    ordered_sentences = [""] * vocab_size
    missing_count = 0

    for item_str, idx in item_vocab.items():
        text = item_texts.get(item_str, "")
        if not text:
            missing_count += 1
            text = "" # Fallback
        ordered_sentences[idx] = text

    print(f"Missing text for {missing_count} background history items (they will get zero-like dense vectors).")
    
    print("Encoding sentences (this may take a minute depending on batch size)...")
    embeddings = st_model.encode(
        ordered_sentences, 
        batch_size=args.batch_size, 
        show_progress_bar=True, 
        convert_to_tensor=True
    ).cpu()

    # ensure index 0 (padding) is explicitly purely zeros just in case model encoded "" as non-zero
    embeddings[0] = 0.0

    print(f"Generated embedding matrix shape: {embeddings.shape}")
    
    torch.save(embeddings, out_file)
    print(f"✅ Successfully written to {out_file}")

if __name__ == "__main__":
    main()
