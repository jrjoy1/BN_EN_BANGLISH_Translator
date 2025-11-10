# main.py (BN TO EN)

import argparse
from typing import List
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
import sacrebleu


import ast

def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    if 'translation' not in df.columns:
        raise ValueError("CSV must contain a 'translation' column")
    
    # Convert the string dictionary to an actual dictionary
    df['translation'] = df['translation'].apply(ast.literal_eval)
    
    # Extract bn and en columns
    df['bn'] = df['translation'].apply(lambda x: x['bn'])
    df['en'] = df['translation'].apply(lambda x: x['en'])
    
    # Keep only bn and en
    df = df[['bn', 'en']]
    return df


def translate_bn_to_en_batch(
    sentences: List[str], model, tokenizer, device, batch_size: int = 16
) -> List[str]:
    """Translate Bangla sentences to English in batches."""
    translations = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating"):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs)
        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        translations.extend(decoded)
    return translations


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Compute corpus BLEU score."""
    refs = [[r for r in references]]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return bleu.score


def main():
    parser = argparse.ArgumentParser(description="Translate Bengali to English")
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV with predictions')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--interactive', action='store_true', help='Enable interactive translation mode')
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = load_dataframe(args.input)
    print(f"Loaded dataset with shape {df.shape}")

    # Load MarianMT model
    model_name = "Helsinki-NLP/opus-mt-bn-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    # Translate first 200 rows for evaluation
    df_subset = df.head(20).copy()
    df_subset['predicted'] = translate_bn_to_en_batch(
        df_subset['bn'].tolist(), model, tokenizer, device, batch_size=args.batch_size
    )

    # Compute BLEU
    bleu_score = compute_bleu(df_subset['en'].tolist(), df_subset['predicted'].tolist())
    print(f"🌟 BLEU Score: {bleu_score}")

    # Display 10 random examples
    sample_df = df_subset.sample(10, random_state=42)
    for _, row in sample_df.iterrows():
        print(f"Bangla     : {row['bn']}")
        print(f"Reference  : {row['en']}")
        print(f"Predicted  : {row['predicted']}")
        print("---")

    mismatches = sum(1 for r, p in zip(df_subset['en'], df_subset['predicted']) if r != p)
    print(f"\nTotal mismatches in subset: {mismatches} / {len(df_subset)}")

    # Save predictions
    df_subset.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    # Interactive translation
    if args.interactive:
        print("\nType Bangla sentences to translate. Type 'exit' to quit.")
        while True:
            bangla_text = input("\nBangla: ")
            if bangla_text.lower() == 'exit':
                print("Exiting translator. ✅")
                break
            english_translation = translate_bn_to_en_batch(
                [bangla_text], model, tokenizer, device, batch_size=1
            )
            print("English:", english_translation[0])


if __name__ == '__main__':
    main()