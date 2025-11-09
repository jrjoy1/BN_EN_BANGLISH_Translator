#!/usr/bin/env python3
"""bn_to_banglish/main.py

Full script for training, translating, evaluating, and saving a MarianMT model.
"""

import os
import argparse
from typing import List
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import sacrebleu


def load_dataframe(path: str, text_col: str = "bangla", ref_col: str = "banglish") -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = df.rename(columns={'text': text_col, 'text_banglish': ref_col})
    df = df[[text_col, ref_col]]
    return df


def preprocess_dataset(df: pd.DataFrame, tokenizer, max_length: int = 64) -> Dataset:
    dataset = Dataset.from_pandas(df)

    def preprocess(examples):
        model_inputs = tokenizer(examples['bangla'], max_length=max_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['banglish'], max_length=max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess, batched=True)
    return tokenized_dataset


def translate_batch(sentences: List[str], model, tokenizer, device, batch_size=16) -> List[str]:
    translations = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        translated = model.generate(**inputs)
        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        translations.extend(decoded)
    return translations


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    refs = [[r for r in references]]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return bleu.score


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Bengali to Banglish translation")
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file with text and text_banglish columns')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV with predictions')
    parser.add_argument('--model', type=str, default='Helsinki-NLP/opus-mt-bn-en', help='HuggingFace Marian model')
    parser.add_argument('--hf-token-env', type=str, default='HF_TOKEN', help='Env variable for HuggingFace token')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    hf_token = os.environ.get(args.hf_token_env, None)

    df = load_dataframe(args.input)
    print(f"Loaded dataset with shape {df.shape}")

    tokenizer = MarianTokenizer.from_pretrained(args.model, use_auth_token=hf_token)
    model = MarianMTModel.from_pretrained(args.model, use_auth_token=hf_token).to(device)

    tokenized_dataset = preprocess_dataset(df, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./bn_to_banglish',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        predict_with_generate=True,
        report_to=[]
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    trainer.train()

    # Translation & BLEU
    df_subset = df.head(200).copy()
    df_subset['predicted'] = translate_batch(df_subset['bangla'].tolist(), model, tokenizer, device, batch_size=args.batch_size)

    bleu_score = compute_bleu(df_subset['banglish'].tolist(), df_subset['predicted'].tolist())
    print(f"🌟 BLEU Score: {bleu_score}")

    df_subset.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    # Save model
    save_path = './bn_to_banglish_model'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()

