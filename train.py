import os
import torch
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # usa la seconda GPU

from datasets import Dataset
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, TrainerCallback

from data import build_datasets, load_pairs_tsv
# from tokenizer import tokenizer, tokenize_batch  # usa tokenizer con BOS
from tokenizer import tokenizer, tokenize_batch, detokenize_batch   # questo per l'ultima versione di 1-mer
from log_callback import GenerationLoggerCallback
from model import NucConfig, NucTransformer
from utils import set_seed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`tokenizer` is deprecated")
warnings.filterwarnings("ignore", message="mtime may not be reliable on this filesystem")

class DebugGenerazioneCallback(TrainerCallback):
    def __init__(self, model, tokenizer, sample_batch, device):
        self.model = model
        self.tokenizer = tokenizer
        self.sample_batch = sample_batch
        self.device = device

    def on_epoch_end(self, args, state, control, **kwargs):
        print()
        self.model.eval()
        with torch.no_grad():
            tokenized = self.tokenizer(self.sample_batch["source"], padding=True, return_tensors="pt").to(self.device)
            input_ids = tokenized["input_ids"]

            def reconstruct_from_3mers(kmers):
                tokens = kmers.strip().split()
                if not tokens:
                    return ""
                seq = tokens[0]
                for token in tokens[1:]:
                    seq += token[-1]
                return seq

            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                min_new_tokens=12,
                temperature=1.0,
                top_k=10,
            )
            generated_seq = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # reconstructed_seq = reconstruct_from_3mers(generated_seq).replace("P", "[STOP]")
            reconstructed_seq = generated_seq.replace(" ", "")

            print("\n" + "=" * 60)
            print(f"Epoca {int(state.epoch)} → Generazione di esempio")
            print("." * 60)
            print(f"Input  (seq1)         : {self.sample_batch['source'][0]}")
            print(f"Target (seq2)         : {self.sample_batch['target'][0]}")
            print(f"Generato (3-mer)      : {generated_seq}")
            print(f"Ricostruito (nt)      : {reconstructed_seq}")
            print(f"Lunghezza gen. (token): {generated_ids.shape[1]}")
            print("=" * 60)

def main(
    # dataset train_augmented.tsv ha 75380 coppie, test_augmented.tsv ha 18848 coppie
    # ho aumentato da 32 a 48 la batch size
    train_file="data/dry_run/all_train_augmented.tsv", valid_file="data/dry_run/all_test_noaug.tsv",
    output_dir="ckpt/dry_run", logging_dir="ckpt/dry_run/logs",
    learning_rate=3e-5, train_batch_size=48, eval_batch_size=48,
    num_epochs=150, bf16=True, grad_acc_steps=2, warmup_steps=1000,
    scheduler="cosine", early_stop=30, num_workers=8, seed=42 
):
    set_seed(seed)

    # Dataset
    train_ds, valid_ds = build_datasets(train_file, valid_file)
    train_ds = train_ds.map(tokenize_batch, batched=True)
    valid_ds = valid_ds.map(tokenize_batch, batched=True)

    # Config e modello
    config = NucConfig()
    config.pad_token_id = tokenizer.vocab["PAD"]
    config.eos_token_id = tokenizer.vocab["EOS"]
    config.vocab_size = len(tokenizer.vocab)
    print(f"VOCAB SIZE: {len(tokenizer.vocab)}")
    print(f"VOCAB KEYS: {list(tokenizer.vocab.keys())[:10]} ...")

    model = NucTransformer(config)
    model.print_model_params()

    # Parametri training
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir, logging_dir=logging_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=learning_rate, num_train_epochs=num_epochs,
        bf16=bf16, warmup_steps=warmup_steps, lr_scheduler_type=scheduler,
        eval_strategy="epoch", save_strategy="epoch",
        logging_strategy="epoch", report_to=["tensorboard"],
        dataloader_num_workers=num_workers, save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    device = torch.device("cuda")

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=early_stop),
        GenerationLoggerCallback(),
        DebugGenerazioneCallback(model, tokenizer, sample_batch, device)
    ]

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=valid_ds,
        tokenizer=tokenizer, callbacks=callbacks
    )

    trainer.train()
    print(trainer.evaluate())
    print()
    trainer.save_model("ckpt/dry_run/final_model")

    for name in ["no_match_validation.tsv", "validation_final.tsv", "no_match_test.csv", "test_augmented.tsv"]:
        filename = os.path.join("data/dry_run", name)
        sep = "\t" if filename.endswith(".tsv") else ";"
        df_final = pd.read_csv(filename, sep=sep)
        print(f"Validazione finale su {name}")
        final_ds = Dataset.from_pandas(df_final).map(tokenize_batch, batched=True)
        final_metrics = trainer.evaluate(eval_dataset=final_ds)
        print(f"Risultati validazione finale su {name}:", final_metrics)
        print()

if __name__ == "__main__":
    main()
