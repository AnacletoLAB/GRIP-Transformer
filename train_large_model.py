import warnings

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from datasets import Dataset
# from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

from data import build_datasets
from tokenizer import tokenizer, tokenize_batch
from log_callback import GenerationLoggerCallback
from model import NucConfig, NucTransformer
from utils import set_seed

warnings.filterwarnings("ignore", category=FutureWarning, message="`tokenizer` is deprecated")
warnings.filterwarnings("ignore", message="mtime may not be reliable on this filesystem")

def main(
    train_file="data/dry_run/all_train_aug.tsv", valid_file="data/dry_run/all_test_noaug.tsv",
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

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=early_stop),
        GenerationLoggerCallback(),
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

if __name__ == "__main__":
    main()
