from transformers import PreTrainedTokenizer

class NucTokenizer(PreTrainedTokenizer):
    """
    Tokenizer minimale per RNA con tokenizzazione 1-mer e simboli speciali.
    """
    def __init__(self):
        vocab = ["PAD", "EOS", "UNK", "A", "U", "C", "G"]
        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}

        super().__init__(
            pad_token="PAD",
            eos_token="EOS",
            unk_token="UNK",
            pad_token_id=self.vocab["PAD"],
            eos_token_id=self.vocab["EOS"],
            unk_token_id=self.vocab["UNK"],
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def added_tokens_encoder(self):
        return {}

    def _tokenize(self, text: str):
        return list(text)

    def _convert_token_to_id(self, token: str):
        return self.vocab.get(token, self.vocab["EOS"])

    def _convert_id_to_token(self, index: int):
        return self.id_to_token.get(index, "EOS")

    def get_vocab(self):
        return dict(self.vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        return []

    def save_pretrained(self, save_directory: str, **kwargs):
        return []


# === Funzioni di supporto ===
tokenizer = NucTokenizer()

def tokenize_batch(batch):
    if "source" in batch and "target" in batch:
        src_key = "source"
        tgt_key = "target"

    elif "seq1" in batch and "seq2" in batch:
        src_key = "seq1"
        tgt_key = "seq2"
    
    elif "RNA_sequence_x" in batch and "RNA_sequence_y" in batch:
        src_key = "RNA_sequence_x"
        tgt_key = "RNA_sequence_y"
    
    else:
        raise KeyError(
            f"Colonne non riconosciute. Trovate: {list(batch.keys())}. "
            "Attese: source/target, seq1/seq2 oppure RNA_sequence_x/RNA_sequence_y."
        )

    src_raw_texts = ["".join(x) if isinstance(x, list) else x for x in batch[src_key]]
    tgt_raw_texts = ["" + "".join(x) + " EOS" if isinstance(x, list) else "" + x + " EOS" for x in batch[tgt_key]]

    src_enc = tokenizer(src_raw_texts, padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
    tgt_enc = tokenizer(tgt_raw_texts, padding="max_length", truncation=True, max_length=1024, return_tensors="pt", add_special_tokens=True)

    decoder_input_ids = tgt_enc["input_ids"][:, :-1]
    labels = tgt_enc["input_ids"][:, 1:]

    return {
        "input_ids": src_enc["input_ids"],
        "attention_mask": src_enc["attention_mask"],
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }


def detokenize_batch(batch_ids):
    seqs = []
    id2tok = tokenizer.id_to_token

    for ids in batch_ids:
        tokens = [id2tok.get(i, "") for i in ids if id2tok.get(i) not in ["PAD", "EOS"]]
        seqs.append("".join(tokens))
    return seqs
