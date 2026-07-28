import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from pos_encoding import PositionalEncoding
# from tokenizer import tokenizer             usato in 3-mer senza overlapping
from tokenizer import tokenizer, tokenize_batch, detokenize_batch     # usato nell'ultima versione di 1-mer


class NucConfig(PretrainedConfig):
    """Configurazione per il modello NucTransformer"""
    model_type = "nucformer"
    
    def __init__(
        self, 
        vocab_size=tokenizer.vocab_size, 
        d_model=1024, 
        nhead=16,
        num_encoder_layers=12, 
        num_decoder_layers=12,
        dim_feedforward=4096, 
        dropout=0.1, 
        max_len=1024, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.max_len = max_len

        # Token speciali (coerenti con il tokenizer)
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2


class NucTransformer(PreTrainedModel):
    config_class = NucConfig
    
    def __init__(self, config: NucConfig):
        super().__init__(config)
        self.embed_src = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_tgt = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_len)
        self.transformer = nn.Transformer(
            d_model=config.d_model, 
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, 
            batch_first=True
        )
        self.out = nn.Linear(config.d_model, config.vocab_size)
        self.init_weights()

    def forward(self, input_ids, decoder_input_ids=None, attention_mask=None, labels=None):
        """Forward pass del modello"""
        src = self.pos_enc(self.embed_src(input_ids))
        
        if decoder_input_ids is None:
            raise ValueError("decoder_input_ids must be provided for the decoder input.")
        
        tgt = self.pos_enc(self.embed_tgt(decoder_input_ids))
        
        # Encoder
        memory = self.transformer.encoder(src)
        
        # Decoder con causal mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(src.device)
        dec = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Proiezione al vocabolario
        logits = self.out(dec)
        
        # Calcolo della loss se labels è fornito
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.reshape(-1)
            )
        
        return {"loss": loss, "logits": logits}

    def _get_special_token_ids(self):
        """Recupera gli ID dei token speciali in modo robusto"""
        # Recupera i token ID dal config, con fallback di sicurezza
        pad_id = getattr(self.config, "pad_token_id", 0)
        eos_id = getattr(self.config, "eos_token_id", 1)
        unk_id = getattr(self.config, "unk_token_id", 2)
    
        # Converti a int in modo sicuro
        pad_id = int(pad_id)
        eos_id = int(eos_id)
        unk_id = int(unk_id)
    
        # Aggiorna la config per coerenza
        self.config.pad_token_id = pad_id
        self.config.eos_token_id = eos_id
        self.config.unk_token_id = unk_id
    
        # Restituisce gli ID utili al modello
        return pad_id, eos_id, unk_id

    def _apply_top_p_filtering(self, probs, top_p):
        """Applica nucleus (top-p) sampling alle probabilità"""
        if top_p >= 1.0:
            return probs
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Rimuovi token oltre la soglia top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Mantieni sempre almeno il token più probabile
        sorted_indices_to_remove[..., 0] = False
        if sorted_indices_to_remove.dim() > 1:
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        
        # Azzera le probabilità dei token rimossi
        probs_filtered = probs.clone()
        probs_filtered.scatter_(
            -1, 
            sorted_indices, 
            sorted_probs * (~sorted_indices_to_remove).float()
        )
        
        # Rinormalizza
        probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)
        
        return probs_filtered
        """
        Generazione autoregressiva di sequenze di nucleotidi.
        
        Args:
            input_ids: Tensor di input (batch_size, seq_len)
            attention_mask: Maschera di attenzione opzionale
            max_new_tokens: Massimo numero di nuovi token da generare
            min_new_tokens: Minimo numero di token prima di permettere EOS
            temperature: Temperatura per il sampling (default: 1.0)
            top_p: Soglia per nucleus sampling (default: 1.0, disabilitato)
            top_k: Numero di top token per il sampling (default: 30)
            verbose: Se True, stampa informazioni di debug
        
        Returns:
            Tensor con i token generati (batch_size, generated_length)
        """
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        max_new_tokens=50,
        min_new_tokens=0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        # Parametri legacy per compatibilità
        target_length=None,
        max_length=None,
        min_length=None,
        verbose=False
    ):       
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Validazione parametri
        if temperature <= 0:
            raise ValueError(f"temperature deve essere > 0, ricevuto: {temperature}")
        if not 0 < top_p <= 1.0:
            raise ValueError(f"top_p deve essere in (0, 1], ricevuto: {top_p}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k deve essere > 0 o None, ricevuto: {top_k}")
        
        # Recupera token speciali
        pad_id, eos_id, unk_id = self._get_special_token_ids()
        
        # Gestione parametri legacy
        if target_length is not None:
            max_new_tokens = int(target_length)
        elif max_length is not None:
            max_new_tokens = int(max_length)
        
        if min_length is not None:
            min_new_tokens = int(min_length)
            
        if decoder_input_ids is not None:
            # Evita che max_length o target_length sovrascrivano max_new_tokens
            target_length = None
            max_length = None

        max_new_tokens = max(1, int(max_new_tokens))
        # print(f"[DEBUG generate()] max_new_tokens effettivo = {max_new_tokens}")
        min_new_tokens = max(0, int(min_new_tokens))
        
        if verbose:
            print(f"Generazione con: max_new_tokens={max_new_tokens}, "
                  f"min_new_tokens={min_new_tokens}, temperature={temperature}, "
                  f"top_k={top_k}, top_p={top_p}")
        
        # *** OTTIMIZZAZIONE: Calcola l'encoder UNA SOLA VOLTA ***
        src = self.pos_enc(self.embed_src(input_ids))
        memory = self.transformer.encoder(src)
        
        # --- Inizializzazione del decoder ---
        if decoder_input_ids is not None:
            # Usa il prompt fornito (es. metà target)
            generated = decoder_input_ids.clone().to(device)
            if verbose:
                print(f"Prompt iniziale fornito: {generated.shape[1]} token")
        else:
            # Comportamento standard: inizia con PAD
            generated = torch.full(
                (batch_size, 1),
                pad_id,
                device=device,
                dtype=torch.long
            )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generazione autoregressiva
        # Generazione autoregressiva — corretta
        initial_len = generated.size(1)
        target_total_len = initial_len + max_new_tokens
        
        step = 0
        while generated.size(1) < target_total_len:
            step += 1
        
            # Decoder step
            tgt = self.pos_enc(self.embed_tgt(generated))
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                tgt.size(1)
            ).to(device)
            dec = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.out(dec)[:, -1, :]
        
            # Temperatura e top-k
            logits = logits / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
                filtered_logits = torch.full_like(logits, float("-inf"))
                filtered_logits.scatter_(-1, topk_idx, topk_vals)
                logits = filtered_logits
        
            # Sampling
            probs = F.softmax(logits, dim=-1)
            probs = self._apply_top_p_filtering(probs, top_p)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
            # Aggiungi token e aggiorna 'finished'
            next_tokens = torch.where(
                finished,
                torch.tensor(pad_id, device=device, dtype=torch.long),
                next_tokens
            )
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            finished |= (next_tokens == eos_id)
        
            # Termina se tutti finiti
            if finished.all():
                break
        
            # Sicurezza: non superare il limite
            if generated.size(1) >= target_total_len:
                if verbose:
                    print(f"[DEBUG] Stop forzato a {generated.size(1)} token (target={target_total_len})")
                break

        return generated

    def print_model_params(self, full=False):
        """
        Stampa la struttura e il numero di parametri del modello.
        
        Args:
            full: Se True, stampa anche i valori dei parametri
        """
        print(f"Model structure:\n{self}\n")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")
        
        for name, param in self.named_parameters():
            if full:
                print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
            else:
                print(f"Layer: {name} | Size: {param.size()} | "
                      f"Trainable: {param.requires_grad}")