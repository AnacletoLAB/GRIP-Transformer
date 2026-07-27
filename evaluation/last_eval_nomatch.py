# Esegue una valutazione del modello NucTransformer sul dataset no_match.csv e calcola la loss
# (la funzione di perdita). Il dataset no_match.csv è stato costruito con le coppie di test che non hanno match
# con il training set applicando l'algoritmo BLASTn.

import pandas as pd
from tokenizer import tokenize_batch
from tokenizer import tokenizer
import torch
from model import NucTransformer, NucConfig

# Leggi il dataset filtrato 
df = pd.read_csv("data/dry_run/no_match.csv", sep=';')

# commentare dopo
# print(df.head())
# print("Numero righe:", len(df))

# Prepara il batch per la funzione tokenize_batch
batch = {
    "source": df["seq1"].tolist(),
    "target": df["seq2"].tolist(),
}

# print("Esempio batch:")
# print("source[0]:", batch["source"][0])
# print("target[0]:", batch["target"][0])

# Tokenizza il batch
tokenized = tokenize_batch(batch)

# Caricamento del modello addestrato salvato 
model = NucTransformer.from_pretrained("ckpt/dry_run/final_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Sposta i tensori sullo stesso device
input_ids = tokenized["input_ids"].to(device)
labels = tokenized["labels"].to(device)
'''
# Esempio di tokenizzazione
for i in range(2):
    print(f"\nSequenza sorgente {i}: {batch['source'][i]}")
    print("Token ids input_ids:", tokenized["input_ids"][i].tolist())
    print(f"Sequenza target   {i}: {batch['target'][i]}")
    print("Token ids labels:   ", tokenized["labels"][i].tolist())

# Mostra anche il mapping inverso 
print("\nDecodifica inversa dal tensore:")
for i in range(2):
    print("Input decodificato:", tokenizer.decode(tokenized["input_ids"][i], skip_special_tokens=False))
    print("Label decodificata:", tokenizer.decode(tokenized["labels"][i], skip_special_tokens=False))
'''
with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"].item()
    
print("Loss sul dataset filtrato:", loss)


