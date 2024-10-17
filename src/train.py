import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from src.config import Config
from sklearn.metrics import accuracy_score

def carregar_dados(tokenizer, caminho, max_len, batch_size):
    df = pd.read_csv(caminho)
    tokens = tokenizer(
        df['texto'].tolist(),
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    labels = torch.tensor(df['sentimento'].values)
    dataset = torch.utils.data.TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    return loader

def treinar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    modelo = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    modelo.to(device)
    loader_treino = carregar_dados(tokenizer, f"{Config.DATA_DIR}/processed/treino.csv", Config.MAX_LEN, Config.BATCH_SIZE)
    loader_teste = carregar_dados(tokenizer, f"{Config.DATA_DIR}/processed/teste.csv", Config.MAX_LEN, Config.BATCH_SIZE)

    otimizador = AdamW(modelo.parameters(), lr=Config.LEARNING_RATE)

    modelo.train()
    for epoca in range(Config.EPOCHS):
        total_loss = 0
        for batch in loader_treino:
            otimizador.zero_grad()
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            saida = modelo(input_ids, attention_mask=attention_mask, labels=labels)
            perda = saida.loss
            total_loss += perda.item()
            perda.backward()
            otimizador.step()

        print(f"Época [{epoca + 1}/{Config.EPOCHS}], Perda: {total_loss/len(loader_treino):.4f}")
    modelo.eval()
    predicoes, labels_verdadeiros = [], []

    with torch.no_grad():
        for batch in loader_teste:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            saida = modelo(input_ids, attention_mask=attention_mask)
            logits = saida.logits
            predicoes.extend(torch.argmax(logits, axis=1).cpu().numpy())
            labels_verdadeiros.extend(labels.cpu().numpy())

    acuracia = accuracy_score(labels_verdadeiros, predicoes)
    print(f"Acurácia no conjunto de teste: {acuracia:.2f}")
    modelo.save_pretrained(Config.MODEL_DIR)
    tokenizer.save_pretrained(Config.MODEL_DIR)
    print("Modelo e tokenizador salvos.")
