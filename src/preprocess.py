import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from src.config import Config

def preprocessar_dados():
    df = pd.read_csv(f"{Config.DATA_DIR}/raw/dataset.csv")
    df = df.dropna(subset=['texto', 'sentimento'])
    df['texto'] = df['texto'].str.lower()
    df_treino, df_teste = train_test_split(df, test_size=0.2, random_state=42)
    df_treino.to_csv(f"{Config.DATA_DIR}/processed/treino.csv", index=False)
    df_teste.to_csv(f"{Config.DATA_DIR}/processed/teste.csv", index=False)

    print("Dados pré-processados e salvos.")
