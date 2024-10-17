import os
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

class Config:
    DATA_DIR = os.getenv("DATA_DIR")
    MODEL_DIR = os.getenv("MODEL_DIR")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    EPOCHS = int(os.getenv("EPOCHS"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
    MAX_LEN = int(os.getenv("MAX_LEN"))