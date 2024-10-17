import unittest
import pandas as pd
from src.preprocess import preprocessar_dados
from src.config import Config

class TestPreprocess(unittest.TestCase):
    def test_preprocessar_dados(self):
        preprocessar_dados()
        df_treino = pd.read_csv(f"{Config.DATA_DIR}/processed/treino.csv")
        df_teste = pd.read_csv(f"{Config.DATA_DIR}/processed/teste.csv")
        self.assertGreater(len(df_treino), 0)
        self.assertGreater(len(df_teste), 0)

if __name__ == "__main__":
    unittest.main()
