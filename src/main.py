from src.preprocess import preprocessar_dados
from src.train import treinar_modelo

def main():
    preprocessar_dados()
    treinar_modelo()

if __name__ == "__main__":
    main()
