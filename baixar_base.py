from pathlib import Path
import zipfile
import sys
import gdown

PASTA_DADOS = Path("dados")
PASTA_DADOS.mkdir(exist_ok=True)

# Link do arquivo publicado pela PRF para acidentes de 2024 agrupados por ocorrência.
FILE_ID = "14lB0vqMFkaZj8HZ44b0njYgxs9nAN8KO"
ARQUIVO_ZIP = PASTA_DADOS / "datatran2024.zip"

def main():
    if not ARQUIVO_ZIP.exists():
        print("Baixando a base oficial da PRF (2024, agrupada por ocorrência)...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        resultado = gdown.download(url, str(ARQUIVO_ZIP), quiet=False)
        if resultado is None:
            print("\nNão foi possível baixar automaticamente.")
            print("Baixe manualmente o ZIP na página oficial da PRF e salve como:")
            print(ARQUIVO_ZIP.resolve())
            sys.exit(1)

    if not zipfile.is_zipfile(ARQUIVO_ZIP):
        print(f"O arquivo {ARQUIVO_ZIP} não parece ser um ZIP válido.")
        print("Apague-o e tente baixar novamente pela página oficial da PRF.")
        sys.exit(1)

    with zipfile.ZipFile(ARQUIVO_ZIP) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            print("O ZIP não contém CSV. Confira se baixou 'agrupados por ocorrência'.")
            sys.exit(1)
        for nome in csvs:
            destino = PASTA_DADOS / Path(nome).name
            with z.open(nome) as origem, open(destino, "wb") as saida:
                saida.write(origem.read())
            print(f"CSV extraído: {destino}")

    print("\nConcluído. Agora execute: python analise/01_exploracao.py")

if __name__ == "__main__":
    main()
