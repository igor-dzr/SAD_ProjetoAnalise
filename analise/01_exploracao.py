from pathlib import Path
import pandas as pd

PASTA_DADOS = Path("dados")

def localizar_csv():
    arquivos = list(PASTA_DADOS.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(
            "Nenhum CSV encontrado em 'dados/'. Execute primeiro: python baixar_base.py"
        )
    # Prioriza o arquivo cujo nome contém 'datatran'.
    return next((p for p in arquivos if "datatran" in p.name.lower()), arquivos[0])

def ler_csv(caminho):
    # O arquivo da PRF pode apresentar variações de codificação/separador.
    erros = []
    for encoding in ("latin1", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(caminho, sep=None, engine="python",
                               encoding=encoding)
        except Exception as exc:
            erros.append(f"{encoding}: {exc}")
    raise RuntimeError("Não foi possível ler o CSV:\n" + "\n".join(erros))

def main():
    caminho = localizar_csv()
    df = ler_csv(caminho)

    print("\n========== ARQUIVO ==========")
    print(caminho.resolve())
    print("\n========== DIMENSÕES ==========")
    print(f"Registros: {df.shape[0]:,}")
    print(f"Colunas: {df.shape[1]}")

    print("\n========== NOMES DAS COLUNAS ==========")
    for coluna in df.columns:
        print("-", coluna)

    print("\n========== PRIMEIROS 5 REGISTROS ==========")
    print(df.head().to_string())

    print("\n========== TIPOS DE DADOS ==========")
    print(df.dtypes.to_string())

    print("\n========== VALORES AUSENTES ==========")
    ausentes = df.isna().sum().sort_values(ascending=False)
    print(ausentes[ausentes > 0].to_string() or "Nenhum valor ausente.")

    print("\n========== CLASSIFICAÇÃO DO ACIDENTE ==========")
    coluna = next((c for c in df.columns
                   if c.strip().lower() == "classificacao_acidente"), None)
    if coluna:
        print(df[coluna].value_counts(dropna=False).to_string())
    else:
        print("Coluna classificacao_acidente não encontrada. Confira os nomes acima.")

    print("\n========== RESUMO ==========")
    print(df.describe(include="all").transpose().to_string())

    # Salva um resumo simples para consulta posterior.
    Path("analise").mkdir(exist_ok=True)
    with open("analise/resumo_colunas.txt", "w", encoding="utf-8") as f:
        f.write(f"Arquivo: {caminho}\n")
        f.write(f"Registros: {df.shape[0]}\nColunas: {df.shape[1]}\n\n")
        f.write("\n".join(map(str, df.columns)))
    print("\nResumo das colunas salvo em analise/resumo_colunas.txt")

if __name__ == "__main__":
    main()
