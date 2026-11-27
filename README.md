# SAD_PRF — Análise de acidentes rodoviários da PRF

Projeto inicial em Python para explorar a base oficial de acidentes da PRF de 2024,
agrupada por ocorrência.

## 1. Preparação

Recomendado: Python 3.10 ou superior.

No terminal, dentro desta pasta:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Obter a base oficial

Por limitações de acesso ao download binário do Google Drive neste ambiente, o CSV
não foi incorporado fisicamente ao ZIP. O projeto já está configurado para baixar
automaticamente o ZIP oficial da PRF:

```powershell
python baixar_base.py
```

O script salva e extrai o arquivo em `dados/`. A fonte é a página oficial:
https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf

Selecione na página o item **Documento CSV de Acidentes 2024 (Agrupados por ocorrência)**.
O ID do arquivo do Drive utilizado pelo script corresponde ao link publicado nessa página.

Se o download automático falhar, baixe manualmente o ZIP pela página oficial e coloque-o
em `dados/` com o nome `datatran2024.zip`; depois execute novamente `python baixar_base.py`.

## 3. Explorar a base

```powershell
python analise/01_exploracao.py
```

O programa procura automaticamente um CSV em `dados/`, identifica o separador,
mostra dimensões, colunas, tipos, ausências e distribuição da classificação do acidente.

## 4. Abrir uma interface inicial

```powershell
streamlit run app.py
```

A interface inicial serve para conferir o arquivo e visualizar registros. Os modelos
de Árvore de Decisão e Regressão Logística serão adicionados após validar as colunas
e definir corretamente a variável-alvo, evitando usar informações que revelem a resposta.

## Observação metodológica importante

A base é agrupada por ocorrência. Antes de treinar, precisamos confirmar no dicionário
da PRF os significados dos campos e decidir a definição operacional de "grave".
Não use `mortos`, `feridos`, `classificacao_acidente` ou variáveis derivadas do desfecho
como entradas do modelo se elas forem usadas para construir o rótulo. Isso causaria
vazamento de informação (data leakage).

Fonte e dicionário:
- https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
