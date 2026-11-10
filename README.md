# Roubo ou furto? Classificação de ocorrências de celulares em São Paulo com MLP

Global Solution — Aprendizado Profundo com Redes Neurais Convolucionais.
Uma rede **MLP** (TensorFlow/Keras) classifica boletins de ocorrência de celulares subtraídos na **cidade de São Paulo** como **roubo** ou **furto**, e uma aplicação web (Flask) permite testar novos exemplos.

> ⚠️ O modelo classifica o **tipo** de uma ocorrência já registrada (ou hipotética). Ele **não** prevê crimes nem calcula o risco de um crime acontecer em um local/horário.

**Equipe:** _[Ana Clara Alcântara Lopes, Ana Flávia Couto Chaves, 
Izabely dos Santos Silva, Natalia Heloisa Rocha de Alcântara Bezerra e Raissa Marinho de Jesus Viana]_  
**Aplicação web:** _[colar aqui a URL do deploy]_  
**Repositório:** _[colar aqui a URL do GitHub]_

## Resultados (conjunto de teste, 3.000 registros)
| Modelo | Acurácia | Precision | Recall | F1 (roubo) |
|---|---|---|---|---|
| **MLP M3 (128-64-32 + Dropout)** — escolhida pela validação | 83,3% | 0,754 | 0,819 | 0,785 |
| Regressão logística (referência) | 83,4% | 0,761 | 0,808 | 0,784 |
| Sempre "furto" (chute) | 62,8% | – | 0 | 0 |

Foram comparadas 4 arquiteturas (M1 16 · M2 64-32 · M3 128-64-32+Dropout · M4 128-64+L2+Dropout); as diferenças são pequenas. Detalhes em `notebooks/mlp_roubo_furto.ipynb` e `reports/comparacao_arquiteturas.csv`.

## Fonte dos dados e como obtê-los
- **SSP-SP – Portal de Estatísticas → Bases de Dados → Celulares subtraídos (2025)**: https://www.ssp.sp.gov.br/estatistica/consultas
- Baixe `CelularesSubtraidos_2025.xlsx` (aba `CELULAR_2025`) e salve em `data/`.
- Recorte: `CIDADE == 'S.PAULO'` (local do fato), rubrica Furto (art. 155) ou Roubo (art. 157), 1 linha por boletim, sem boletins ambíguos. Funil completo em `data/funil.json`.
- `data/amostra_sp_20k.csv` é a amostra de 20.000 registros usada no treino (já incluída: dá para reproduzir sem o xlsx).
- **Sem vazamento:** não são usados rubrica, conduta, circunstância, desdobramento, flagrante, autoria nem forma de apresentação. Sem dados pessoais (logradouro, CEP, coordenadas).

## Estrutura
```
app/            aplicação Flask (app.py, templates/index.html)
src/            features.py (pré-processamento compartilhado) e data_prep.py (recorte)
notebooks/      mlp_roubo_furto.ipynb (executado, com resultados)
model/          mlp_roubo_furto.keras, mlp_weights.npz, spec.json, metrics.json
data/           amostra_sp_20k.csv, funil.json
reports/        relatorio_tecnico_10_slides.pptx/.pdf, figuras, comparação das arquiteturas
tools/          scripts que geram o notebook e os slides
```

## Como executar
```bash
# 1) Aplicação web (modelo já treinado incluído) — só precisa de Flask/NumPy/Pandas
pip install -r requirements.txt
python app/app.py            # http://127.0.0.1:5000

# 2) Reproduzir o treino
pip install -r requirements-train.txt
cd notebooks && jupyter nbconvert --to notebook --execute --inplace mlp_roubo_furto.ipynb
```
O notebook usa `data/CelularesSubtraidos_2025.xlsx` se existir (refaz o recorte e a amostra); caso contrário, usa `data/amostra_sp_20k.csv`. Seeds fixas (42).

## Publicar a aplicação (link para o professor)
**Render.com** (gratuito): *New → Web Service* → conecte o repositório do GitHub →
Build command: `pip install -r requirements.txt` · Start command: `gunicorn --chdir app app:app`.
Alternativas: Railway, Fly.io ou Hugging Face Spaces (Docker). Depois cole a URL acima.

## Como a aplicação usa a MLP
`app/app.py` carrega `model/mlp_weights.npz` (pesos exportados do Keras) e repete a mesma rede (Dense+ReLU … Dense+sigmoide) em NumPy, usando o mesmo `src/features.py` do treino. A diferença máxima de saída vs. Keras é < 1e-7 (verificada no notebook). Isso deixa o deploy leve (sem TensorFlow).

## Uso de IA generativa
Claude (Anthropic) apoiou exploração dos dados, código de pré-processamento/treino/app e rascunho da documentação. A equipe revisou o código, verificou vazamento de dados e é responsável por explicar os resultados.
