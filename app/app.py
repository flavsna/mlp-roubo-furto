"""Aplicação web (Flask) — classifica uma ocorrência como roubo ou furto usando a MLP treinada.
A inferência usa os pesos exportados do Keras (NumPy puro), reproduzindo exatamente a rede treinada."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import transform, periodo_from_hora, DIAS, PERIODOS  # noqa: E402

SPEC = json.load(open(ROOT / 'model' / 'spec.json'))
W = np.load(ROOT / 'model' / 'mlp_weights.npz')
META = json.load(open(ROOT / 'model' / 'metrics.json'))
N_LAYERS = len([k for k in W.files if k.startswith('W')])
app = Flask(__name__)


def predict_proba_roubo(X):
    a = X
    for i in range(N_LAYERS - 1):
        a = np.maximum(a @ W[f'W{i}'] + W[f'b{i}'], 0)           # camadas ocultas (ReLU)
    z = a @ W[f'W{N_LAYERS-1}'] + W[f'b{N_LAYERS-1}']
    return (1 / (1 + np.exp(-z))).ravel()                          # sigmoide


def build_row(p):
    hora = None
    if p.get('hora'):
        hh, mm = str(p['hora']).split(':')[:2]
        hora = int(hh) + int(mm) / 60
    periodo = periodo_from_hora(hora) if hora is not None else p.get('periodo', 'Não informado')
    return pd.DataFrame([{
        'bairro': str(p['bairro']).strip().upper(), 'tipo_local': p['tipo_local'], 'subtipo_local': p['subtipo_local'],
        'marca': p['marca'], 'periodo': periodo, 'dia_semana': int(p['dia_semana']), 'mes': int(p['mes']),
        'hora': np.nan if hora is None else hora, 'dias_registro': float(p.get('dias_registro', 0)),
        'qtd': int(p.get('qtd', 1))}])


@app.get('/')
def index():
    c = SPEC['cats']
    return render_template('index.html', bairros=[b for b in c['bairro'] if b != 'Outros'],
                           tipos=[t for t in c['tipo_local'] if t != 'Outros'] + ['Outros'],
                           subtipos=[t for t in c['subtipo_local'] if t != 'Outros'] + ['Outros'],
                           marcas=[t for t in c['marca'] if t != 'Outros'] + ['Outros'],
                           periodos=[p for p in PERIODOS], dias=DIAS, meta=META)


@app.post('/predict')
def predict():
    try:
        row = build_row(request.get_json(force=True))
        p = float(predict_proba_roubo(transform(row, SPEC))[0])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify(error=f'Entrada inválida: {e}'), 400
    return jsonify(classe='roubo' if p >= 0.5 else 'furto', prob_roubo=p, prob_furto=1 - p)


@app.get('/health')
def health():
    return jsonify(status='ok', modelo=META['modelo'])


if __name__ == '__main__':
    app.run(debug=True, port=5000)
