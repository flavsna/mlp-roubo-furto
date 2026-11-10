"""Pré-processamento compartilhado entre notebook (treino) e aplicação web (inferência).
Os parâmetros (categorias, médias, desvios) são ajustados SOMENTE no treino e salvos em JSON."""
import numpy as np, pandas as pd

PERIODOS = ['Madrugada', 'Manhã', 'Tarde', 'Noite', 'Em hora incerta', 'Não informado']
MAP_PERIODO = {'De madrugada': 'Madrugada', 'Pela manhã': 'Manhã', 'A tarde': 'Tarde',
               'A noite': 'Noite', 'Em hora incerta': 'Em hora incerta'}
CAT_COLS = ['bairro', 'tipo_local', 'subtipo_local', 'marca']
DIAS = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
INPUT_COLS = ['bairro', 'tipo_local', 'subtipo_local', 'marca', 'periodo', 'dia_semana',
              'mes', 'hora', 'dias_registro', 'qtd']


def periodo_from_hora(h):
    if h is None or pd.isna(h):
        return np.nan
    return 'Madrugada' if h < 6 else 'Manhã' if h < 12 else 'Tarde' if h < 18 else 'Noite'


def fit_spec(train, top_bairro=40, top_sub=20, top_marca=6):
    """Aprende categorias frequentes e estatísticas de escala usando apenas o treino."""
    tops = {'bairro': top_bairro, 'tipo_local': 12, 'subtipo_local': top_sub, 'marca': top_marca}
    cats = {c: train[c].value_counts().head(n).index.tolist() + ['Outros'] for c, n in tops.items()}
    cats['periodo'] = PERIODOS
    d = np.log1p(train['dias_registro'].clip(0, 365))
    spec = {'cats': cats, 'dias_mean': float(d.mean()), 'dias_std': float(d.std())}
    spec['feature_names'] = list(transform(train.head(1), spec, return_names=True))
    return spec


def transform(df, spec, return_names=False):
    cols, names = [], []
    for c in CAT_COLS + ['periodo']:
        cats = spec['cats'][c]
        v = df[c].where(df[c].isin(cats), 'Outros') if c != 'periodo' else df[c].fillna('Não informado')
        for k in cats:
            cols.append((v == k).astype('float32').values); names.append(f'{c}={k}')
    for i, k in enumerate(DIAS):
        cols.append((df['dia_semana'] == i).astype('float32').values); names.append(f'dia={k}')
    m = 2 * np.pi * (df['mes'].values - 1) / 12
    h = df['hora'].astype(float).values
    hk = ~np.isnan(h)
    ang = np.where(hk, 2 * np.pi * np.nan_to_num(h) / 24, 0.0)
    d = (np.log1p(df['dias_registro'].clip(0, 365).values) - spec['dias_mean']) / spec['dias_std']
    for arr, n in [(np.sin(m), 'mes_sin'), (np.cos(m), 'mes_cos'),
                   (np.where(hk, np.sin(ang), 0), 'hora_sin'), (np.where(hk, np.cos(ang), 0), 'hora_cos'),
                   (hk.astype(float), 'hora_informada'), (d, 'dias_registro_z'),
                   ((np.minimum(df['qtd'].values, 3) - 1) / 2, 'qtd_norm')]:
        cols.append(arr.astype('float32')); names.append(n)
    if return_names:
        return names
    return np.column_stack(cols).astype('float32')
