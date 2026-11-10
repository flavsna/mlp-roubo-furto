"""Construção do conjunto de dados: recorte geográfico, rótulo e atributos (sem vazamento)."""
import pandas as pd, numpy as np
from features import MAP_PERIODO, periodo_from_hora

COLS = ['ID_DELEGACIA', 'NUM_BO', 'ANO_BO', 'CIDADE', 'RUBRICA', 'DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA',
        'DATAHORA_REGISTRO_BO', 'DESCR_PERIODO', 'DESCR_TIPOLOCAL', 'DESCR_SUBTIPOLOCAL', 'BAIRRO',
        'MARCA_OBJETO', 'QUANTIDADE_OBJETO']


def build_dataset(raw):
    """raw: DataFrame da aba CELULAR_2025. Retorna (df_limpo, funil_de_registros)."""
    f = {'registros na base (estado de SP)': len(raw)}
    d = raw[COLS].copy()
    d = d[d.CIDADE == 'S.PAULO']                     # CIDADE = local do FATO (não a delegacia de registro)
    f['fatos ocorridos em São Paulo (capital)'] = len(d)
    d = d[d.RUBRICA.isin(['Furto (art. 155)', 'Roubo (art. 157)'])]
    f['rubrica = furto ou roubo'] = len(d)
    k = ['ID_DELEGACIA', 'NUM_BO', 'ANO_BO']
    amb = d.groupby(k).RUBRICA.transform('nunique') > 1   # BO com furto E roubo = ambíguo
    d = d[~amb]
    f['sem boletins ambíguos (furto+roubo)'] = len(d)
    d = d.drop_duplicates(k)                          # 1 linha por boletim (BO com vários celulares)
    f['1 linha por boletim (sem duplicidades)'] = len(d)
    d['y'] = (d.RUBRICA == 'Roubo (art. 157)').astype(int)
    occ = pd.to_datetime(d.DATA_OCORRENCIA_BO); reg = pd.to_datetime(d.DATAHORA_REGISTRO_BO).dt.normalize()
    d['dias_registro'] = (reg - occ).dt.days
    d = d[(occ >= '2024-01-01') & (occ <= '2025-12-31') & (d.dias_registro >= 0)]
    f['datas consistentes (2024-2025, registro >= fato)'] = len(d)
    occ = pd.to_datetime(d.DATA_OCORRENCIA_BO)
    t = pd.to_datetime(d.HORA_OCORRENCIA.astype(str), format='%H:%M:%S', errors='coerce')
    d['hora'] = t.dt.hour + t.dt.minute / 60
    per = d.hora.map(periodo_from_hora)
    d['periodo'] = per.fillna(d.DESCR_PERIODO.map(MAP_PERIODO)).fillna('Não informado')
    d['dia_semana'] = occ.dt.dayofweek; d['mes'] = occ.dt.month
    d['bairro'] = d.BAIRRO.str.strip().str.upper().fillna('NÃO INFORMADO')
    d['tipo_local'] = d.DESCR_TIPOLOCAL.fillna('Outros'); d['subtipo_local'] = d.DESCR_SUBTIPOLOCAL.fillna('Outros')
    d['marca'] = d.MARCA_OBJETO.fillna('Outros')
    d['qtd'] = pd.to_numeric(d.QUANTIDADE_OBJETO, errors='coerce').fillna(1).clip(1, 3)
    out = d[['bairro', 'tipo_local', 'subtipo_local', 'marca', 'periodo', 'dia_semana', 'mes', 'hora',
             'dias_registro', 'qtd', 'y']].reset_index(drop=True)
    return out, f
