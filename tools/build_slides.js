const pptxgen = require('pptxgenjs');
const fs = require('fs');
const M = JSON.parse(fs.readFileSync('model/metrics.json'));
const F = JSON.parse(fs.readFileSync('data/funil.json'));
const pres = new pptxgen(); pres.layout = 'LAYOUT_16x9';
const NAVY='14213D', AMB='E09F1F', INK='1F2937', MUT='6B7280', LIGHT='F3F4F6', WHITE='FFFFFF';
const fn = 'Calibri';
const pct = x => (x*100).toFixed(1)+'%';
const f3 = x => Number(x).toFixed(3);
const BD = { type:'solid', pt:0.5, color:'D1D5DB' };
function base(title, n) {
  const s = pres.addSlide(); s.background = { color: WHITE };
  s.addText(title, { x:0.5, y:0.25, w:9, h:0.6, fontFace:fn, fontSize:24, bold:true, color:NAVY, margin:0, isTextBox:true });
  s.addText(String(n), { x:9.2, y:5.2, w:0.5, h:0.3, fontFace:fn, fontSize:10, color:MUT, align:'right', margin:0, isTextBox:true });
  return s;
}
const bullets = arr => arr.map((t,i)=>({ text:t, options:{ bullet:true, breakLine:i<arr.length-1, paraSpaceAfter:4 } }));
const hdr = t => ({ text:t, options:{ bold:true, color:WHITE, fill:{color:NAVY}, fontFace:fn, fontSize:11, align:'center' } });
const cell = (t,o={}) => ({ text:String(t), options:Object.assign({ fontFace:fn, fontSize:11, color:INK, align:'center' }, o) });
const tb = (s, text, o) => s.addText(text, Object.assign({ fontFace:fn, margin:0, isTextBox:true }, o));

// 1
let s = pres.addSlide(); s.background = { color: NAVY };
tb(s,'Roubo ou furto?',{ x:0.6, y:1.2, w:8.8, h:0.9, fontSize:44, bold:true, color:WHITE });
tb(s,'Classificação de ocorrências de celulares subtraídos em São Paulo com uma rede MLP',{ x:0.6, y:2.1, w:8, h:0.8, fontSize:20, color:AMB });
tb(s,'Objetivo: dado um boletim de ocorrência já registrado (ou hipotético), estimar se ele é mais compatível com roubo ou furto e disponibilizar o modelo em uma aplicação web. A MLP não prevê a ocorrência de crimes.',{ x:0.6, y:3.1, w:8, h:0.9, fontSize:14, color:'D1D5DB' });
tb(s,'Integrantes: [nome 1] · [nome 2] · [nome 3]\nGlobal Solution – Aprendizado Profundo com Redes Neurais Convolucionais',{ x:0.6, y:4.4, w:8.5, h:0.7, fontSize:12, color:'9CA3AF' });

// 2
s = base('Problema, fonte e recorte', 2);
tb(s,[{text:'Fonte: ',options:{bold:true}},{text:'SSP-SP, Portal de Estatísticas → Bases de Dados → Celulares subtraídos (2025)\nhttps://www.ssp.sp.gov.br/estatistica/consultas'}],{ x:0.5, y:0.95, w:9, h:0.6, fontSize:12, color:INK });
const fr = Object.entries(F).map(([k,v])=>[cell(k,{align:'left'}), cell(v.toLocaleString('pt-BR'),{align:'right',bold:k.startsWith('amostra')})]);
s.addTable([[hdr('Etapa do recorte'),hdr('Registros')], ...fr], { x:0.5, y:1.65, w:5.3, colW:[3.9,1.4], border:BD });
tb(s,bullets([
 'Local do fato: CIDADE = S.PAULO (não o local de registro do BO)',
 'Rótulo: rubrica Roubo (art. 157) = 1 · Furto (art. 155) = 0',
 'Atributos: bairro, tipo e subtipo de local, período/hora, dia, mês, marca, dias até o registro, quantidade',
 'Excluídos por vazamento: rubrica, conduta, circunstância, desdobramento, flagrante, autoria, forma de apresentação',
 'Sem dados pessoais: logradouro, número, CEP, coordenadas']),{ x:6.0, y:1.65, w:3.5, h:3.4, fontSize:11, color:INK, valign:'top' });

// 3
s = base('Preparação dos dados', 3);
const prep = [['Bairro, tipo/subtipo de local, marca','One-hot; categorias raras → "Outros" (40 bairros mais frequentes)'],
 ['Período do dia','One-hot (hora exata quando existe; senão, campo DESCR_PERIODO)'],
 ['Dia da semana','One-hot (7)'],['Mês e hora','Cíclico: seno e cosseno (dez≈jan, 23h≈0h) + flag "hora informada"'],
 ['Dias até o registro','log(1+x) e z-score (média/desvio só do treino)'],['Quantidade','Limitada a 3 e escalada 0–1']];
s.addTable([[hdr('Variável'),hdr('Tratamento')], ...prep.map(r=>[cell(r[0],{align:'left'}),cell(r[1],{align:'left'})])], { x:0.5, y:0.95, w:9, colW:[3,6], border:BD });
const box=(x,w,big,small,c)=>{ s.addShape(pres.shapes.RECTANGLE,{x,y:3.7,w,h:1.1,fill:{color:LIGHT},line:{color:LIGHT}}); tb(s,big,{x,y:3.75,w,h:0.6,fontSize:26,bold:true,color:c,align:'center'}); tb(s,small,{x,y:4.35,w,h:0.4,fontSize:11,color:MUT,align:'center'}); };
box(0.5,2.1,String(M.n_features),'atributos de entrada',NAVY);
box(2.75,2.1,M.n_treino.toLocaleString('pt-BR'),'treino (70%)',NAVY);
box(5.0,2.1,M.n_val.toLocaleString('pt-BR'),'validação (15%)',NAVY);
box(7.25,2.25,M.n_teste.toLocaleString('pt-BR'),'teste (15%) — uso único','B7791F');
tb(s,'Divisão estratificada (seed 42). Sem hora exata → hora=0 + flag. Escalonamento e categorias ajustados só no treino (sem vazamento).',{ x:0.5, y:4.95, w:8.6, h:0.3, fontSize:10, color:MUT });

// 4
s = base('Arquitetura da MLP escolhida: ' + M.modelo, 4);
const layers=[['Entrada',`${M.n_features}`,'atributos',MUT],['Densa 1','128','ReLU + Dropout 0,3',NAVY],['Densa 2','64','ReLU + Dropout 0,3',NAVY],['Densa 3','32','ReLU + Dropout 0,3',NAVY],['Saída','1','Sigmoide → P(roubo)','B7791F']];
layers.forEach((l,i)=>{ const x=0.5+i*1.85; s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:1.2,w:1.55,h:1.5,fill:{color:l[3]},line:{color:l[3]},rectRadius:0.1});
 s.addText([{text:l[0],options:{fontSize:12,breakLine:true}},{text:l[1],options:{fontSize:30,bold:true,breakLine:true}},{text:l[2],options:{fontSize:10}}],{x,y:1.2,w:1.55,h:1.5,fontFace:fn,color:WHITE,align:'center',valign:'middle',margin:2,isTextBox:true});
 if(i<4) tb(s,'→',{x:x+1.55,y:1.7,w:0.3,h:0.5,fontSize:20,color:MUT,align:'center'}); });
const npar = M.comparacao.find(r=>r.modelo===M.modelo)['parâmetros'].toLocaleString('pt-BR');
tb(s,bullets([
 'Cada neurônio calcula z = w·x + b e aplica uma ativação; as camadas ocultas aprendem combinações dos atributos (ex.: tipo de local × período × marca).',
 'ReLU nas ocultas: simples e evita saturação do gradiente. Sigmoide na saída: transforma z em probabilidade entre 0 e 1 (limiar 0,5).',
 'Dropout (0,3): desliga aleatoriamente 30% dos neurônios durante o treino, reduzindo overfitting.',
 `Total de ${npar} parâmetros treináveis.`]),{ x:0.5, y:3.0, w:9, h:2.1, fontSize:13, color:INK, valign:'top' });

// 5
s = base('Configuração do treinamento', 5);
const cfg=[['Função de perda','Binary cross-entropy (saída probabilística, 2 classes)'],['Otimizador','Adam, learning rate 1e-3 (5e-4 no M4)'],['Batch size','256 (128 no M4)'],['Épocas','Até 80, com early stopping (paciência 8 em val_loss, restaura os melhores pesos)'],['Métricas','Acurácia, precision, recall e F1 da classe roubo'],['Limiar de decisão','0,5'],['Reprodutibilidade','Seed 42; TensorFlow/Keras; pré-processamento ajustado só no treino']];
s.addTable([[hdr('Item'),hdr('Escolha')], ...cfg.map(r=>[cell(r[0],{align:'left',bold:true}),cell(r[1],{align:'left'})])], { x:0.5, y:1.0, w:9, colW:[2.4,6.6], border:BD });
tb(s,'Por que BCE + sigmoide? É a combinação natural para classificação binária: a perda penaliza fortemente previsões confiantes e erradas.',{ x:0.5, y:4.6, w:9, h:0.5, fontSize:12, color:MUT, italic:true });

// 6
s = base('Quatro arquiteturas testadas', 6);
const ch={ 'M1 (16)':['1 camada, 16 neurônios','Baseline simples'],'M2 (64-32)':['2 camadas, 64-32','Mais capacidade'],'M3 (128-64-32 +DO)':['3 camadas, 128-64-32 + Dropout 0,3','Mais profundidade + regularização'],'M4 (128-64 +L2+DO)':['2 camadas, 128-64 + L2 1e-4 + Dropout 0,3; LR 5e-4; batch 128','Regularização dupla e LR menor'] };
s.addTable([[hdr('Modelo'),hdr('Estrutura'),hdr('Por que mudamos'),hdr('Params'),hdr('Épocas*')],
 ...M.comparacao.map(r=>[cell(r.modelo,{bold:true,align:'left'}),cell(ch[r.modelo][0],{align:'left'}),cell(ch[r.modelo][1],{align:'left'}),cell(r['parâmetros'].toLocaleString('pt-BR')),cell(r['épocas (melhor)'])])],
 { x:0.5, y:1.0, w:9, colW:[1.9,3.2,2.4,0.8,0.7], border:BD });
tb(s,'* época com menor perda de validação (early stopping).\nEfeito observado: aumentar a rede quase não alterou o desempenho; o M2 atinge a melhor perda de validação já nas primeiras épocas (começa a sobreajustar); o Dropout (M3) deu a melhor perda e o melhor F1 de validação; L2 + LR menor (M4) não trouxe ganho.',{ x:0.5, y:3.8, w:9, h:1.3, fontSize:12, color:INK, valign:'top' });

// 7
s = base('Comparação e escolha do modelo', 7);
const rows=M.comparacao.map(r=>{ const b=r.modelo===M.modelo; const o={bold:b, fill:{color:b?'FEF3C7':WHITE}}; return [cell(r.modelo,Object.assign({align:'left'},o)),cell(pct(r['acc val']),o),cell(f3(r['precision val']),o),cell(f3(r['recall val']),o),cell(f3(r['F1 val']),o),cell(f3(r.val_loss),o)]; });
tb(s,'Validação (3.000 registros) — critério de seleção',{ x:0.5, y:0.9, w:9, h:0.3, fontSize:12, bold:true, color:MUT });
s.addTable([[hdr('Modelo'),hdr('Acurácia'),hdr('Precision'),hdr('Recall'),hdr('F1'),hdr('Val loss')], ...rows], { x:0.5, y:1.2, w:9, colW:[2.8,1.2,1.2,1.2,1.2,1.4], border:BD });
const T=M.teste, L=M.referencias.logistica;
tb(s,'Teste final (uso único) — modelo escolhido vs. referências',{ x:0.5, y:3.0, w:9, h:0.3, fontSize:12, bold:true, color:MUT });
s.addTable([[hdr('Modelo'),hdr('Acurácia'),hdr('Precision'),hdr('Recall'),hdr('F1')],
 [cell(M.modelo,{bold:true,align:'left'}),cell(pct(T.acuracia),{bold:true}),cell(f3(T.precision),{bold:true}),cell(f3(T.recall),{bold:true}),cell(f3(T.f1),{bold:true})],
 [cell('Regressão logística (referência)',{align:'left'}),cell(pct(L.acuracia)),cell(f3(L.precision)),cell(f3(L.recall)),cell(f3(L.f1))],
 [cell('Sempre "furto" (chute)',{align:'left'}),cell(pct(M.referencias.majoritaria_acc)),cell('–'),cell('0,000'),cell('0,000')]],
 { x:0.5, y:3.3, w:9, colW:[3.4,1.4,1.4,1.4,1.4], border:BD });
tb(s,'Escolha: M3 (maior F1 e menor perda na validação). As diferenças entre modelos são pequenas, dentro do ruído; a regressão logística empata com a MLP.',{ x:0.5, y:4.85, w:9, h:0.4, fontSize:11, color:INK });

// 8
s = base('Curvas, matriz de confusão e diagnóstico', 8);
s.addImage({ path:'reports/figures/curvas_modelo_escolhido.png', x:0.4, y:0.95, w:6.0, h:6.0*4.5/14 });
s.addImage({ path:'reports/figures/matriz_confusao.png', x:6.6, y:0.85, w:3.0, h:3.0*4.2/5 });
const cm=M.matriz_confusao;
tb(s,bullets([
 'Treino ≈ validação em todos os modelos (gap < 2 p.p.): sem overfitting severo; leve sinal no M2 e no M4.',
 'Sem underfitting relevante: o M1 (16 neurônios) já chega perto das redes maiores.',
 `Teste: ${cm[1][1]} roubos acertados, ${cm[1][0]} roubos classificados como furto, ${cm[0][1]} furtos classificados como roubo, ${cm[0][0]} furtos acertados.`,
 'O teto de desempenho vem da informação disponível nos atributos, não do tamanho da rede.']),{ x:0.5, y:2.95, w:5.9, h:2.3, fontSize:11.5, color:INK, valign:'top' });
tb(s,'Matriz no conjunto de teste (3.000 registros).',{ x:6.6, y:3.5, w:3.0, h:0.4, fontSize:10, color:MUT });

// 9
s = base('Aplicação web', 9);
s.addImage({ path:'reports/figures/app_screenshot.png', x:0.4, y:0.95, w:5.6, h:5.6*760/1100 });
tb(s,bullets([
 'Flask + NumPy: o servidor carrega os pesos da MLP treinada (exportados do Keras) e reproduz a rede; diferença máxima vs. Keras < 1e-7.',
 'Formulário com exatamente os atributos usados no treino; mesmo pré-processamento (src/features.py).',
 'Mostra a classe estimada e as probabilidades de roubo e furto.',
 'Aviso fixo: o modelo classifica o tipo de uma ocorrência, não o risco de crime.',
 'Link: [colar URL do deploy] · Código: [colar URL do GitHub]']),{ x:6.2, y:1.0, w:3.4, h:4.2, fontSize:11.5, color:INK, valign:'top' });

// 10
s = base('Conclusões, limitações e uso de IA generativa', 10);
tb(s,'Conclusões',{ x:0.5, y:0.95, w:4.3, h:0.3, fontSize:14, bold:true, color:NAVY });
tb(s,bullets([`A MLP acerta ${pct(T.acuracia)} no teste (chute: ${pct(M.referencias.majoritaria_acc)}) com F1 de ${T.f1.toFixed(2)} para roubo.`,'Mais camadas ≠ mais acerto: as 4 arquiteturas e a regressão logística ficam muito próximas.','A informação dos atributos, e não a rede, limita o resultado.']),{ x:0.5, y:1.3, w:4.3, h:3.5, fontSize:12, color:INK, valign:'top' });
tb(s,'Limitações',{ x:5.1, y:0.95, w:4.4, h:0.3, fontSize:14, bold:true, color:NAVY });
tb(s,bullets(['Dados administrativos: subnotificação e erros de preenchimento.','Só celulares, 2024–2025; ~50% sem hora exata.','Divisão aleatória (sem validação temporal); probabilidades não calibradas.','Não usar para decisões sobre pessoas ou policiamento.']),{ x:5.1, y:1.3, w:4.4, h:2.2, fontSize:12, color:INK, valign:'top' });
s.addShape(pres.shapes.RECTANGLE,{x:5.1,y:3.6,w:4.4,h:1.5,fill:{color:LIGHT},line:{color:LIGHT}});
tb(s,[{text:'IA generativa (Claude)',options:{bold:true,breakLine:true}},{text:'Apoio em exploração da base, código de pré-processamento, treino e app, e rascunho da documentação. A equipe revisou o código, checou vazamento de dados e valida e explica os resultados.'}],{ x:5.2, y:3.65, w:4.2, h:1.4, fontSize:11, color:INK, valign:'top' });

pres.writeFile({ fileName: 'reports/relatorio_tecnico_10_slides.pptx' }).then(()=>console.log('ok'));
