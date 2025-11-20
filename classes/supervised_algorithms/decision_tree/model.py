#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pipenv install pandas plotly matplotlib pingouin nbformat ipykernel scikit-learn optuna ipywidgets gradio


# In[2]:


# EDA
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Otimização de Hiperparâmetros
import optuna


# In[3]:


# Carregar Dataset
df_segmento = pd.read_csv("./datasets/dataset_segmento_clientes.csv")


# ### EDA

# In[4]:


# Visualizar os dados
df_segmento.head(5)


# In[5]:


# Estrutura do Dataset
df_segmento.info()


# In[6]:


# Valores possíveis - Variáveis Categóricas
df_segmento['atividade_economica'].unique()


# In[7]:


# Valores possíveis - Variáveis Categóricas
df_segmento['localizacao'].unique()


# In[8]:


# Valores possíveis - Variáveis Categóricas
df_segmento['segmento_de_cliente'].unique()


# In[9]:


# Valores possíveis - Variáveis Categóricas
df_segmento['inovacao'].unique()


# In[10]:


# Distribuição da Variável Segmento de Cliente (Target)
contagem_target = df_segmento.value_counts('segmento_de_cliente')
contagem_target


# In[11]:


# Criar uma lista ordenada do target
lista_segmentos = [
  'Starter', 'Bronze', 'Silver', 'Gold'
]


# In[12]:


# Distribuição da Variável Target - Contagem
px.bar(contagem_target, color=contagem_target.index, category_orders={
  'segmento_de_cliente': lista_segmentos
})


# In[13]:


# Distribuição da Variável Target - Percentual
percentual_target = contagem_target / len(df_segmento) * 100
px.bar(percentual_target, color=percentual_target.index, category_orders={
  'segmento_de_cliente': lista_segmentos
})


# In[14]:


# Distribuição da Variável Localização - Percentual
percentual_localizacao = df_segmento.value_counts('localizacao') / len(df_segmento) * 100
px.bar(percentual_localizacao, color=percentual_localizacao.index)


# In[15]:


# Distribuição da Variável Atividade Econômica - Percentual
percentual_atividade = df_segmento.value_counts('atividade_economica') / len(df_segmento) * 100
px.bar(percentual_atividade, color=percentual_atividade.index)


# In[16]:


# Distribuição da Variável Inovação - Percentual
percentual_inovacao = df_segmento.value_counts('inovacao') / len(df_segmento) * 100
px.bar(percentual_inovacao, color=percentual_inovacao.index)


# In[17]:


# Tabela de Contingência entre Localização e Target
crosstab_localizacao = pd.crosstab(df_segmento['localizacao'], df_segmento['segmento_de_cliente'], margins=True)[lista_segmentos].reset_index()

tabela_localizacao = ff.create_table(crosstab_localizacao)

# Mostrar a Crosstab
tabela_localizacao.show()


# In[18]:


# Tabela de Contingência entre Atividade e Target
crosstab_atividade = pd.crosstab(df_segmento['atividade_economica'], df_segmento['segmento_de_cliente'], margins=True)[lista_segmentos].reset_index()

tabela_atividade = ff.create_table(crosstab_atividade)

# Mostrar a Crosstab
tabela_atividade.show()


# In[19]:


# Tabela de Contingência entre Inovação e Target
crosstab_inovacao = pd.crosstab(df_segmento['inovacao'], df_segmento['segmento_de_cliente'], margins=True)[lista_segmentos].reset_index()

tabela_inovacao = ff.create_table(crosstab_inovacao)

# Mostrar a Crosstab
tabela_inovacao.show()


# In[20]:


# Distribuição Idade da Empresa
px.histogram(df_segmento, x='idade')


# In[21]:


# Distribuição Faturamento Mensal da Empresa
px.histogram(df_segmento, x='faturamento_mensal')


# In[22]:


# BoxPlot entre Idade e Segmento
px.box(df_segmento, x='segmento_de_cliente', y='idade', color='segmento_de_cliente', category_orders={
  'segmento_de_cliente': lista_segmentos
})


# In[23]:


# BoxPlot entre Faturamento Mensal e Segmento
px.box(df_segmento, x='segmento_de_cliente', y='faturamento_mensal', color='segmento_de_cliente', category_orders={
  'segmento_de_cliente': lista_segmentos
})


# In[24]:


# Teste de Qui-Quadrado de Pearson
# H0 - as variáveis são independentes
# H1 - as variáveis não são independentes
# Se p-value > 0.05, aceita a hipótese nula, caso contrário rejeita

valor_esperado, valor_observado, estatísticas = pg.chi2_independence(df_segmento, 'segmento_de_cliente', 'inovacao')


# In[25]:


# Valor Esperado
# É a frequência que seria esperada se não houvesse associação entre as variáveis
# É calculado utilizando a distribuição assumida no teste qui-quadrado
valor_esperado


# In[26]:


# Valor Observado
# É a frequência real dos dados coletados
valor_observado


# In[27]:


# Estatísticas
estatísticas.round(5)


# As variáveis localização e segmento de clientes são independentes. -------------------- Qui-Quadrado (p-value = 0.81714)

# As variáveis atividade econômica e segmento de clientes são independentes. --------- Qui-Quadrado (p-value = 0.0.35292)

# As variáveis inovação e segmento de clientes são independentes. ----------------------- Qui-Quadrado (p-value = 0.0)

# ### Treinamento do Modelo

# In[28]:


# Separar X e y
X = df_segmento.drop(columns=['segmento_de_cliente'])
y = df_segmento['segmento_de_cliente']


# In[29]:


# Pipeline
# OneHotEncode nas variáveis categóricas
# Treinamento do Modelo

# Lista de variáveis categóricas
categorical_features = ['atividade_economica', 'localizacao']

# Criar um transformador de variáveis categóricas usando OneHotEncoder
categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
  transformers = [
    ('cat', categorical_transformer, categorical_features)
  ]
)

# Pipeline com Pre-Processor e o Modelo de Árvore de Decisão
dt_model = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('classifier', DecisionTreeClassifier())
])


# ### Validação Cruzada

# In[30]:


# Treinar o Modelo com Validação Cruzada, usandop StratifiedKFold, dado que as classes estão desbalanceadas

cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=51)
metrics_result = cross_validate(dt_model, X, y, cv=cv_folds, scoring=['accuracy'], return_estimator=True)


# In[31]:


# Mostrar Retorno do Cross Validation
metrics_result


# In[32]:


# Média da Acurácia, considerando os 3 splits
metrics_result['test_accuracy'].mean()


# In[33]:


# Acurácia
# total de previsões corretas / total de previsões


# ### Métricas

# In[34]:


# Fazendo predições usando Cross Validation
y_pred = cross_val_predict(dt_model, X, y, cv=cv_folds)


# In[35]:


# Avaliar o desempenho do modelo
classification_report_str = classification_report(y, y_pred)

print(f'Relatório de Classificação:\n{classification_report_str}')


# In[36]:


# Mostrar Matriz de Confusão
confusion_matrix_modelo = confusion_matrix(y, y_pred, labels=lista_segmentos)
disp = ConfusionMatrixDisplay(confusion_matrix_modelo, display_labels=lista_segmentos)
disp.plot()


# ### Tuning de Hiperparâmetros

# In[37]:


# Ajustar hiperparâmetros do Modelo usando Optuna
# min_samples_leaf = Mínimo de instâncias requerido para formar uma folha (nó terminal)
# max_depth = Profundidade máxima da árvore

def decisiontree_optuna(trial):

  min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
  max_depth = trial.suggest_int('max_depth', 2, 8)

  dt_model.set_params(classifier__min_samples_leaf=min_samples_leaf)
  dt_model.set_params(classifier__max_depth=max_depth)

  scores = cross_val_score(dt_model, X, y, cv=cv_folds, scoring='accuracy')

  return scores.mean()


# In[38]:


# Executar a automação de experimentos
estudo_decisiontree = optuna.create_study(direction='maximize')
estudo_decisiontree.optimize(decisiontree_optuna, n_trials=200)


# In[39]:


# Mostrar melhor resultado e melhor conjunto de hiperparâmetros
print(f'Melhor acurácia: {estudo_decisiontree.best_value}')
print(f'Melhores parâmetros: {estudo_decisiontree.best_params}')


# ### Visualizar Árvore

# In[40]:


# Preparar o Conjunto de Dados para treinar e conseguir visualizar a árvore
X_train_tree = X.copy()
X_train_tree['localizacao_label'] = X_train_tree.localizacao.astype('category').cat.codes
X_train_tree['atividade_economica_label'] = X_train_tree.atividade_economica.astype('category').cat.codes
X_train_tree.drop(columns=['localizacao','atividade_economica'], axis=1, inplace=True)
X_train_tree.rename(columns={'localizacao_label': 'localizacao','atividade_economica_label': 'atividade_economica'}, inplace=True)
X_train_tree.head(5)


# In[41]:


# Treinar o modelo com o conjunto de hiperparâmetros ideal

clf_decisiontree = DecisionTreeClassifier(
  min_samples_leaf=estudo_decisiontree.best_params['min_samples_leaf'],
  max_depth=estudo_decisiontree.best_params['max_depth']
)

y_train_tree = y.copy()

clf_decisiontree.fit(X_train_tree, y_train_tree)


# In[42]:


# Visualizar Árvore de Decisão com Plot Tree
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize=(10,10), dpi=600)

plot_tree(
  clf_decisiontree, 
  feature_names=X_train_tree.columns.to_numpy(),
  class_names=lista_segmentos,
  filled = True
)


# ### Salvar Modelo

# In[43]:


import joblib

# Criar um pipeline "tunado"
dt_model_tunado = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('classifier', DecisionTreeClassifier(
    min_samples_leaf=estudo_decisiontree.best_params['min_samples_leaf'],
    max_depth=estudo_decisiontree.best_params['max_depth']
    )
  )
])

# Treinar Modelo Tunado
dt_model_tunado.fit(X, y)

# Salvar Modelo
joblib.dump(dt_model_tunado, 'model.pkl')


# ### Entregar modelo como App de Predição Batch (por arquivo)

# In[45]:


import gradio as gr

modelo = joblib.load('./model.pkl')

def predict(arquivo):
  df_empresas = pd.read_csv(arquivo.name)
  y_pred = modelo.predict(df_empresas)
  df_segmentos = pd.DataFrame(y_pred, columns=['segmento_de_cliente'])
  df_predicoes = pd.concat([df_empresas, df_segmentos], axis=1)
  df_predicoes.to_csv('./predicoes.csv', index=False)

  return './predicoes.csv'

demo = gr.Interface(
  predict,
  gr.File(file_types=[".csv"]),
  "file"
)

demo.launch()

