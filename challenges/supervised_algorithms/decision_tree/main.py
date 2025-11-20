# %%
import joblib
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

pio.renderers.default = "png"

# %%
df_transactions = pd.read_csv("./dataset/transacoes_fraude.csv")
# %%
df_transactions.rename(
    columns={col: col.replace(" ", "_") for col in df_transactions.columns},
    inplace=True,
)
df_transactions.drop(columns=["Cliente"], inplace=True)

# %%
df_transactions.head(5)

# %%
df_transactions.info()


# %%
df_transactions["Tipo_de_Transacao"].unique()

# %%
contagem_fraudes = df_transactions["Classe"].value_counts()
print(contagem_fraudes)

# %%
lista_fraude = ["0", "1"]
px.bar(
    contagem_fraudes,
    color=contagem_fraudes.index,
    category_orders={"fraude": lista_fraude},
)

# %%
percentual_fraudes = contagem_fraudes / len(df_transactions) * 100
px.bar(
    percentual_fraudes,
    color=percentual_fraudes.index,
    category_orders={"fraude": lista_fraude},
)

# %%
percentual_tipo_transacao = (
    df_transactions.value_counts("Tipo_de_Transacao") / len(df_transactions) * 100
)
px.bar(percentual_tipo_transacao, color=percentual_tipo_transacao.index)

# %%
crosstab_tipo_transacao = pd.crosstab(
    df_transactions["Classe"], df_transactions["Tipo_de_Transacao"], margins=True
).reset_index()
tabela_tipo_transacao = ff.create_table(crosstab_tipo_transacao)
tabela_tipo_transacao.show()

# %%
px.histogram(df_transactions, x="Valor_da_Transacao")

# %%
px.histogram(df_transactions, x="Valor_Anterior_a_Transacao")

# %%
px.histogram(df_transactions, x="Valor_Apos_a_Transacao")

# %%
px.box(
    df_transactions,
    x="Classe",
    y="Valor_da_Transacao",
    color="Classe",
    category_orders={"fraude": lista_fraude},
)

# %%
px.box(
    df_transactions,
    x="Classe",
    y="Valor_Anterior_a_Transacao",
    color="Classe",
    category_orders={"fraude": lista_fraude},
)

# %%
px.box(
    df_transactions,
    x="Classe",
    y="Valor_Apos_a_Transacao",
    color="Classe",
    category_orders={"fraude": lista_fraude},
)

# %%
_, _, estatísticas = pg.chi2_independence(
    df_transactions, "Classe", "Tipo_de_Transacao"
)

# %%
estatísticas.round(5)

# As variáveis Tipo_de_Transacao e Classe são dependentes ----> Qui-quadrado (p-value = 0.034 < 0.05)
# As variáveis Horario_da_Transacao e Classe são independentes ----> Qui-quadrado (p-value = 0.49 < 0.05)
