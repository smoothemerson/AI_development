# %%
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

pio.renderers.default = "png"

# %%
df_transactions = pd.read_csv("./dataset/transacoes_fraude.csv")
# %%
df_transactions.rename(
    columns={col: col.replace(" ", "_") for col in df_transactions.columns},
    inplace=True,
)
df_transactions.drop(columns=["Cliente"], inplace=True)

df_transactions["Horario"] = pd.to_datetime(
    df_transactions["Horario_da_Transacao"]
).dt.hour
df_transactions.drop(columns=["Horario_da_Transacao"], inplace=True)

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

# %%


# %%
categorical_features = ["Tipo_de_Transacao"]
numerical_features = [
    "Valor_da_Transacao",
    "Valor_Anterior_a_Transacao",
    "Valor_Apos_a_Transacao",
    "Horario",
]

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ],
)
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("standard", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

parametros = {
    "classifier__criterion": [
        "gini",
        "entropy",
        "log_loss",
    ],
    "classifier__max_depth": [None, 5, 10, 20, 30],
    "classifier__min_samples_split": [
        2,
        5,
        10,
        20,
        30,
        40,
        50,
    ],
    "classifier__min_samples_leaf": [
        1,
        2,
        5,
        10,
        20,
    ],
    "classifier__max_features": [
        None,
        "sqrt",
        "log2",
    ],
    "classifier__splitter": [
        "best",
        "random",
    ],
    "classifier__class_weight": [
        None,
        "balanced",
    ],
}

modelo_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(random_state=42),
        ),
    ]
)

model_dt = GridSearchCV(
    estimator=modelo_pipeline,
    param_grid=parametros,
    scoring="f1",
    cv=5,
    verbose=1,
    n_jobs=-1,
)

# %%
X = df_transactions.drop(columns=["Classe"], axis=1)
y = df_transactions["Classe"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=51
)

# %%
model_dt.fit(X_train, y_train)

# %%
print("Melhores parâmetros:", model_dt.best_params_)
modelo_otimizado = model_dt.best_estimator_
y_pred = modelo_otimizado.predict(X_test)

# %%
classification_report_str = classification_report(y_test, y_pred)
print(f"Relatório de Classificação:\n{classification_report_str}")

# %%
confusion_matrix_modelo_baseline = confusion_matrix(y_test, y_pred)
disp_modelo_baseline = ConfusionMatrixDisplay(confusion_matrix_modelo_baseline)
disp_modelo_baseline.plot()

# %%
