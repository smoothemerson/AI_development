# %%
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pio.renderers.default = "png"

# %%
df = pd.read_csv("./dataset/diabetes.csv")

# %%
df.head(5)

# %%
df.info()

# %%
df.describe()

# %%
px.bar(df.value_counts("diabetes"))

# %%
px.bar(df.value_counts("diabetes") / len(df) * 100)

# %%
px.histogram(df, x="glicemia")

# %%
px.histogram(df, x="pressao_arterial")

# %%
px.box(df, y="glicemia")

# %%
px.box(df, y="pressao_arterial")

# %%
px.box(
    df,
    x="diabetes",
    y="glicemia",
    color="diabetes",
)

# %%
px.box(
    df,
    x="diabetes",
    y="pressao_arterial",
    color="diabetes",
)

# %%
numerical_features = ["glicemia", "pressao_arterial"]
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[("num", numerical_transformer, numerical_features)]
)

# %%
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
model_baseline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB()),
    ]
)

# %%
model_baseline.fit(X_train, y_train)

# %%
y_pred = model_baseline.predict(X_test)

# %%
classification_report_str = classification_report(y_test, y_pred)
print(f"Relatório de Classificação:\n{classification_report_str}")

# %%
confusion_matrix_modelo_baseline = confusion_matrix(y_test, y_pred)
disp_modelo_baseline = ConfusionMatrixDisplay(confusion_matrix_modelo_baseline)
disp_modelo_baseline.plot()

# %%
