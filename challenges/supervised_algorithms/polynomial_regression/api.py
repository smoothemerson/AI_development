from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Criar uma instância do FastAPI
app = FastAPI()

# Criar uma classe com os dados de entrada que virão no request body com os tipos esperados
class request_body(BaseModel):
  tempo_de_experiencia: int
  numero_de_vendas: int
  fator_sazonal: int

# Carregar model para realizar a predição
modelo_poly = joblib.load('./model.pkl')

@app.post('/predict')
def predict(data : request_body):

  input_features = {
    'tempo_de_experiencia': data.tempo_de_experiencia,
    'numero_de_vendas': data.numero_de_vendas,
    'fator_sazonal': data.fator_sazonal
  }

  pred_df = pd.DataFrame(input_features, index=[1])

  # Predição
  y_pred = modelo_poly.predict(pred_df)[0].astype(float)

  return {'receita_em_reais': y_pred.tolist()}