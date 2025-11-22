# Predição de Diabetes em Pacientes

## Objetivo:
O objetivo do desafio é construir um modelo de classificação utilizando Naive Bayes que seja capaz de prever com alta precisão se um paciente apresenta diabetes ou não. Este desafio promove a aplicação prática de técnicas de classificação para diagnóstico médico, utilizando dados de glicemia e pressão arterial.

## Dataset:
### Descrição do Dataset
O dataset fornecido contém informações sobre pacientes e seus indicadores de saúde relacionados ao diabetes. Cada linha do dataset representa um paciente, e as colunas contêm informações relevantes sobre os parâmetros clínicos e o status de diabetes. Abaixo está a descrição das variáveis presentes no dataset:

- **Glicemia**: Nível de glicose no sangue do paciente (mg/dL)
- **Pressão Arterial**: Medida da pressão arterial do paciente (mmHg)
- **Diabetes**: A variável alvo, indicando se o paciente tem diabetes (1) ou não tem diabetes (0)

## Resultados:
Após a implementação do modelo Naive Bayes (Gaussiano), os resultados obtidos foram os seguintes:

### Matriz de Confusão e Métricas de Desempenho

Com base nos dados de teste (199 amostras, representando 20% do total de 995 pacientes), a matriz de confusão apresentou os seguintes resultados:

- **Verdadeiros Positivos (VP)**: 86 casos de diabetes detectados corretamente
- **Falsos Negativos (FN)**: 8 casos de diabetes não detectados pelo modelo
- **Verdadeiros Negativos (VN)**: 97 casos sem diabetes classificados corretamente
- **Falsos Positivos (FP)**: 8 casos sem diabetes classificados erroneamente como diabetes

**Métricas Calculadas:**
- **Sensibilidade (Recall)**: 91,49% - O modelo detecta aproximadamente 9 em cada 10 casos reais de diabetes
- **Especificidade**: 92,38% - O modelo identifica corretamente a maioria dos pacientes saudáveis
- **Precisão**: 91,49% - Das predições positivas, aproximadamente 9 em cada 10 estão corretas
- **Acurácia**: 92,00% - Taxa geral de acertos do modelo
- **F1-Score**: 92,00% - Média harmônica entre precisão e sensibilidade

### Conclusão

O modelo Naive Bayes demonstrou **performance excepcional** para predição de diabetes, com métricas consistentemente acima de 91%. A combinação de alta sensibilidade (91,49%) e alta especificidade (92,38%) torna este modelo altamente adequado para aplicação clínica real.
