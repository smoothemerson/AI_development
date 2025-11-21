# Classificação de Fraudes em Transações Bancárias

## Objetivo:
O objetivo do desafio é construir um modelo de classificação utilizando árvore de decisão que seja capaz de prever com alta precisão se uma transação é fraudulenta ou não. O modelo deve ser avaliado com base em métricas que considerem tanto a capacidade de detectar fraudes (sensibilidade) quanto a capacidade de evitar falsos alarmes (especificidade). 

## Dataset:
Descrição do Dataset
O dataset fornecido contém informações sobre transações bancárias realizadas pelos clientes do Banco SeguraMais. Cada linha do dataset representa uma transação, e as colunas contêm informações relevantes sobre a transação e seu status (fraude ou não fraude). Abaixo está a descrição das variáveis presentes no dataset:

- Cliente: Identificador único do cliente que realizou a transação
- Tipo de Transação: O tipo de transação realizada (ex.: Saque, PIX, Débito, Crédito)
- Valor da Transação: O valor monetário da transação
- Valor Anterior à Transação: O saldo do cliente antes da transação
- Valor Após a Transação: O saldo do cliente após a transação
- Horário da Transação: O horário em que a transação foi realizada
- Classe: A variável alvo, indicando se a transação foi fraudulenta (1) ou legítima (0)

## Resultados:
Após a implementação do modelo de árvore de decisão, os resultados obtidos foram os seguintes:
### Matriz de Confusão e Métricas de Desempenho

Com base nos dados de teste, a matriz de confusão apresentou os seguintes resultados:

- **Verdadeiros Positivos (VP)**: 193 fraudes detectadas corretamente
- **Falsos Negativos (FN)**: 239 fraudes não detectadas pelo modelo
- **Verdadeiros Negativos (VN)**: 2.615 transações legítimas classificadas corretamente
- **Falsos Positivos (FP)**: 853 transações legítimas classificadas erroneamente como fraude

**Métricas Calculadas:**
- **Sensibilidade (Recall)**: 44,68% - O modelo detecta aproximadamente 4 em cada 9 fraudes reais
- **Especificidade**: 75,40% - O modelo identifica corretamente a maioria das transações legítimas

### Análise Crítica do Desempenho

#### 🔍 **Limitações Críticas Identificadas**

**1. Capacidade Moderada de Detecção de Fraudes**
O modelo apresenta uma **sensibilidade moderada (44,68%)**, significando que 193 das 432 fraudes reais foram detectadas. Representa uma limitação para um sistema de detecção de fraudes, pois:
- **55,32% das fraudes ainda passam despercebidas**, representando um risco financeiro significativo
- A instituição permanece exposta a perdas por fraudes não detectadas
- A confiança dos clientes pode ser comprometida por transações fraudulentas não bloqueadas

**2. Desbalanceamento Severo entre Classes**
O dataset evidencia um forte desbalanceamento, com fraudes representando apenas cerca de 11% do total de transações. Esta característica levou o modelo a desenvolver um viés conservador, priorizando a classificação da classe majoritária (transações legítimas).

**3. Alta Taxa de Falsos Alarmes**
A especificidade de 75,40% significa que **24,60% das transações legítimas são incorretamente sinalizadas como fraude**. Isso resulta em:
- Inconvenientes significativos para clientes legítimos
- Sobrecarga operacional considerável para análise manual
- Possível perda de clientes por bloqueios desnecessários

#### ✅ **Aspectos Positivos**

**Boa na Detecção de Fraudes**
O modelo demonstra uma boa capacidade de detectar fraudes, detectando quase metade das fraudes reais, o que representa um avanço importante no sistema de segurança.

#### 🚀 **Recomendações para Melhoria**

**1. Estratégias de Balanceamento de Classes**
- Aplicar **undersampling** inteligente da classe majoritária
- Utilizar **ensemble methods** com diferentes estratégias de amostragem

**2. Otimização de Algoritmos**
- Experimentar **Random Forest** com parâmetro `class_weight='balanced'`
- Implementar **XGBoost** com ajuste de `scale_pos_weight`
- Explorar algoritmos baseados em **detecção de anomalias**

**3. Otimização de Métricas e Threshold**
- Focar na otimização do **F1-score** em vez da acurácia geral
- Implementar **threshold customizado** que priorize a detecção de fraudes
- Utilizar **validação cruzada estratificada** para melhor avaliação

**4. Métricas de Avaliação Complementares**
- Analisar **curva ROC** e **AUC** para melhor compreensão do desempenho
- Implementar **curva Precision-Recall** específica para classes desbalanceadas
- Calcular **custo-benefício** considerando perdas financeiras reais

### Conclusão

O modelo atual apresenta **boa detecção de fraudes** com sensibilidade de 44,68%, porém ainda enfrenta desafios importantes com alta taxa de falsos positivos. É fundamental implementar as melhorias sugeridas para encontrar um equilíbrio mais adequado entre detecção de fraudes e redução de falsos alarmes, considerando que ambos os aspectos são críticos para a eficácia operacional do sistema.
