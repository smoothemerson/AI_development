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