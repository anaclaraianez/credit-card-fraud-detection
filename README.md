# credit-card-fraud-detection
Detecção de fraudes em transações de cartão de crédito usando Machine Learning, técnicas de balanceamento, XGBoost e SHAP.

  Credit Card Fraud Detection with Machine Learning Project
  
Projeto de Machine Learning desenvolvido para detectar transações fraudulentas em cartões de crédito, com foco especial no desafio de trabalhar com dados altamente desbalanceados.

O projeto foi desenvolvido como parte de um desafio prático de Data Science, explorando desde a preparação dos dados até a comparação e interpretação de diferentes modelos de classificação.


  Objetivo
  
O objetivo foi desenvolver modelos capazes de identificar transações fraudulentas em uma base na qual fraudes representam aproximadamente 0,17% das observações.

Esse desbalanceamento torna a acurácia uma métrica potencialmente enganosa. Um modelo que classificasse praticamente todas as operações como legítimas poderia apresentar uma acurácia elevada e, ainda assim, falhar justamente na identificação das fraudes.


  Por isso, a análise deu maior atenção a:

-Precision
-Recall
-F1-score
-ROC Curve
-Precision-Recall Curve
-Com destaque especial para o recall da classe de fraude.


  Dataset

A base contém:
-Time
-Amount
-Class
-variáveis V1 a V28
-As variáveis V1 a V28 foram transformadas utilizando PCA para preservar a privacidade das informações originais.


  A variável Class representa:
0 → transação normal
1 → fraude
A análise inicial confirmou um forte desbalanceamento, com aproximadamente 0,17% das transações pertencendo à classe de fraude.


  Preparação dos dados
Durante a preparação foram realizadas etapas como:
-exploração inicial com Pandas;
-transformação logarítmica de Amount;
-padronização com StandardScaler;
-separação entre treino e teste;
-uso de stratify para preservar a proporção das classes;
-aplicação de técnicas específicas para dados desbalanceados.
-Tratamento do desbalanceamento


  Foram testadas duas abordagens:

  Undersampling
O número de exemplos da classe majoritária foi reduzido durante o treinamento.
Resultado para a classe fraude:
Precision: 0.05
Recall: 0.88
F1-score: 0.10
O modelo apresentou alto recall, conseguindo identificar grande parte das fraudes, porém com baixa precisão.

  SMOTE
Também foi utilizado SMOTE, aumentando a representação da classe minoritária no conjunto de treinamento.
Resultado para a classe fraude:
Precision: 0.13
Recall: 0.85
F1-score: 0.23
O SMOTE apresentou recall ligeiramente menor que o undersampling, mas melhorou precision e F1-score.

Isso evidencia um dos principais aprendizados do projeto: não existe necessariamente uma única estratégia de balanceamento ideal. O resultado deve ser avaliado considerando o objetivo do problema.


  Modelos utilizados
Foram explorados:
-Logistic Regression
-Random Forest
-XGBoost


  Também foram utilizados:
-ajuste do limiar de decisão (threshold);
-pesos de classe;
-GridSearchCV;
-SMOTE;
-Random Undersampling.


  Resultado do XGBoost
O XGBoost apresentou um comportamento significativamente mais equilibrado na classe de fraude:
Métrica	Resultado
Precision	0.94
Recall	0.78
F1-score	0.85
Embora o recall seja inferior ao observado com undersampling e SMOTE, o modelo apresentou precision e F1-score consideravelmente maiores.Esse resultado demonstra o trade-off existente entre detectar o maior número possível de fraudes e evitar falsos positivos.


  Ajuste do limiar de decisão
Além do threshold padrão do classificador, também foi testado um limiar de:
0.30
O objetivo foi observar como uma alteração no ponto de decisão influencia a relação entre Precision e Recall. Esse ajuste pode ser relevante porque o custo de deixar uma fraude passar pode ser diferente do custo de gerar um falso alerta.

  
  Avaliação
Além do Classification Report, foram utilizadas:
  ROC Curve
A curva ROC foi utilizada para analisar a capacidade do classificador de separar as duas classes em diferentes limiares.
  Precision-Recall Curve
Como o conjunto de dados é altamente desbalanceado, a relação entre Precision e Recall ganhou atenção especial durante a avaliação.


  Feature Importance
A análise de importância das variáveis no XGBoost mostrou destaque para:
V10
V17
V4
V12
V14 apresentou a maior importância relativa no modelo treinado.
Como V1 a V28 são componentes transformados por PCA, a análise indica a relevância dessas variáveis para as previsões, mas não permite atribuir diretamente um significado financeiro específico a cada componente.


  Explicabilidade com SHAP
SHAP foi utilizado para complementar a análise de importância e entender como as variáveis contribuíram para as previsões do modelo. A análise permite observar não apenas quais características são relevantes, mas também como diferentes valores influenciam a saída do modelo.


  Otimização com GridSearchCV
O XGBoost também foi submetido a uma busca de hiperparâmetros utilizando GridSearchCV, com recall como métrica de otimização.
Foram testados diferentes valores de:
-max_depth
-n_estimators
A melhor combinação encontrada durante o experimento foi:
-max_depth = 5
-n_estimators = 100


  Principais aprendizados
Este projeto reforçou alguns pontos importantes sobre Machine Learning aplicado a problemas desbalanceados:

-alta acurácia não significa necessariamente um bom modelo;
-Precision, Recall e F1-score precisam ser analisados em conjunto;
-maximizar Recall pode aumentar falsos positivos;
-técnicas como SMOTE e undersampling produzem comportamentos diferentes;
-ajustar o threshold pode alterar significativamente a decisão do classificador;
-modelos como XGBoost podem oferecer um equilíbrio interessante entre Precision e Recall;
-Feature Importance e SHAP ajudam a compreender melhor as decisões do modelo.
-Mais importante do que simplesmente comparar algoritmos foi entender os trade-offs envolvidos em cada estratégia.

  Diferenciais explorados
Além do pipeline básico de classificação, o projeto explorou:
-comparação entre undersampling e SMOTE;
-ajuste manual do threshold;
-Random Forest com class_weight;
-XGBoost com tratamento do desbalanceamento;
-otimização de hiperparâmetros focada em Recall;
-Feature Importance;
-explicabilidade com SHAP.

  Tecnologias
-Python
-Pandas
-NumPy
-Matplotlib
-Scikit-learn
-Imbalanced-learn
-XGBoost
-SHAP
