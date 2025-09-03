# 🔍 Avaliação de Modelos de Classificação com Métricas Detalhadas

Este projeto tem como objetivo calcular e visualizar métricas de desempenho para modelos de classificação multiclasse, utilizando uma matriz de confusão normalizada. As métricas são calculadas individualmente para cada classe, permitindo uma análise precisa e completa do comportamento do modelo.

---

## 📊 Funcionalidades

- Cálculo dos componentes da matriz de confusão:
  - VP (Verdadeiros Positivos)
  - FP (Falsos Positivos)
  - FN (Falsos Negativos)
  - VN (Verdadeiros Negativos)

- Cálculo das métricas de avaliação por classe:
  - *Acurácia*
  - *Precisão*
  - *Recall (Sensibilidade)*
  - *Especificidade*
  - *F1-Score*

- Visualização da matriz de confusão como heatmap com escala de intensidade.

---

## 🧠 Lógica de Cálculo

Para cada classe i, o modelo é avaliado como um problema binário (um contra todos):

- VP = con_mat_norm[i, i]  
- FN = soma da linha i - VP  
- FP = soma da coluna i - VP  
- VN = soma total - (VP + FP + FN)

As métricas são então calculadas com proteção contra divisão por zero:

```python
precisao = VP / (VP + FP) if (VP + FP) else 0
recall = VP / (VP + FN) if (VP + FN) else 0
especificidade = VN / (VN + FP) if (VN + FP) else 0
f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) else 0
acuracia = (VP + VN) / np.sum(con_mat_norm)

🛠 Requisitos

- Python 3.x
- Bibliotecas:
  - numpy
  - matplotlib
  - seaborn
