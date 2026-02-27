# 📡 Predicción de Churn en TelecomX

## 🎯 Propósito del Análisis

Este proyecto desarrolla modelos de machine learning para **predecir el churn (cancelación) de clientes** en TelecomX, una empresa de telecomunicaciones. El objetivo principal es identificar, con anticipación, qué clientes tienen mayor probabilidad de abandonar el servicio, permitiendo a la empresa implementar estrategias de retención proactivas y personalizadas.

El foco del modelado está en maximizar el **Recall** para la clase `Churn`, minimizando los falsos negativos (clientes que abandonan pero no son detectados), ya que dejar pasar un cliente en riesgo tiene un costo mayor para el negocio que actuar preventivamente sobre uno que no cancelaría.

---

## 🗂️ Estructura del Proyecto

```
challenge_telecomx_latam2/
│
├── Challenge2_TelecomX_LATAM_.ipynb                # Cuaderno principal con todo el análisis
├── datos_tratados.csv                              # Dataset limpio y preprocesado
├── coeficientes_modelo_optimizado.csv              # Coeficientes del modelo final exportados
├── tenure_churn.png                                # Boxplot: Tenure vs Churn
├── chargesmonthly_churn.png                        # Boxplot: ChargesMonthly vs Churn
├── matriz_correlacion_numericas.png                # Heatmap: variables numéricas
├── matriz_correlacion_todas_variables.png          # Heatmap: todas las variables codificadas
├── matriz_confusion_desiciontree.png               # Matriz de confusión: Decision Tree
├── matriz_confusion_randomforest.png               # Matriz de confusión: Random Forest
├── matriz_confusion_randomforest_optimizado.png    # Matriz de confusión: Random Forest optimizado
├── matriz_confusion_regresionlogistica.png         # Matriz de confusión: Regresión Logística
├── matriz_confusion_regresionlogistica_optimizada.png  # Matriz de confusión: RL optimizada
└── README.md                                       # Documentación del proyecto
```


---

## 🔧 Preparación de los Datos

### Clasificación de Variables

**Variables Numéricas:**
| Variable | Descripción |
|---|---|
| `Tenure` | Antigüedad del cliente (meses) |
| `ChargesMonthly` | Cargos mensuales |
| `ChargesTotal` | Cargos totales acumulados |
| `ChargesDaily` | Cargos diarios estimados |

**Variables Categóricas:**
| Variable | Descripción |
|---|---|
| `Gender` | Género del cliente |
| `SeniorCitizen` | Si es ciudadano de la tercera edad |
| `Partner` / `Dependents` | Si tiene pareja o dependientes |
| `InternetService` | Tipo de servicio de internet (DSL, Fibra, Ninguno) |
| `Contract` | Tipo de contrato (mes a mes, 1 año, 2 años) |
| `PaymentMethod` | Método de pago |
| `PaperlessBilling` | Facturación sin papel |
| Servicios adicionales | `OnlineSecurity`, `TechSupport`, `StreamingTV`, etc. |

**Variable Objetivo:**
- `Churn`: Indica si el cliente canceló el servicio (`Yes` / `No`)

---

### Etapas de Preprocesamiento

#### 1. Eliminación de Columnas Irrelevantes
- **`CustomerID`**: Identificador único sin valor predictivo → eliminado.
- **`ChargesDaily` y `ChargesTotal`**: Presentaban alta colinealidad con `ChargesMonthly` y `Tenure` → eliminadas para evitar multicolinealidad.
- **`Gender` y `PhoneService`**: Test Chi-cuadrado reveló relación estadísticamente no significativa con `Churn` → eliminadas.

#### 2. Tratamiento de Outliers
Se aplicó **capping por IQR** (Rango Intercuartílico) sobre `Tenure` y `ChargesMonthly`. Los valores por encima del límite superior (`Q3 + 1.5×IQR`) o debajo del límite inferior (`Q1 - 1.5×IQR`) fueron recortados, preservando la distribución sin eliminar registros.

#### 3. Codificación de Variables Categóricas
Se aplicó **One-Hot Encoding** (`pd.get_dummies`) sobre todas las variables categóricas restantes, generando columnas binarias por cada categoría. Esto evita que el modelo interprete orden donde no existe.

#### 4. Balanceo de Clases
El dataset presentaba desbalance: ~73% `No Churn` vs ~27% `Churn`. Se abordó de dos formas:
- **SMOTE** (Synthetic Minority Over-sampling Technique): para Random Forest y Decision Tree, genera ejemplos sintéticos de la clase minoritaria en el conjunto de entrenamiento.
- **`class_weight='balanced'`**: para Regresión Logística, ajusta automáticamente los pesos internos sin generar datos nuevos.

---

### Separación Entrenamiento / Prueba

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **80% entrenamiento / 20% prueba**
- Se usó `stratify=y` para mantener la proporción de clases en ambos conjuntos.
- SMOTE se aplicó **únicamente sobre el conjunto de entrenamiento** para evitar data leakage.

---

### Justificaciones de Modelado

| Decisión | Justificación |
|---|---|
| Foco en Recall alto | Detectar todos los posibles churns es más valioso que evitar falsas alarmas |
| GridSearchCV para optimización | Reduce el overfitting observado en modelos de árbol con hiperparámetros por defecto |
| Regresión Logística como modelo final | Mejor balance entre Recall (0.79) y generalización; menor brecha train/test |
| SMOTE solo en train | Previene contaminación del set de prueba con datos sintéticos |

---

## 📊 EDA: Gráficos e Insights

### Antigüedad (Tenure) vs. Churn
Los clientes con **menor antigüedad** concentran la mayor proporción de churn. A mayor tiempo con la empresa, menor probabilidad de cancelación, siendo `Tenure` el factor protector más fuerte del modelo.

![Tenure vs Churn](https://raw.githubusercontent.com/Solsmi/challenge_telecomx_latam2/main/tenure_churn.png)

---

### Cargos Mensuales vs. Churn
Los clientes que cancelaron el servicio presentan en promedio **cargos mensuales más altos**. Existe una concentración notable de churn en los tramos de mayor costo.

![ChargesMonthly vs Churn](https://raw.githubusercontent.com/Solsmi/challenge_telecomx_latam2/main/chargesmonthly_churn.png)

---

### Matriz de Correlación — Variables Numéricas
Se identificó alta correlación entre `ChargesMonthly`, `ChargesTotal` y `Tenure`, lo que justificó la eliminación de variables redundantes para evitar multicolinealidad.

![Matriz de Correlación Numéricas](https://raw.githubusercontent.com/Solsmi/challenge_telecomx_latam2/main/matriz_correlacion_numericas.png)

---

### Matriz de Correlación — Todas las Variables Codificadas
Visión completa de las correlaciones tras el One-Hot Encoding, destacando las variables con mayor relación con `Churn`.

![Matriz de Correlación Todas las Variables](https://raw.githubusercontent.com/Solsmi/challenge_telecomx_latam2/main/matriz_correlacion_todas_variables.png)

---

### Matrices de Confusión por Modelo

**Decision Tree**

![Matriz de Confusión Decision Tree](https://raw.githubusercontent.com/Solsmi/challenge_telecomx_latam2/main/matriz_confusion_desiciontree.png)

**Random Forest**

![Matriz de Confusión Random Forest](https://raw.githubusercontent.com/Solsmi/challenge_telecomx_latam2/main/matriz_confusion_randomforest.png)

**Random Forest Optimizado**

![Matriz de Confusión Random Forest Optimizado](https://raw.githubusercontent.com/Solsmi/challenge_telecomx_latam2/main/matriz_confusion_randomforest_optimizado.png)

**Regresión Logística**

![Matriz de Confusión Regresión Logística](https://raw.githubusercontent.com/Solsmi/challenge_telecomx_latam2/main/matriz_confusion_regresionlogistica.png)

**Regresión Logística Optimizada** ✅ Modelo seleccionado

![Matriz de Confusión Regresión Logística Optimizada](https://raw.githubusercontent.com/Solsmi/challenge_telecomx_latam2/main/matriz_confusion_regresionlogistica_optimizada.png)

---

### Coeficientes del Modelo Final
Los coeficientes de la Regresión Logística Optimizada están exportados en `coeficientes_modelo_optimizado.csv`, permitiendo identificar el peso e impacto de cada variable sobre la predicción de churn.

---

## ▶️ Instrucciones de Ejecución

### Requisitos Previos

Asegúrate de tener Python 3.8+ instalado. Luego instala las dependencias:

```bash
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn numpy
```

### Cargar los Datos

El cuaderno espera el archivo `datos_tratados.csv` en el mismo directorio. Si lo tienes en otra ruta, modifica la celda de carga:

```python
import pandas as pd

# Ruta por defecto
df = pd.read_csv('datos_tratados.csv')

# Si está en otra carpeta
# df = pd.read_csv('ruta/a/tu/datos_tratados.csv')
```

### Ejecutar el Cuaderno

Ejecutar cuaderno en Google Colab Challenge2_TelecomX_LATAM_.ipynb

---

## 🏆 Resultados del Modelo

| Modelo | Accuracy | Recall (Churn) | F1 (Churn) | Generalización |
|---|---|---|---|---|
| Random Forest (optimizado) | Alta | Moderado | Moderado | Overfitting notable |
| Decision Tree (optimizado) | Moderada | Moderado | Moderado | Overfitting notable |
| **Regresión Logística (optimizada)** | **Moderada** | **0.7888** | **0.6191** | **✅ Mejor generalización** |

> **Modelo seleccionado: Regresión Logística Optimizada** por su alto Recall y estabilidad entre train y test.

---

## 💡 Factores Clave de Churn

1. 🔻 **Baja antigüedad** → mayor riesgo
2. 📶 **Fibra óptica** → mayor churn que DSL o sin internet
3. 💸 **Cargos mensuales altos** → mayor probabilidad de cancelar
4. 📅 **Contrato mes a mes** → el tipo de mayor riesgo
5. 💳 **Pago con cheque electrónico** y **facturación sin papel** → asociados a mayor churn
6. 🔒 **Sin seguridad en línea / sin soporte técnico** → aumentan el riesgo

---

## 💻 Tecnologías Utilizadas

- **Python 3.8+**
- `pandas` — manipulación y análisis de datos
- `scikit-learn` — modelado, preprocesamiento y evaluación
- `imbalanced-learn` — manejo de desbalance con SMOTE
- `matplotlib` / `seaborn` — visualización
