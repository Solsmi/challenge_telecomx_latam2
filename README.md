# Predicción de Churn en TelecomX

Este proyecto de análisis de datos y machine learning se centra en desarrollar modelos predictivos para identificar clientes de TelecomX con alta probabilidad de cancelar sus servicios (`Churn`), con el fin de implementar estrategias de retención proactivas.

## 🚀 Visión General del Proyecto
Este proyecto tiene como finalidad desarrollar y evaluar modelos predictivos avanzados capaces de prever qué clientes de TelecomX tienen una mayor probabilidad de cancelar sus servicios (`Churn`). La retención de clientes es un pilar fundamental para el éxito y la sostenibilidad de cualquier empresa de telecomunicaciones. Al identificar proactivamente a los clientes en riesgo, TelecomX puede implementar estrategias de retención dirigidas y personalizadas, optimizando recursos y minimizando pérdidas de ingresos.

## 🎯 Objetivo
El objetivo principal es construir un modelo de clasificación robusto que maximice la detección de clientes que realmente abandonarán el servicio (alto `Recall`), permitiendo así una intervención temprana y efectiva por parte de la empresa.

## 📊 Conjunto de Datos
El análisis se basa en un conjunto de datos que contiene información detallada sobre los clientes de TelecomX, incluyendo:
*   **Demografía**: Género, si son ciudadanos de la tercera edad, si tienen pareja o dependientes.
*   **Servicios**: Servicio telefónico, múltiples líneas, servicio de internet (DSL, fibra óptica), seguridad en línea, respaldo en línea, protección de dispositivos, soporte técnico, streaming de TV y películas.
*   **Facturación y Contrato**: Duración del contrato (`Tenure`), tipo de contrato, facturación sin papel, método de pago, cargos mensuales (`ChargesMonthly`) y cargos totales (`ChargesTotal`).
*   **Variable Objetivo**: `Churn` (Sí/No).

## 🛠️ Metodología

### 1. Extracción y Preprocesamiento
*   **Carga de Datos**: El conjunto de datos `datos_tratados.csv` se cargó en un DataFrame de Pandas.
*   **Verificación de Calidad**: Se realizaron verificaciones exhaustivas de valores nulos, espacios vacíos o en blanco en todas las columnas.
*   **Eliminación de Columnas Irrelevantes**: Se eliminó `CustomerID` por ser un identificador único sin valor predictivo y `ChargesDaily` y `ChargesTotal` para evitar multicolinealidad con `ChargesMonthly` y `Tenure`.
*   **Análisis de Importancia Categórica (Chi-cuadrado)**: Se utilizó el test Chi-cuadrado para evaluar la relación entre las variables categóricas y `Churn`. Se identificó que `Gender` y `PhoneService` eran irrelevantes y fueron eliminadas.
*   **Tratamiento de Outliers**: Se aplicó un método de capping basado en el rango intercuartílico (IQR) a las columnas numéricas `Tenure` y `ChargesMonthly` para manejar valores atípicos.
*   **One-Hot Encoding**: Las variables categóricas restantes se convirtieron a formato numérico utilizando One-Hot Encoding con `pd.get_dummies`.

### 2. Balanceo de Clases
La variable `Churn` presentaba un desbalance significativo. Para abordarlo:
*   **SMOTE**: Se aplicó SMOTE (Synthetic Minority Over-sampling Technique) en el conjunto de entrenamiento para los modelos `RandomForestClassifier` y `DecisionTreeClassifier`.
*   **`class_weight='balanced'`**: Para `LogisticRegression`, se utilizó el parámetro `class_weight='balanced'` para ajustar el peso de las clases durante el entrenamiento sin generar datos sintéticos.

### 3. Modelado Predictivo y Evaluación Inicial
Se entrenaron y evaluaron tres modelos principales:
*   **Random Forest Classifier**
*   **Decision Tree Classifier**
*   **Regresión Logística**

La evaluación se realizó utilizando métricas clave como `Accuracy`, `Precision`, `Recall`, `F1-Score` y matrices de confusión, prestando especial atención al rendimiento para la clase minoritaria (`Churn`). Se observó un fuerte overfitting en los modelos basados en árboles en su versión inicial.

### 4. Optimización de Modelos
Se empleó `GridSearchCV` para la optimización de hiperparámetros en `RandomForestClassifier` y `LogisticRegression` para mejorar su capacidad de generalización y reducir el overfitting.

## 🏆 Resultados Clave y Conclusiones

*   **Modelo Preferido**: La **Regresión Logística Optimizada** fue seleccionada como el modelo más adecuado. Aunque su `Accuracy` y `Precision` global no fueron los más altos, demostró el **mejor `Recall` para la clase 'Churn' (0.7888)** y un `F1-Score` competitivo (0.6191). Esto es crucial para la retención de clientes, ya que minimiza los falsos negativos (clientes que abandonan y no son identificados).
*   **Overfitting**: Los modelos `Random Forest` y `Decision Tree` mostraron un overfitting significativo entre los conjuntos de entrenamiento y prueba, a pesar de la optimización, lo que los hace menos fiables para datos no vistos.
*   **Generalización**: La Regresión Logística mostró una brecha menor de rendimiento entre el entrenamiento y la prueba, indicando una mejor capacidad de generalización.

## 🔑 Factores Principales de Churn
El análisis de los coeficientes del modelo de Regresión Logística optimizado reveló los siguientes factores clave en la predicción del `Churn`:
1.  **`Tenure` (Antigüedad del Cliente)**: El factor más protector; a mayor antigüedad, menor `Churn`.
2.  **`InternetService_No` vs `InternetService_Fiber optic`**: No tener servicio de internet reduce el `Churn`, mientras que la fibra óptica lo aumenta.
3.  **`ChargesMonthly` (Cargos Mensuales)**: Cargos más altos se asocian con mayor `Churn`.
4.  **`Contract_Month-to-month` vs `Contract_Two year`**: Contratos mes a mes aumentan el `Churn`, contratos de dos años lo reducen.
5.  **`PaymentMethod_Electronic check` y `PaperlessBilling_Yes`**: Estos métodos de pago/facturación se asocian con mayor `Churn`.
6.  **`OnlineSecurity_Yes` y `TechSupport_No`**: No tener soporte técnico aumenta el `Churn`; tener seguridad en línea lo reduce.

## 💡 Estrategias de Retención Propuestas
Basado en los factores identificados, se proponen las siguientes estrategias:
*   **Fidelización por Antigüedad**: Programas de recompensas y beneficios para clientes leales.
*   **Gestión del Servicio de Internet**: Investigar problemas en fibra óptica; ofrecer paquetes de internet a clientes solo de telefonía.
*   **Optimización de Precios**: Planes flexibles y comunicación clara del valor para clientes con cargos altos.
*   **Incentivos para Contratos Largos**: Ofertas atractivas para migrar a contratos de uno o dos años.
*   **Monitoreo de Clientes Específicos**: Atención especial a clientes con pago electrónico y facturación sin papel.
*   **Promoción de Servicios de Soporte/Seguridad**: Destacar el valor y ofrecer incentivos para la adopción de estos servicios.

## 💻 Tecnologías Utilizadas
*   Python
*   Pandas (manipulación de datos)
*   Scikit-learn (modelado y preprocesamiento)
*   Matplotlib y Seaborn (visualización de datos)
*   Imblearn (manejo de desbalance de clases con SMOTE)
