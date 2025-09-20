import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# ==============================
# 🚀 1. Configuración de Spark
# ==============================
spark = SparkSession.builder.appName("RegresionLineal").getOrCreate()

st.title("📊 Regresión Lineal Múltiple con PySpark")
st.write("Entrenamiento, evaluación y predicción interactiva")

# ==============================
# 📂 2. Cargar datos
# ==============================
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = spark.read.csv(uploaded_file, header=True, inferSchema=True)
    st.write("### Vista previa de los datos:")
    st.dataframe(df.limit(10).toPandas())

    # ==============================
    # ⚡ 3. Entrenamiento del modelo
    # ==============================
    atributos = [c for c in df.columns if c != 'precio']
    assembler = VectorAssembler(inputCols=atributos, outputCol="Valores")
    df_rl = assembler.transform(df).select('Valores', 'precio')

    # División en train y test
    train_data, test_data = df_rl.randomSplit([0.7, 0.3], seed=0)

    lr = LinearRegression(featuresCol='Valores', labelCol='precio')
    lr_model = lr.fit(train_data)

    # ==============================
    # 📊 4. Evaluación del modelo
    # ==============================
    predictions = lr_model.transform(test_data)
    evaluator_rmse = RegressionEvaluator(labelCol="precio", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="precio", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    st.subheader("📈 Métricas del modelo")
    st.write(f"**Error Cuadrático Medio (RMSE):** {rmse:.2f}")
    st.write(f"**Coeficiente de Determinación (R²):** {r2:.2f}")
    st.write(f"**Coeficientes:** {lr_model.coefficients}")
    st.write(f"**Intersección (Intercept):** {lr_model.intercept}")

    # ==============================
    # 🔮 5. Predicción interactiva
    # ==============================
    st.subheader("🔮 Predicción")
    st.write("Introduce los valores de las variables para predecir el precio:")

    # Entradas dinámicas según las columnas
    valores_usuario = []
    for i, atributo in enumerate(atributos):
        valor = st.number_input(f"{atributo}", value=0.0)
        valores_usuario.append(valor)

    if st.button("Predecir"):
        # Calcular manualmente: Σ(coef[i]*x[i]) + intercept
        prediccion = sum([valores_usuario[i] * lr_model.coefficients[i] for i in range(len(valores_usuario))]) + lr_model.intercept
        st.success(f"✅ Predicción de precio: **{prediccion:.2f}**")
else:
    st.info("Sube un archivo CSV para comenzar.")
