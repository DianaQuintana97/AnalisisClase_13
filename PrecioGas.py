import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción del Precio de la Gasolina con Streamlit ''')

# Use columns to center the image
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("bomba-de-gasolina.png", caption="Precio de gasolina regular.")


st.header('Factores evaluados')

def user_input_features():
  # Entrada
  año = st.number_input('año (2017 - 2030 ):',  min_value=2017, max_value=2030, value = 2030, step = 1)
  estados = [
    "Aguascalientes",
    "Baja California",
    "Baja California Sur",
    "Campeche",
    "Chiapas",
    "Chihuahua",
    "Ciudad de México",
    "Coahuila de Zaragoza",
    "Colima",
    "Durango",
    "Guanajuato",
    "Guerrero",
    "Hidalgo",
    "Jalisco",
    "Michoacán de Ocampo",
    "Morelos",
    "México",
    "Nayarit",
    "Nuevo León",
    "Oaxaca",
    "Puebla",
    "Querétaro",
    "Quintana Roo",
    "San Luis Potosí",
    "Sinaloa",
    "Sonora",
    "Tabasco",
    "Tamaulipas",
    "Tlaxcala",
    "Veracruz de Ignacio de la Llave",
    "Yucatán",
    "Zacatecas"
]
  entidad_name = st.selectbox("Selecciona un estado", estados)
  meses = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]
  mes_seleccionado = st.selectbox("Selecciona el mes:", meses)
  mes_num = meses.index(mes_seleccionado) + 1

  #mes_num = st.number_input('Mes (1 - 12 donde enero =1 ... diciembre=12):', min_value=1, max_value=12, value = 1, step = 1)

  # Create a mapping from state names to numerical values
  estado_mapping = {state: i for i, state in enumerate(estados)}
  entidad_num = estado_mapping[entidad_name]


  user_input_data = {'año': año,
                     'mes_num': mes_num,
                     'entidad_num': entidad_num}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
#año|mes_num|precio|entidad_num|
precios =  pd.read_csv('PreciosGas.csv', encoding='latin-1')
X = precios.drop(columns='precio')
y = precios['precio']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df.año + b1[1]*df.mes_num + b1[2]*df.entidad_num
st.subheader('Cálculo del precio de la gasolina regular')
st.write('Precio estimado ', prediccion)
