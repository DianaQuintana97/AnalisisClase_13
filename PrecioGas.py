import numpy as np
import streamlit as st
import pandas as pd
import streamlit as st


st.write(''' # Predicción del Precio de la Gasolina con Streamlit ''')
st.image("gas.jpeg", caption="Precio de gasolina.")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  año = st.number_input('año (2017 - 2025 ):',  min_value=0, max_value=1, value = 0, step = 1)
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
  entidad_num = st.selectbox("Selecciona un estado", estados)
  mes_num = st.number_input('Mes (1 - 12 donde enero =1 ... diciembre=12):', min_value=0, max_value=230, value = 0, step = 1)
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
st.subheader('Cálculo de precio')
st.write('Precio estimado ', prediccion)
