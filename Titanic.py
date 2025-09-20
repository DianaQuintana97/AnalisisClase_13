import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

st.write(''' # Predicción de sobrevivientes del Titanic ''')
st.image("billete-titanic.jpg", caption="El Titanic navegaba desde Southampton, Inglaterra, hasta Nueva York en Estados Unidos.")

st.sidebar.header('Datos de evaluación')

def user_input_features():
  Pclass = st.sidebar.slider('Clase', 1, 3)
  Sex = st.sidebar.slider('Género', 0, 1)
  Age = st.sidebar.slider('Edad', 0, 100)
  SibSp = st.sidebar.slider('Hermanos(as)/Esposo(a)', 0, 10)
  Parch = st.sidebar.slider('Padres/Hijos', 0, 10)
  Fare = st.sidebar.slider('Tarifa', 0, 512)
  Embarked = st.sidebar.slider('Lugar de Embarque', 0, 3)

  user_input_data = {'Pclass': Pclass,
                     'Sex': Sex,
                     'Age': Age,
                     'SibSp': SibSp,
                     'Parch': Parch,
                     'Fare': Fare,
                     'Embarked': Embarked}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

st.subheader('Datos de evaluación')
st.write(
    pd.DataFrame(
        {
            "Atributos": ['Clase', 'Género', 'Edad', 'Hermanos(as)/Esposo(a)',
                          'Padres/Hijos', 'Tarifa', 'Lugar de Embarque'],
            "Valores": [df.iloc[0,0], df.iloc[0,1], df.iloc[0,2], df.iloc[0,3],
                        df.iloc[0,4], df.iloc[0,5], df.iloc[0,6]]
        }))

titanic =  pd.read_csv('Titanic2.csv', encoding='latin-1')
X = titanic.drop(columns='Survived')
Y = titanic['Survived']

classifier = DecisionTreeClassifier(max_depth=5, criterion='gini', min_samples_leaf=20, max_features=5, random_state=0)
classifier.fit(X, Y)

prediction = classifier.predict(df)
prediction_probabilities = classifier.predict_proba(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No sobrevive')
elif prediction == 1:
  st.write('Sobrevive')
else:
  st.write('Sin predicción')

st.subheader('Probabilidad de predicción')
st.write(prediction_probabilities)
