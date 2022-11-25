import streamlit as st
import pandas as pd #manipulacao de dados

###################importando os dados do csv ########################
dados = pd.read_csv("heart.csv")
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
dados['Sex'] = encoder.fit_transform(pd.DataFrame(dados['Sex']))
dados['ChestPainType'] = encoder.fit_transform(pd.DataFrame(dados['ChestPainType']))
dados['RestingECG'] = encoder.fit_transform(pd.DataFrame(dados['RestingECG']))
dados['ExerciseAngina'] = encoder.fit_transform(pd.DataFrame(dados['ExerciseAngina']))
dados['ST_Slope'] = encoder.fit_transform(pd.DataFrame(dados['ST_Slope']))

#coleta os nomes das colunas (DEATH_EVENT)
nomes_colunas = dados.columns.to_list()
tamanho = len(nomes_colunas)
nomes_colunas = nomes_colunas[:tamanho-1]
features = dados[nomes_colunas]
classes = dados['HeartDisease']
features.shape,classes.shape

#dividir os dados entre treino e teste
from sklearn.model_selection import train_test_split

features_treino,features_teste,classes_treino,classes_teste = train_test_split(features,classes,test_size=0.3,random_state=2)

#Para usar recursos da árvore de decisão da biblioteca sklearn.tree
from sklearn.tree import DecisionTreeClassifier 

arvore = DecisionTreeClassifier()

#treinando a arvore
arvore.fit(features_treino,classes_treino)

#testando a arvore treinada
#resultado = arvore.predict(features_teste)

st.markdown("<h1 style='text-align: center; color: green;'>Aplicativo IA - Naives Bayes</h1>", unsafe_allow_html=True)
Age = st.number_input('Digite a idade do paciente:')
Sex = st.number_input('Digite o sexo:')
ChestPainType = st.number_input('Digite o tipo de dor:')
RestingBP = st.number_input('Digite o valor da pressão arterial:')
Cholesterol = st.number_input('Digite o valor da pressão arterial:')
FastingBS = st.number_input('Digite o valor da pressão arterial:')
RestingECG = st.number_input('Digite o valor da pressão arterial:')
MaxHR = st.number_input('Digite o valor da pressão arterial:')
ExerciseAngina = st.number_input('Digite o valor da pressão arterial:')
Oldpeak = st.number_input('Digite o valor da pressão arterial:')
ST_Slope = st.number_input('Digite o valor da pressão arterial:')

if st.button('Clique aqui'):
  resultado = arvore.predict([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])
  
  st.write('Resultado:',resultado)
 
 # if resultado == "Iris-versicolor":
  # st.write("Iris-versicolor")
   #st.image("Iris-versicolor.jpg")
  
  #if resultado == "Iris-setosa":
   #st.write("Iris-setosa")
   #st.image("Iris-setosa.jpg")
  
 # if resultado == "Iris-virginica":
  # st.write("Iris-virginica")
   #st.image("Iris-virginica.jpg")

