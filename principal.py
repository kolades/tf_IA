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

st.markdown("<h1 style='text-align: center; color: green;'>Aplicativo IA - Previsão de Doença Cardiaca</h1>", unsafe_allow_html=True)
Age = st.date_input('Digite a idade do paciente:', min_value=1,max_value=150)
#Age =  st.date_input('Digite a idade do paciente:', datetime.date(2019, 7, 6))
#Age = st.date_input("Qual a data de seu anivesário", datetime.date(2019, 7, 6))
Sex = st.number_input('Digite o sexo:')
ChestPainType = st.number_input('Digite o tipo de dor:')
RestingBP = st.number_input('Digite o valor da pressão arterial:', min_value=120,max_value=180)
Cholesterol = st.number_input('Digite o colesterol:')
FastingBS = st.number_input('Digite o valor da FastingBS:')
RestingECG = st.number_input('Digite o valor da RestingECG:')
MaxHR = st.number_input('Digite o valor da MaxHR:')
ExerciseAngina = st.number_input('Digite o valor da ExerciseAngina:')
Oldpeak = st.number_input('Digite o valor da Oldpeak:')
ST_Slope = st.number_input('Digite o valor da ST_Slope:')

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
