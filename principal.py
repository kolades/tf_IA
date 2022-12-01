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
#Age = st.number_input('Digite a idade do paciente:', min_value=1,max_value=150)
Age = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
Sex_ = st.selectbox("Informe seu genero:",("Feminino","Masculino"))
if Sex_ == "Feminino":
            Sex = 1
else:
            Sex = 0
#Sex = form.selectbox(
 #   "Enter the Gender:",
  #  ["Female", "Male"],
   # index=0,
#)
ChestPainType_= st.selectbox('Informe o tipo de dor no peito [TA: Angina Típica, ATA: Angina Atípica, NAP: Dor Não Anginosa, ASY: Assintomática] :',("ATA","NAP","ASY","TA"))
if ChestPainType_== "ATA
            ChestPainType = 0
elif ChestPainType_ == "NAP"
            ChestPainType = 1
elif ChestPainType_ == "ASY
            ChestPainType = 2
            else:
            ChestPainType = 3
RestingBP = st.slider(
            "# of results",
            min_value=120,
            max_value=180,
            value=120,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
Cholesterol = st.number_input('Informe o colesterol[mm/dl]:')
FastingBS = st.selectbox('Informe o açúcar no sangue em jejum [1: se JejumBS > 120 mg/dl, 0: caso contrário',("0","1"))
RestingECG = st.selectbox('Informe o resultado do eletrocardiograma em repouso:',("LVH","NORMAL","ST"))
#MaxHR = st.number_input('Informe o valor da Frequencia Cardiáca:',min_value=40,max_value=202)#  [Valor numérico entre 60 e 202]
MaxHR = st.slider(
            "# of results",
            min_value=60,
            max_value=202,
            value=60,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
ExerciseAngina = st.selectbox('Informe se a Angina foi induzida por exercício:',("SIM","NÃO"))
Oldpeak = st.number_input('Digite o valor da Oldpeak:')
ST_Slope = st.selectbox('Informe a inclinação do segmento ST do exercício de pico :',("DOWN","FLAT","UP"))

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
