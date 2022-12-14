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


st.image("coracao-batendo.gif")
st.markdown("<h1 style='text-align: center; color: green;'>Previsão de Doença Cardiaca</h1>", unsafe_allow_html=True)
#Age = st.number_input('Digite a idade do paciente:', min_value=1,max_value=150)
Age = st.slider(
            "Informe a idade do paciente:",
            min_value=0,
            max_value=800,
            value=40,
            help="You can choose the number of keywords/keyphrases to display. Between 0 and 80, default number is 40.",
        )
Sex_ = st.selectbox("Informe o genero:",("Feminino","Masculino"))
if Sex_ == "Feminino":
            Sex = 0
else:
            Sex = 1
#Sex = form.selectbox(
 #   "Enter the Gender:",
  #  ["Female", "Male"],
   # index=0,
#)
ChestPainType_ = st.selectbox('Informe o tipo de dor no peito [TA: Angina Típica, ATA: Angina Atípica, NAP: Dor Não Anginosa, ASY: Assintomática] :',("ATA","NAP","ASY","TA"))
if ChestPainType_ == "ASY":
            ChestPainType = 0
elif ChestPainType_ == "ATA":
            ChestPainType = 1
elif ChestPainType_ == "NAP":
            ChestPainType = 2
else:
            ChestPainType = 3
RestingBP = st.slider(
            "Informe a pressão arterial em repouso [mm Hg]:",
            min_value=100,
            max_value=180,
            value=160,
            help="You can choose the number of keywords/keyphrases to display. Between 100 and 180, default number is 120.",
        )
#Cholesterol = st.number_input('Informe o colesterol[mm/dl]:')
Cholesterol = st.slider(
            "Informe o colesterol:",
            min_value=100,
            max_value=400,
            value=180,
            help="You can choose the number of keywords/keyphrases to display. Between 0 and 80, default number is 200.",
        )
FastingBS = st.selectbox("Informe o açúcar no sangue em jejum [1: se JejumBS > 120 mg/dl, 0: caso contrário",("0","1"))
RestingECG_ = st.selectbox('Informe o resultado do eletrocardiograma em repouso:',("LVH","NORMAL","ST"))
if RestingECG_ == "LVH":
            RestingECG = 0
elif RestingECG_ == "NORMAL":
            RestingECG = 1
else:
            RestingECG = 2
#MaxHR = st.number_input('Informe o valor da Frequencia Cardiáca:',min_value=40,max_value=202)#  [Valor numérico entre 60 e 202]
MaxHR = st.slider(
            "Informe o valor da Frequência Cardiáca:",
            min_value=40,
            max_value=200,
            value=172,
            help="You can choose the number of keywords/keyphrases to display. Between 40 and 200, default number is 172.",
        )
ExerciseAngina_ = st.selectbox('Informe se a Angina foi induzida por exercício:',("SIM","NÃO"))
if ExerciseAngina_ == "NÃO":
            ExerciseAngina = 0
else:
            ExerciseAngina = 1
Oldpeak = st.slider(
            "Digite o valor da Oldpeak:",
            min_value=-2,
            max_value=7,
            value=0,
            help="You can choose the number of keywords/keyphrases to display. Between -2 and 7, default number is 2.",
        )
ST_Slope_ = st.selectbox('Informe a inclinação do segmento ST do exercício de pico :',("UP","FLAT","DOWN"))
if ST_Slope_ == "FLAT":
            ST_Slope = 1
elif ST_Slope_ == "UP":
            ST_Slope = 2
else:
            ST_Slope = 2

if st.button('Clique aqui'):
  resultado = arvore.predict([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])
  
  st.write('Resultado:',resultado)
 
  if resultado == 0:
            st.write("Paciente Normal")
 
  if resultado == 1:
            st.write("Paciente Cardíaco")

