import pandas as pd

#dados = pd.read_csv("/content/drive/MyDrive/Inteligência Artificial/TrabalhoFinal/heart_failure_clinical_records_dataset.csv")
dados = pd.read_csv("/content/drive/MyDrive/Inteligência Artificial/TrabalhoFinal/heart.csv")
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
resultado = arvore.predict(features_teste)

#avalia se a predição da árvore é a esperada (ou correta)
from sklearn import metrics

dados

print(metrics.classification_report(classes_teste,resultado,target_names=['Sim','Não']))