import pandas as pd
import os 
from minhas_classes import BinarizarDatasetTreino, BinarizarDatasetTreinoTeste, BinarizarInstancia



url = os.path.dirname(os.path.abspath(__file__)) + "\\datasets\\"



############### DADOS DO DATASET ############################
dataset_original_treino = url + "titanic_train.csv" 

dataset_binarizado_treino = url + "titanic_train_binary_accuracy.csv"
dataset_binarizado_teste = url + "titanic_test_binary_accuracy.csv"

coluna_target = "Survived"


#abrir dataset
df_treino = pd.read_csv(dataset_original_treino)



#REMOVER COLUNAS:
#Remover colunas caso exista
#colunas_a_remover = ['PassengerId','Ticket', 'Name', 'Cabin']
colunas_a_remover = ['Ticket', 'Name', 'Cabin']
df_treino = df_treino.drop(colunas_a_remover, axis=1)




#Binarizar datasets Treino e Teste:

#1- Criar o objeto BinarizarDataset
#objeto = BinarizarDatasetTreinoTeste(df_treino=df_treino, df_teste=df_teste, coluna_target=coluna_target)
objeto = BinarizarDatasetTreinoTeste(df_treino=df_treino,  coluna_target=coluna_target, proporcao_treino=0.7)



#2- Binarizar variáveis categóricas
objeto.binarizar_categorico('Pclass', prefix="Pclass")
objeto.binarizar_categorico('Sex', prefix='Sex')
objeto.binarizar_numerico('Age', num_bins=5)
objeto.binarizar_categorico('SibSp', prefix='SibSp')
objeto.binarizar_numerico('Fare', num_bins=6)
objeto.binarizar_categorico('Parch', prefix='Parch')
objeto.binarizar_categorico('Embarked', prefix='Embarked')

#Adicionar colunas faltantes ao dataset de teste:
colunas_faltantes = set(objeto.df_treino.columns) - set(objeto.df_teste.columns)
for col in colunas_faltantes:
    objeto.df_teste[col] = False
#Ordenar as colunas do dataset de teste para coincidir com o dataset de treino:
objeto.df_teste = objeto.df_teste[objeto.df_treino.columns]




#3- Tornar datasets consistentes
objeto.deixar_consistente()



#Salvar datasets binarizados
objeto.salvar_datasets(dataset_binarizado_treino, dataset_binarizado_teste)


