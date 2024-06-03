import pandas as pd
import os 
from minhas_classes import BinarizarDatasetTreino, BinarizarDatasetTreinoTeste, BinarizarInstancia



url = os.path.dirname(os.path.abspath(__file__)) + "\\datasets\\"



############### DADOS DO DATASET ############################
dataset_original_treino = url + "seeds.csv" 

dataset_binarizado_treino = url + "seeds_train_binary_accuracy.csv"
dataset_binarizado_teste = url + "seeds_test_binary_accuracy.csv"

coluna_target = "Type"


#abrir dataset
df_treino = pd.read_csv(dataset_original_treino)



#REMOVER COLUNAS:
#Remover colunas caso exista
#colunas_a_remover = ['PassengerId','Ticket', 'Name', 'Cabin']





#Binarizar datasets Treino e Teste:

#1- Criar o objeto BinarizarDataset
#objeto = BinarizarDatasetTreinoTeste(df_treino=df_treino, df_teste=df_teste, coluna_target=coluna_target)
objeto = BinarizarDatasetTreinoTeste(df_treino=df_treino,  coluna_target=coluna_target, proporcao_treino=0.7)



#2- Binarizar variáveis categóricas
objeto.binarizar_numerico('Area', num_bins=5)
objeto.binarizar_numerico('Perimeter', num_bins=5)
objeto.binarizar_numerico('Compactness', num_bins=5)
objeto.binarizar_numerico('Kernel.Length', num_bins=5)
objeto.binarizar_numerico('Kernel.Width', num_bins=5)
objeto.binarizar_numerico('Asymmetry.Coeff', num_bins=5)
objeto.binarizar_numerico('Kernel.Groove', num_bins=5)

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


