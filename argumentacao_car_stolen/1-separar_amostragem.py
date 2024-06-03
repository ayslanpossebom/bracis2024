#Geração da lista de argumentos

import pandas as pd
from itertools import combinations
from collections import Counter
import os

url = os.path.dirname(os.path.abspath(__file__)) + "\\"



#CONFIGURAÇÃO
dataset_original_treino = url + "car_stolen.csv" 
dataset_original_teste = url + "car_stolen_test.csv" 

dataset_binarizado_treino = url + "car_stolen_binary.csv"
dataset_binarizado_teste = url + "car_stolen_test_binary.csv"

dataset_binarizado_amostras = url + "car_stolen_binary_amostras.csv"

coluna_target = "Stolen"



total_amostras_por_classe = 0   #0 para todo o dataset,  10 para 10 itens em cada classe


#Abrir dataset original
df = pd.read_csv(dataset_binarizado_treino)
resultados_possiveis = df[coluna_target].unique()






print("Dataset original tem: ",len(df))

#Criar dataset com o total de amostras desejado
if(total_amostras_por_classe != 0):
    # Embaralhando o DataFrame
    df_original = df.sample(frac=1).reset_index(drop=True)
    # Depois de embaralhar, selecione as amostras para cada classe possivel
    df = df_original.groupby(coluna_target).head(total_amostras_por_classe)



print("Dataset amostras tem: ", len(df))

df.to_csv(dataset_binarizado_amostras, index=False)




