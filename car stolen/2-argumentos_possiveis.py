import pandas as pd
import os 
from itertools import combinations
import pickle



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


############### DADOS DO DATASET ############################
dataset_original_treino = url + "car_stolen.csv" 
dataset_original_teste = url + "car_stolen_test.csv" 

dataset_binarizado_treino = url + "car_stolen_binary.csv"
dataset_binarizado_teste = url + "car_stolen_test_binary.csv"

coluna_target = "Stolen"

arquivo_argumentos_possiveis= url + "argumentos_possiveis.ob"


#Passo 1: abrir dataset treino e teste
df_treino = pd.read_csv(dataset_binarizado_treino)

df_teste = pd.read_csv(dataset_binarizado_teste)
instancias_teste = df_teste.sample(1)
instancias_teste = pd.DataFrame(
    {
        "Ford"  : [False],
        "Honda" : [False],
        "Toyota": [True],

        "Age__0" : [False],
        "Age__1": [True],
        "Age__2": [False],
        "Age__3": [False],
        "Age__4": [False],
        "Age__5": [False],
        "Age__6": [False],

        "Rural"     : [False],
        "Suburban"  : [False],
        "Urban"     : [True],

        "Basic"     : [False],
        "Advanced"  : [True],

        "Stolen"    : "Yes"
    }

)

print("Instância de teste: ")
print(instancias_teste)
print()



base_argumentos_possiveis = []
lista_premissas_argumentos_possiveis = []
id = 0


for index, linha in instancias_teste.iterrows():
    #print(linha.values)
    #print(linha.index)

    X_teste = linha.drop(coluna_target)
    y_teste = linha[coluna_target]
    
    atributos_com_1 = [col for col in df_treino.columns if col != coluna_target and linha[col] == 1]
    print("Linha: ", atributos_com_1)

    for i in range(1, len(atributos_com_1)+1):
        combinacoes = combinations(atributos_com_1, i)
        for combinacao in combinacoes:
            temp = set(combinacao)            
            
            if temp not in lista_premissas_argumentos_possiveis:
                lista_premissas_argumentos_possiveis.append(temp)
                base_argumentos_possiveis.append({"premissa": temp, "conclusao": y_teste, "id": id})
                id += 1


for item in base_argumentos_possiveis:
    print(f"{item['id']}<{item['premissa']}, {item['conclusao']}>")



#Salvar lista de argumentos possíveis:
with open(arquivo_argumentos_possiveis, 'wb') as fp:
    pickle.dump(base_argumentos_possiveis, fp)

