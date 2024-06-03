import pandas as pd
import os 
from minhas_classes import BinarizarDatasetTreino, BinarizarDatasetTreinoTeste, BinarizarInstancia



url = os.path.dirname(os.path.abspath(__file__)) + "\\datasets\\"



############### DADOS DO DATASET ############################
dataset_original_treino = url + "PlayTennis.csv" 
dataset_original_teste = url + "PlayTennis_test.csv" 

dataset_binarizado_treino = url + "PlayTennis_binary.csv"
dataset_binarizado_teste = url + "PlayTennis_test_binary.csv"

coluna_target = "Play Tennis"


#abrir dataset
df_treino = pd.read_csv(dataset_original_treino)
df_teste = pd.read_csv(dataset_original_teste)

"""
#Binarizar datasets Treino e Teste:

#1- Criar o objeto BinarizarDataset
objeto = BinarizarDatasetTreinoTeste(df_treino=df_treino, df_teste=df_teste, coluna_target=coluna_target)


#2- Binarizar variáveis categóricas
objeto.binarizar_categorico('Outlook')
objeto.binarizar_categorico('Temperature')
objeto.binarizar_categorico('Humidity')
objeto.binarizar_categorico('Wind')


#3- Tornar datasets consistentes
objeto.deixar_consistente()



#Salvar datasets binarizados
objeto.salvar_datasets(dataset_binarizado_treino, dataset_binarizado_teste)
"""



"""
#Binarizar datasets Treino e Teste:
#1- Criar o objeto BinarizarDataset
objeto = BinarizarDatasetTreino(df_treino=df_treino, coluna_target=coluna_target)
#2- Binarizar variáveis categóricas
objeto.binarizar_categorico('Outlook')
objeto.binarizar_categorico('Temperature')
objeto.binarizar_categorico('Humidity')
objeto.binarizar_categorico('Wind')
#3- Tornar datasets consistentes
objeto.deixar_consistente()
#Salvar datasets binarizados
objeto.salvar_datasets(dataset_binarizado_treino)

#1- Criar o objeto BinarizarDataset
objeto = BinarizarDatasetTreino(df_treino=df_teste, coluna_target=coluna_target)
#2- Binarizar variáveis categóricas
objeto.binarizar_categorico('Outlook')
objeto.binarizar_categorico('Temperature')
objeto.binarizar_categorico('Humidity')
objeto.binarizar_categorico('Wind')
#3- Tornar datasets consistentes
objeto.deixar_consistente()
#Salvar datasets binarizados
objeto.salvar_datasets(dataset_binarizado_teste)
"""




#Teste único
instancia = pd.DataFrame(
    {
        'Outlook': ['Sunny'], 
        'Temperature': ['Hot'], 
        'Humidity': ['High'], 
        'Wind': ['Weak'],
        'Play Tennis': ['No']
    }
)

instancia = [
    {
        'Outlook': 'Sunny', 
        'Temperature': 'Hot', 
        'Humidity': 'High', 
        'Wind': 'Weak',
        'Play Tennis': 'No'
    }
]
print(instancia)
objeto_instancia = BinarizarInstancia(instancia=instancia, coluna_target=coluna_target)
#print(objeto_instancia.df_instancia)
#print(objeto_instancia.X)
#print(objeto_instancia.y)

objeto_instancia.binarizar_categorico('Outlook', possiveis_valores=df_treino['Outlook'].unique())
objeto_instancia.binarizar_categorico('Temperature', possiveis_valores=df_treino['Temperature'].unique())
objeto_instancia.binarizar_categorico('Humidity', possiveis_valores=df_treino['Humidity'].unique())
objeto_instancia.binarizar_categorico('Wind', possiveis_valores=df_treino['Wind'].unique())

print(objeto_instancia.df_instancia)


