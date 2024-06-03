import pandas as pd
import os 



url = os.path.dirname(os.path.abspath(__file__)) + "\\"



############### DADOS DO DATASET ############################
dataset_original_treino = url + "car_stolen.csv" 
dataset_original_teste = url + "car_stolen_test.csv" 

dataset_binarizado_treino = url + "car_stolen_binary.csv"
dataset_binarizado_teste = url + "car_stolen_test_binary.csv"

coluna_target = "Stolen"




#abrir dataset
df_treino = pd.read_csv(dataset_original_treino)
df_teste = pd.read_csv(dataset_original_teste)



#Remover linhas com valore ausentes
df_treino = df_treino.dropna()
df_teste = df_teste.dropna()



#Separar X e y
X_treino = df_treino.drop(coluna_target, axis=1)
y_treino = df_treino[coluna_target]

X_teste = df_teste.drop(coluna_target, axis=1)
y_teste = df_teste[coluna_target]



#Para Brand #####################################
dummies_brand_treino = pd.get_dummies(df_treino['Brand'], drop_first=False)
dummies_brand_teste = pd.get_dummies(df_teste['Brand'], drop_first=False)



#Para Age #####################################
dummies_age_treino = pd.get_dummies(df_treino['Age'], prefix="Age_", drop_first=False)
dummies_age_teste = pd.get_dummies(df_teste['Age'], prefix="Age_", drop_first=False)




#Para Location #####################################
dummies_location_treino = pd.get_dummies(df_treino['Location'], drop_first=False)
dummies_location_teste = pd.get_dummies(df_teste['Location'], drop_first=False)


#Para Insurance #####################################
dummies_insurance_treino = pd.get_dummies(df_treino['Insurance'], drop_first=False)
dummies_insurance_teste = pd.get_dummies(df_teste['Insurance'], drop_first=False)




#Gerar os novos datasets binarizados
df_final_treino = pd.concat([dummies_brand_treino, dummies_age_treino, dummies_location_treino, dummies_insurance_treino, y_treino], axis=1)
df_final_teste = pd.concat([dummies_brand_teste, dummies_age_teste, dummies_location_teste, dummies_insurance_teste, y_teste], axis=1)




#Remover linhas duplicadas para DF ficar consistente
print("TREINO: Tamanho atual: ", len(df_final_treino))
df_final_treino = df_final_treino.drop_duplicates()
print("TREINO: Tamanho depois de remover linhas duplicadas: ", len(df_final_treino))

print("TESTE: Tamanho atual: ", len(df_final_teste))
df_final_teste = df_final_teste.drop_duplicates()
print("TESTE: Tamanho depois de remover linhas duplicadas: ", len(df_final_teste))


colunas = df_final_treino.drop(coluna_target, axis=1)
duplicadas = df_final_treino.duplicated(subset=colunas, keep=False)
df_duplicates = df_final_treino[duplicadas]
df_final_treino = df_final_treino.drop(df_duplicates.index)
print("TREINO: Tamanho depois de remover duplicadas inconsistentes: ", len(df_final_treino))

colunas = df_final_teste.drop(coluna_target, axis=1)
duplicadas = df_final_teste.duplicated(subset=colunas, keep=False)
df_duplicates = df_final_teste[duplicadas]
df_final_teste = df_final_teste.drop(df_duplicates.index)
print("TESTE: Tamanho depois de remover duplicadas inconsistentes: ", len(df_final_teste))



#Salvar datasets binarizados
df_final_treino.to_csv(dataset_binarizado_treino, index=False)
df_final_teste.to_csv(dataset_binarizado_teste, index=False)

print(df_final_treino)

print(df_final_teste)




