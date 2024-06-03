import pandas as pd
from itertools import combinations
from collections import Counter
import os

url = os.path.dirname(os.path.abspath(__file__)) + "\\"


#CONFIGURAÇÃO
#argumentos_heart_disease = 'playtennis_argumentos.ob'
#dataset_binarizado = "PlayTennis_binarizado.csv"
#coluna_target = "Play Tennis"
#quantia_para_teste = 14

dataset_original_treino = url + "car_stolen.csv" 
dataset_original_teste = url + "car_stolen_test.csv" 

dataset_binarizado_treino = url + "car_stolen_binary.csv"
dataset_binarizado_teste = url + "car_stolen_test_binary.csv"

dataset_binarizado_amostras = url + "car_stolen_binary_amostras.csv"

coluna_target = "Stolen"

argumentos_car_stolen = 'car_stolen_argumentos.ob'

quantia_para_teste = 0   #quantia para testes, 0 para todos








#######################################################################
class ArgumentosJustificaveis:
    def __init__(self, df_temp, resultados_possiveis, argumentos):

        self.quantia_classificada_correto = 0
        self.quantia_classificada_incorreto = 0
        
        #opcoes_disponiveis = df[coluna_target].unique()
        opcoes_disponiveis = resultados_possiveis
        self.quantidades_opcoes_disponiveis = []

        self.maiores_chaves = []
        self.justificativas_global = []
        self.respostasCorretas = []

        for index, row in df_temp.iterrows():

            atributos_com_1 = [col for col in df_temp.columns if col != coluna_target and row[col] == 1]
            premissa = set(atributos_com_1)
            print(f"\nTESTANDO: <{premissa},{row[coluna_target]}>")
                
            argsjust = self.obterArgumentosJustificaveis(argumentos,premissa)
            argsjust_conclusoes = [arg["conclusao"] for arg in argsjust]
            resultado = {opcao: 0 for opcao in opcoes_disponiveis}
            resultado.update(Counter(argsjust_conclusoes))
            #print(resultado, "\n")
            
            self.quantidades_opcoes_disponiveis.append(resultado)

            maior_chave = max(resultado, key=resultado.get)
            
            maior_valor = max(resultado.values())
            print("Maior valor: ", maior_valor)

            maiores_chaves = [chave for chave, valor in resultado.items() if valor == maior_valor]
            print("Maiores chaves: ", maiores_chaves)
            print("Resultado: ", resultado)


            self.maiores_chaves.append(maiores_chaves)
            print("Resultado: ", maiores_chaves, " de ", resultado)





            justificativas_local = []

            for item in argsjust:
                #if(item["conclusao"] == maior_chave and item["premissas"] not in justificativas_local):
                if(item["conclusao"] in maiores_chaves and item["premissas"] not in justificativas_local):
                    #justificativas_local.append(item["premissas"])
                    justificativas_local.append(item)
            print("Justificativas: ")
            for item in justificativas_local:
                print("\t-", item)
            self.justificativas_global.append(justificativas_local)

            self.respostasCorretas.append(row[coluna_target])
            #if(maior_chave == row[coluna_target]):
            if(row[coluna_target] in maiores_chaves):
                self.quantia_classificada_correto += 1
            else:
                self.quantia_classificada_incorreto += 1
                #print("\t\tERRO: ", resultado)
                #input("Continuar...")



    def obterArgumentosJustificaveis(self, listaArgumentos,premissa): 
        argumentosJustificaveis = []
        totalPremissas = 0
        for argumento in listaArgumentos:
            #print("---ARGUMENTO: ", argumento)

            intersecao = premissa.intersection(argumento["premissas"])
            #print("----INTERSEÇÃO: ", intersecao)

            """
            #se a interseção for maior que o que se sabe, zerar a interseção
            if(len(intersecao) > totalPremissas):
                    argumentosJustificaveis = []
                    argumentosJustificaveis.append(argumento)
                    totalPremissas = len(intersecao)
                #senão, se for igual, adiciona o argumento
            elif(len(intersecao) == totalPremissas and intersecao not in argumentosJustificaveis):
                    argumentosJustificaveis.append(argumento)
            """


            incluir = True 
            for p in argumento["premissas"]:
                if p not in premissa:
                    incluir = False
                    break
            if(incluir):
                if(len(intersecao) > totalPremissas):
                        argumentosJustificaveis = []
                        argumentosJustificaveis.append(argumento)
                        totalPremissas = len(intersecao)
                    #senão, se for igual, adiciona o argumento
                elif(len(intersecao) == totalPremissas and intersecao not in argumentosJustificaveis):
                        argumentosJustificaveis.append(argumento)




        return argumentosJustificaveis

    def machineLearning(self, df, df_temp, coluna_target):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB

        X_train = df.drop([coluna_target], axis=1)
        y_train = df[[coluna_target]]

        X_test = df_temp.drop([coluna_target], axis=1)
        y_test = df_temp[[coluna_target]]

        # 3. Treinar os modelos
        # Árvore de Decisão
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_predictions = dt_model.predict(X_test)
        # K-Nearest Neighbors (KNN)
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        knn_predictions = knn_model.predict(X_test)
        # Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        nb_predictions = nb_model.predict(X_test)
        # 4. Avaliar e exibir os resultados

        #print("Árvore de Decisão    ", dt_predictions)
        #print("K-Nearest Neighbors  ", knn_predictions)
        #print("Naive Bayes          ", nb_predictions)

        fim = pd.DataFrame({
            'árvore': pd.Series(dt_predictions),
            'knn': pd.Series(knn_predictions),
            'bayes': pd.Series(nb_predictions)
        })
        return fim

    def obterArgumentosJustificaveisMachineLearning(self, listaArgumentos,premissa, conclusao): 
        argumentosJustificaveis = []
        totalPremissas = 0
        for argumento in listaArgumentos:
            if(argumento["conclusao"] == conclusao):
                intersecao = premissa.intersection(argumento["premissas"])

                """
                #se a interseção for maior que o que se sabe, zerar a interseção
                if(len(intersecao) > totalPremissas):
                        argumentosJustificaveis = []
                        argumentosJustificaveis.append(argumento)
                        totalPremissas = len(intersecao)
                    #senão, se for igual, adiciona o argumento
                elif(len(intersecao) == totalPremissas and intersecao not in argumentosJustificaveis):
                        argumentosJustificaveis.append(argumento)
                """


                incluir = True 
                for p in argumento["premissas"]:
                    if p not in premissa:
                        incluir = False
                        break
                if(incluir):
                    if(len(intersecao) > totalPremissas):
                            argumentosJustificaveis = []
                            argumentosJustificaveis.append(argumento)
                            totalPremissas = len(intersecao)
                        #senão, se for igual, adiciona o argumento
                    elif(len(intersecao) == totalPremissas and intersecao not in argumentosJustificaveis):
                            argumentosJustificaveis.append(argumento)


        return argumentosJustificaveis



#######################################################################


#Realizar testes

df = pd.read_csv(dataset_binarizado_teste)
resultados_possiveis = df[coluna_target].unique()


#Carregar lista de argumentos
import pickle
with open(url + argumentos_car_stolen, 'rb') as fp:
    argumentos = pickle.load(fp)

print("Total de argumentos válidos: ", len(argumentos))
for item in argumentos:
    print(item)

input("Pressione uma tecla para continuar")



#Fatos do usuário
if(quantia_para_teste == 0):
     quantia_para_teste = len(df)
df_temp = df.sample(quantia_para_teste)
#print(df_temp)




#Criar objeto para descoberta das justificativas
argumentosJustificaveis = ArgumentosJustificaveis(df_temp=df_temp, resultados_possiveis=resultados_possiveis, argumentos=argumentos)

respostas = argumentosJustificaveis.maiores_chaves
print("\n\nRESPOSTAS OBTIDAS: ", respostas)

respostasCorretas = argumentosJustificaveis.respostasCorretas

pd_fim = pd.DataFrame({
    'RESULTADOS': pd.Series(respostas),
    'CORRETO': pd.Series(respostasCorretas)
})

fim = pd.concat([pd_fim, argumentosJustificaveis.machineLearning(df=df, df_temp=df_temp, coluna_target=coluna_target)], axis=1)

print(fim)

print("De", len(df_temp), " instâncias:")
print("Quantia classificada corretamente: ", argumentosJustificaveis.quantia_classificada_correto)
print("Quantia classificada incorretamente: ", argumentosJustificaveis.quantia_classificada_incorreto)



cont = 0
for i, row in df_temp.iterrows():
    atributos_com_1 = [col for col in df_temp.columns if col != coluna_target and row[col] == 1]
    premissa = set(atributos_com_1)
    
    print(f"\Argumento: <{premissa},{row[coluna_target]}>")
    just = argumentosJustificaveis.justificativas_global[cont]
    print("Argumentos encontrados: ")
    for item in just:
        print("\t-", item)
    print()

    #ml = fim["árvore"].iloc[cont]
    ml = fim["knn"].iloc[cont]
    print("Machine Learning: ", ml)
    just = argumentosJustificaveis.obterArgumentosJustificaveisMachineLearning(argumentos, premissa, ml)
    for item in just:
        print("\t-", item)
    print()
    print()

    print()
    cont += 1

