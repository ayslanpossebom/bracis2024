import pandas as pd
import os 
import numpy as np
from itertools import combinations



url = os.path.dirname(os.path.abspath(__file__)) + "\\datasets\\"




class BinarizarDatasetTreinoTeste:
    def __init__(self, df_treino, coluna_target, df_teste="", proporcao_treino=0.7):
        if(len(df_teste) == 0):
            self.df_treino = df_treino.sample(frac=proporcao_treino)
            self.df_teste = df_treino.drop(self.df_treino.index)
        else:
            self.df_treino = df_treino
            self.df_teste = df_teste
        self.coluna_target = coluna_target
        self.X_treino = self.df_treino.drop(coluna_target, axis=1)
        self.y_treino = self.df_treino[coluna_target]
        self.X_teste = self.df_teste.drop(coluna_target, axis=1)
        self.y_teste = self.df_teste[coluna_target]

    def binarizar_categorico(self, nome_coluna, prefix=""):
        if prefix:
            coluna_treino = pd.get_dummies(self.df_treino[nome_coluna], drop_first=False, prefix=prefix)
            coluna_teste = pd.get_dummies(self.df_teste[nome_coluna], drop_first=False, prefix=prefix)       
        else:
            coluna_treino = pd.get_dummies(self.df_treino[nome_coluna], drop_first=False)
            coluna_teste = pd.get_dummies(self.df_teste[nome_coluna], drop_first=False)
        
        self.df_treino = pd.concat([self.df_treino, coluna_treino], axis=1)
        self.df_teste = pd.concat([self.df_teste, coluna_teste], axis=1)

        self.df_treino = self.df_treino.drop(nome_coluna, axis=1)
        self.df_teste = self.df_teste.drop(nome_coluna, axis=1)

        self.X_treino = self.df_treino.drop(self.coluna_target, axis=1)
        self.y_treino = self.df_treino[self.coluna_target]
        self.X_teste = self.df_teste.drop(self.coluna_target, axis=1)
        self.y_teste = self.df_teste[self.coluna_target]

    def binarizar_numerico(self, nome_coluna, num_bins):
        min_value = self.df_treino[nome_coluna].min()
        max_value = self.df_treino[nome_coluna].max()
        prefixo = nome_coluna
        bin_width = (max_value - min_value) / num_bins
        bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
        bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]

        
        self.df_treino[nome_coluna+"_label"] = pd.cut(self.df_treino[nome_coluna], bins=bin_edges, labels=bins_labels)
        dummies_treino = pd.get_dummies(self.df_treino[nome_coluna+"_label"], drop_first=False)


        self.df_teste[nome_coluna + "_label"] = pd.cut(self.df_teste[nome_coluna], bins=bin_edges, labels=bins_labels)
        dummies_teste = pd.get_dummies(self.df_teste[nome_coluna + "_label"], drop_first=False)
    
        """
        min_value = self.df_teste[nome_coluna].min()
        max_value = self.df_teste[nome_coluna].max()
        prefixo = nome_coluna
        bin_width = (max_value - min_value) / num_bins
        bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
        bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
        self.df_teste[nome_coluna+"_label"] = pd.cut(self.df_teste[nome_coluna], bins=bin_edges, labels=bins_labels)
        dummies_teste = pd.get_dummies(self.df_teste[nome_coluna+"_label"], drop_first=False)
        """


        self.df_treino = pd.concat([self.df_treino, dummies_treino], axis=1)
        self.df_teste = pd.concat([self.df_teste, dummies_teste], axis=1)

        self.df_treino = self.df_treino.drop(nome_coluna, axis=1)
        self.df_teste = self.df_teste.drop(nome_coluna, axis=1)

        self.df_treino = self.df_treino.drop(nome_coluna+"_label", axis=1)
        self.df_teste = self.df_teste.drop(nome_coluna+"_label", axis=1)


        self.X_treino = self.df_treino.drop(self.coluna_target, axis=1)
        self.y_treino = self.df_treino[self.coluna_target]
        self.X_teste = self.df_teste.drop(self.coluna_target, axis=1)
        self.y_teste = self.df_teste[self.coluna_target]


    def binarizar_categorico_individual(self, dataset, nome_coluna, prefix=""):
        if prefix:
            coluna = pd.get_dummies(dataset[nome_coluna], drop_first=False, prefix=prefix)
        else:
            coluna = pd.get_dummies(dataset[nome_coluna], drop_first=False, prefix=prefix)        
        dataset = pd.concat([dataset, coluna], axis=1)
        dataset = dataset.drop(nome_coluna, axis=1)
        return dataset
    
    def binarizar_numerico_individual(self, dataset, nome_coluna, num_bins):
        min_value = dataset[nome_coluna].min()
        max_value = dataset[nome_coluna].max()
        prefixo = nome_coluna
        bin_width = (max_value - min_value) / num_bins
        bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
        bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
        dataset[nome_coluna+"_label"] = pd.cut(dataset[nome_coluna], bins=bin_edges, labels=bins_labels)
        dummies = pd.get_dummies(dataset[nome_coluna+"_label"], drop_first=False)        

        dataset = pd.concat([dataset, dummies], axis=1)
        return dataset
    
    def deixar_consistente(self):
        self.df_treino  = self.df_treino.dropna()
        self.df_teste = self.df_teste.dropna()

        self.df_treino = self.df_treino.drop_duplicates()
        self.df_teste = self.df_teste.drop_duplicates()

        colunas = self.df_treino.drop(self.coluna_target, axis=1)
        duplicadas = self.df_treino.duplicated(subset=colunas, keep=False)
        df_duplicates = self.df_treino[duplicadas]
        self.df_treino = self.df_treino.drop(df_duplicates.index)

        colunas = self.df_teste.drop(self.coluna_target, axis=1)
        duplicadas = self.df_teste.duplicated(subset=colunas, keep=False)
        df_duplicates = self.df_teste[duplicadas]
        self.df_teste = self.df_teste.drop(df_duplicates.index)

        self.X_treino = self.df_treino.drop(self.coluna_target, axis=1)
        self.y_treino = self.df_treino[self.coluna_target]
        self.X_teste = self.df_teste.drop(self.coluna_target, axis=1)
        self.y_teste = self.df_teste[self.coluna_target]

        #Adicionar colunas faltantes ao dataset de teste:
        colunas_faltantes = set(self.df_treino.columns) - set(self.df_teste.columns)
        for col in colunas_faltantes:
            self.df_teste[col] = False
        #Ordenar as colunas do dataset de teste para coincidir com o dataset de treino:
        self.df_teste = self.df_teste[self.df_treino.columns]

    
    def salvar_datasets(self, nome_arquivo_treino, nome_arquivo_teste):
        self.df_treino.to_csv(nome_arquivo_treino, index=False)
        self.df_teste.to_csv(nome_arquivo_teste, index=False)


#########################################################################
class BinarizarDatasetTreino:
    def __init__(self, df_treino, coluna_target):
        self.df_treino = df_treino

        self.coluna_target = coluna_target
        self.X_treino = self.df_treino.drop(coluna_target, axis=1)
        self.y_treino = self.df_treino[coluna_target]


    def binarizar_categorico(self, nome_coluna, prefix=""):
        if prefix:
            coluna_treino = pd.get_dummies(self.df_treino[nome_coluna], drop_first=False, prefix=prefix)
        else:
            coluna_treino = pd.get_dummies(self.df_treino[nome_coluna], drop_first=False)
        
        self.df_treino = pd.concat([self.df_treino, coluna_treino], axis=1)

        self.df_treino = self.df_treino.drop(nome_coluna, axis=1)

        self.X_treino = self.df_treino.drop(self.coluna_target, axis=1)
        self.y_treino = self.df_treino[self.coluna_target]


    def binarizar_numerico(self, nome_coluna, num_bins):
        min_value = self.df_treino[nome_coluna].min()
        max_value = self.df_treino[nome_coluna].max()
        prefixo = nome_coluna
        bin_width = (max_value - min_value) / num_bins
        bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
        bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
        self.df_treino[nome_coluna+"_label"] = pd.cut(self.df_treino[nome_coluna], bins=bin_edges, labels=bins_labels)
        dummies_treino = pd.get_dummies(self.df_treino[nome_coluna+"_label"], drop_first=False)

        self.df_treino = pd.concat([self.df_treino, dummies_treino], axis=1)

        self.df_treino = self.df_treino.drop(nome_coluna, axis=1)
        self.df_treino = self.df_treino.drop(nome_coluna+"_label", axis=1)
        

        self.X_treino = self.df_treino.drop(self.coluna_target, axis=1)
        self.y_treino = self.df_treino[self.coluna_target]
    
    def deixar_consistente(self):
        self.df_treino  = self.df_treino.dropna()
        
        self.df_treino = self.df_treino.drop_duplicates()

        colunas = self.df_treino.drop(self.coluna_target, axis=1)
        duplicadas = self.df_treino.duplicated(subset=colunas, keep=False)
        df_duplicates = self.df_treino[duplicadas]
        self.df_treino = self.df_treino.drop(df_duplicates.index)

        self.X_treino = self.df_treino.drop(self.coluna_target, axis=1)
        self.y_treino = self.df_treino[self.coluna_target]

    
    def salvar_datasets(self, nome_arquivo_treino):
        self.df_treino.to_csv(nome_arquivo_treino, index=False)



###########################################################################
class BinarizarInstancia:
    def __init__(self, instancia, coluna_target=""):
        self.df_instancia = pd.DataFrame(instancia)
        if(len(coluna_target) == 0):
            self.coluna_target = ""
            self.X = self.df_instancia
            self.y = None
        else:
            self.coluna_target = coluna_target
            self.X = self.df_instancia.drop(coluna_target, axis=1)
            self.y = self.df_instancia[coluna_target]


    def binarizar_categorico(self, nome_coluna, possiveis_valores, prefix=""):
        for valor in possiveis_valores:
            if prefix:
                nome_coluna_binaria = f"{prefix}_{valor}"
            else:
                nome_coluna_binaria = valor
            self.df_instancia[nome_coluna_binaria] = self.df_instancia[nome_coluna] == valor
        
        self.df_instancia = self.df_instancia.drop(nome_coluna, axis=1)
        
        if len(self.coluna_target) == 0:
            self.X = self.df_instancia
            self.y = None
        else:
            self.X = self.df_instancia.drop(self.coluna_target, axis=1)
            self.y = self.df_instancia[self.coluna_target]

        

    def binarizar_numerico(self, nome_coluna, num_bins, min_value, max_value):
        prefixo = nome_coluna
        bin_width = (max_value - min_value) / num_bins
        bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
        bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
        self.df_instancia[nome_coluna+"_label"] = pd.cut(self.df_instancia[nome_coluna], bins=bin_edges, labels=bins_labels)
        dummies_treino = pd.get_dummies(self.df_instancia[nome_coluna+"_label"], drop_first=False)

        self.df_instancia = pd.concat([self.df_instancia, dummies_treino], axis=1)
        self.df_instancia = self.df_instancia.drop(nome_coluna, axis=1)
        if(len(self.coluna_target) == 0):
            self.X = self.df_instancia
            self.y = None
        else:
            self.X = self.df_instancia.drop(self.coluna_target, axis=1)
            self.y = self.df_instancia[self.coluna_target]
   


###########################################################################
class Explicabilidade:
    def __init__(self, df_treino, df_teste, resposta, coluna_target=""):
        self.coluna_target = coluna_target
        self.df_treino = df_treino
        self.df_teste = df_teste
        self.resposta = resposta
        self.argumentos_possiveis = []
        self.argumentos_consistentes = []
        self.argumentos_inconsistentes = []
        self.argumentos_redundantes = []
        self.arguments_essenciais = []

    def gerar_argumentos_possiveis(self):
        id = 0
        lista_premissas_argumentos_possiveis = []
        for index, linha in self.df_teste.iterrows():
            if(len(self.coluna_target) == 0):
                self.X = linha
                self.y = None
            else:
                self.X = linha.drop(self.coluna_target)
                self.y = linha[self.coluna_target]
            
            atributos_com_1 = [col for col in self.X.index if self.X[col] == 1]

            for i in range(1, len(atributos_com_1)+1):
                combinacoes = combinations(atributos_com_1, i)
                for combinacao in combinacoes:
                    temp = set(combinacao)            
                    
                    if temp not in lista_premissas_argumentos_possiveis:
                        lista_premissas_argumentos_possiveis.append(temp)
                        self.argumentos_possiveis.append({"premissa": temp, "conclusao": self.resposta, "id": id})
                        id += 1


    def exibir_argumentos_possiveis(self):
        print("Argumentos possíveis: ")
        print("Instância: ")
        print(self.df_teste)
        for item in self.argumentos_possiveis:
            print(f"{item['id']}<{item['premissa']}, {item['conclusao']}>")

    def gerar_argumentos_consistentes(self):
        for argumento in self.argumentos_possiveis:
            condicao = (self.df_treino[list(argumento["premissa"])] == True).all(axis=1)
            df_filtrado = self.df_treino.loc[condicao]
            valores_unicos = df_filtrado[self.coluna_target].unique()
            #print(valores_unicos)

            if(len(valores_unicos) == 1):
                argumento["conclusao"] = valores_unicos[0]
                self.argumentos_consistentes.append(argumento)
            else:
                argumento["conclusao"] = ""
                self.argumentos_inconsistentes.append(argumento)



    def exibir_argumentos_consistentes(self):
        print("Argumentos consistentes: ")
        for item in self.argumentos_consistentes:
            print(f"{item['id']}<{item['premissa']}, {item['conclusao']}>")

        print("Argumentos inconsistentes: ")
        for item in self.argumentos_inconsistentes:
            print(f"{item['id']}<{item['premissa']}, {item['conclusao']}>")

    def gerar_argumentos_essenciais(self):
        indices_excluir = []
        for i in range(0, len(self.argumentos_consistentes)):
            premissas_origem = set(self.argumentos_consistentes[i]["premissa"])
            if i not in indices_excluir:
                for j in range(i+1, len(self.argumentos_consistentes)):
                    premissas_destino = set(self.argumentos_consistentes[j]["premissa"])
                    if premissas_origem.issubset(premissas_destino):
                        if j not in indices_excluir:
                            indices_excluir.append(j)
        for i in range(0, len(self.argumentos_consistentes)):
            if i not in indices_excluir:
                self.arguments_essenciais.append(self.argumentos_consistentes[i])
            else:
                self.argumentos_redundantes.append(self.argumentos_consistentes[i])

    def exibir_argumentos_essenciais(self):
        print("Argumentos essenciais: ")
        for item in self.arguments_essenciais:
            print(f"{item['id']}<{item['premissa']}, {item['conclusao']}>")

        print("Argumentos redundantes: ")
        for item in self.argumentos_redundantes:
            print(f"{item['id']}<{item['premissa']}, {item['conclusao']}>")


    def explicar(self):
        justificaveis = []
        for argumento in self.arguments_essenciais:
            if(argumento["conclusao"] == self.resposta):
                justificaveis.append(argumento)
        
        self.explicabilidade = []
        totalPremissas = 0
        for argumento in justificaveis:
            t = set(argumento["premissa"])
            if(len(t) > totalPremissas):
                totalPremissas = len(t)
                self.explicabilidade = []
                self.explicabilidade.append(argumento)
            else:
                self.explicabilidade.append(argumento)


    def exibir_explicacao(self):
        print("Melhor explicação: ")
        for item in self.explicabilidade:
            print(f"{item['id']}<{item['premissa']}, {item['conclusao']}>")


    def estatistica_argumentos(self):
        classes = self.df_treino[self.coluna_target].unique()



        
        classes_extraidas = [item['conclusao'] for item in self.arguments_essenciais]

        # Use Counter para contar as ocorrências de cada classe
        from collections import Counter
        contagem_classes = Counter(classes_extraidas)

        # Inicialize contagens com zero para todas as classes possíveis
        for classe in classes:
            if classe not in contagem_classes:
                contagem_classes[classe] = 0

        # Exiba as contagens
        print("Estatísticas de argumentos: ")
        print(contagem_classes)

        if all(contagem > 0 for contagem in contagem_classes.values()):
            input("Todos os valores são maiores que 0. Pressione Enter para continuar...")


        return contagem_classes
        """
        contagem_por_target = {}

        # Percorre cada argumento na lista
        for arg in lista_argumentos:
            # Obtem o valor 'target' do argumento
            valor_target = arg[self.coluna_target]

            # Verifica se o valor 'target' já existe no dicionário de contagem
            if valor_target not in contagem_por_target:
                # Cria uma nova entrada no dicionário com valor inicial 0
                contagem_por_target[valor_target] = 0

            # Incrementa a contagem para o valor 'target' encontrado
            contagem_por_target[valor_target] += 1

        # Exibe a contagem de itens para cada valor 'target'
        print(contagem_por_target)
        """
