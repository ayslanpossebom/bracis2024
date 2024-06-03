import pandas as pd
import os 
from minhas_classes import BinarizarDatasetTreino, BinarizarDatasetTreinoTeste, BinarizarInstancia, Explicabilidade

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



url = os.path.dirname(os.path.abspath(__file__)) + "\\datasets\\"



############### DADOS DO DATASET ############################
dataset_original_treino = url + "titanic_train.csv" 
dataset_original_teste = url + "titanic_test.csv" 

dataset_binarizado_treino = url + "titanic_train_binary.csv"
dataset_binarizado_teste = url + "titanic_test_binary.csv"

coluna_target = "Survived"



#Abrir dataset:
df_treino = pd.read_csv(dataset_binarizado_treino)
df_teste = pd.read_csv(dataset_binarizado_teste)

#df_teste = df_teste.iloc[[1]]         #Definir caso de teste 




#Machine learning
X_train = df_treino.drop([coluna_target, 'PassengerId'], axis=1)
y_train = df_treino[[coluna_target]].values.ravel()



def save_to_file(content, filename):
    with open(filename, 'a') as file:  # Use 'a' mode to append to the file if it exists
        if isinstance(content, str):
            file.write(content + '\n')  # Add a newline after the string
        elif isinstance(content, list):
            for item in content:
                file.write(f"{item}\n")
        else:
            raise TypeError("Content must be either a string or a list")

cont = 2
for index, valores in df_teste.iterrows():
    #X_test = df_teste.drop([coluna_target], axis=1)
    #y_test = df_teste[[coluna_target]].values.ravel()

    linha_dataframe = pd.DataFrame(valores.values.reshape(1, -1), columns=valores.index)
    #X_test = linha_dataframe.drop([coluna_target], axis=1)
    temp = linha_dataframe['PassengerId'][0]
    X_test = linha_dataframe.drop([coluna_target, 'PassengerId'], axis=1)
    y_test = linha_dataframe[[coluna_target]].values.ravel()

    print("Instância a ser justificada: ", temp)
    #print(linha_dataframe)
    for coluna in linha_dataframe.columns:
        print(f"{coluna}: {linha_dataframe[coluna].values[0]}")


    #Classificação
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

    # Avaliação
    #print("Árvore de Decisão    ", dt_predictions)
    #print("K-Nearest Neighbors  ", knn_predictions)
    #print("Naive Bayes          ", nb_predictions)

    fim = pd.DataFrame({
        'corretos': pd.Series(y_test),
        'arvore': pd.Series(dt_predictions),
        'knn': pd.Series(knn_predictions),
        'bayes': pd.Series(nb_predictions)
    })
    print("Respostas obtidas: ")
    print(fim)
    
    #Definir qual resposta será utilizada para justificar
    resposta = dt_predictions[0]
    #resposta = knn_predictions[0]
    #resposta = nb_predictions[0]

    #exp = Explicabilidade(df_treino=df_treino, df_teste=df_teste, resposta=resposta, coluna_target=coluna_target)
    exp = Explicabilidade(df_treino=df_treino, df_teste=linha_dataframe, resposta=resposta, coluna_target=coluna_target)
    
    exp.gerar_argumentos_possiveis()
    #exp.exibir_argumentos_possiveis()
    
    exp.gerar_argumentos_consistentes()
    #exp.exibir_argumentos_consistentes()
    
    exp.gerar_argumentos_essenciais()
    exp.exibir_argumentos_essenciais()

    exp.explicar()
    exp.exibir_explicacao()

    x = exp.estatistica_argumentos()
    print("#"*40)

    """
    save_to_file(f"ARGUMENTO: ({cont})", "explicabilidade.txt")
    save_to_file(list(linha_dataframe), "explicabilidade.txt")
    save_to_file("CONSISTENTES:", "explicabilidade.txt")
    save_to_file(exp.argumentos_consistentes, "explicabilidade.txt")
    save_to_file("INCONSISTENTES:", "explicabilidade.txt")
    save_to_file(exp.argumentos_consistentes, "explicabilidade.txt")
    save_to_file("ESSENCIAIS:", "explicabilidade.txt")
    save_to_file(exp.arguments_essenciais, "explicabilidade.txt")
    save_to_file("REDUNDANTES:", "explicabilidade.txt")
    save_to_file(exp.argumentos_redundantes, "explicabilidade.txt")
    save_to_file("EXPLICACAO:", "explicabilidade.txt")
    save_to_file(exp.explicabilidade, "explicabilidade.txt")
    save_to_file("ESTATISTICAS:", "explicabilidade.txt")
    save_to_file(list(x), "explicabilidade.txt")
    """
    
    cont += 1
    



