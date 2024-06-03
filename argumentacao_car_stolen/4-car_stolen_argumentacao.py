import os
import pickle
import pandas as pd


url = os.path.dirname(os.path.abspath(__file__)) + "\\"
path_essential_arguments = url+"car_stolen_argumentos.ob"


#Carregar lista de argumentos

with open(path_essential_arguments, 'rb') as fp:
    argumentos = pickle.load(fp)

print(f"Total de argumentos essenciais: {len(argumentos)}")
for i in argumentos:
    print(i)

print()



class Argumento:
    def __init__(self, nome, premissa, conclusao):
        self.nome = nome
        self.premissa = premissa
        self.conclusao = conclusao
        self.ataca = []
    
    def atacar(self, argumento):
        self.ataca.append(argumento)

    def pode_atacar(self, argumento):
        #se tiverem conclusões diferentes
        if(self.conclusao != argumento.conclusao):
            #se tiver ao menos uma premissa igual
            premisasIguais = self.premissa.intersection(argumento.premissa)
            if(len(premisasIguais) > 0):
                atual = str(self.nome)+"<"+str(self.premissa)+" , "+str(self.conclusao)+">"
                destino = str(argumento.nome)+"<"+ str(argumento.premissa)+" , "+str(argumento.conclusao)+">"
                print(atual + " ataca " + destino)
                return True
            else:
                #atual = "<"+str(self.premissa)+" , "+str(self.conclusao)+">"
                #destino = "<"+ str(argumento.premissa)+" , "+str(argumento.conclusao)+">"
                #print(atual + " NAO ataca " + destino)
                return False
    def exibirArgumento(self):
        print(str(self.nome)+"<"+str(self.premissa)+" , "+str(self.conclusao)+">")






#encontar um teste
df = pd.read_csv(url+"car_stolen_test_binary.csv")
#df = df.sample(1)

#df = df.loc[(df['Outlook_Sunny'] == 1) & (df['Temperature_Cool'] == 1) & (df['Humidity_Normal'] == 1) & (df['Wind_Weak'] == 1)]



coluna_target = "Stolen"

for index, row in df.iterrows():

    lista_argumentos = []
    i = 1
    for arg in argumentos:
        temp = Argumento(i, arg["premissas"], arg["conclusao"])
        lista_argumentos.append(temp)
        temp.exibirArgumento()
        i += 1



    #Criar relações de ataque do framework
    i = 0
    for i in range(0, len(lista_argumentos)):
        for j in range(0, len(lista_argumentos)):
            if(i != j):
                if(lista_argumentos[i].pode_atacar(lista_argumentos[j])):
                    lista_argumentos[i].atacar(lista_argumentos[j])


    #gerando o caso de testes
    features = set([col for col in df.drop(coluna_target, axis=1).columns if row[col] == 1])
    #features = set(["Humidity_Normal", "Temperature_Cool", "Wind_Weak", "Outlook_Sunny"])
    
    #Gerar ataques do argumento de teste
    argTeste = Argumento("Teste", features, "")
    print("ARGUMENTO TESTE TEM FEATURES: ", features)
    lista_argumentos.append(argTeste)
    for arg in lista_argumentos[:-1]:
        premisasIguais = features.intersection(arg.premissa)
        #if(len(premisasIguais) > 0 and len(premisasIguais)<len(arg.premissa) ):
        if(len(premisasIguais)<len(arg.premissa) ):
            print(f"{argTeste.nome} ataca {arg.nome}")
            argTeste.atacar(arg)

    #gerar AAF para usar em ConArg
    #https://conarg.dmi.unipg.it/web_interface.php 
    for arg in lista_argumentos:
        print(f"arg({arg.nome}).")

    for arg in lista_argumentos:
        for ataque in arg.ataca:
            print(f"att({arg.nome},{ataque.nome}).")

    

            

    #Abstract argumentation
    from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
    from py_arg.abstract_argumentation_classes.argument import Argument
    from py_arg.abstract_argumentation_classes.defeat import Defeat
    from py_arg.algorithms.semantics.get_grounded_extension import get_grounded_extension


    arguments = []
    defeats = []
    for arg in lista_argumentos:
        argument = Argument(str(arg.nome))
        arguments.append(argument)



    for arg in lista_argumentos:
        origem = ""
        for item in arguments:
            if str(item.name) == str(arg.nome):
                origem = item
                break   
        for ataque in arg.ataca:
            destino = ""
            for item in arguments:
                if str(item.name) == str(ataque.nome):
                    destino = item
                    break
            defeat = Defeat(origem, destino)
            defeats.append(defeat)
        

    af = AbstractArgumentationFramework('af', arguments, defeats)
    ces = get_grounded_extension(af)
    print("Grounded extension: ", ces)
    print()

    for item in ces:
        for argumento in lista_argumentos:
            if str(argumento.nome) == str(item.name):
                argumento.exibirArgumento()
                break
    print()



from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

knn = KNeighborsClassifier(n_neighbors=3)
dtree = DecisionTreeClassifier()

df_treino = pd.read_csv(url+"car_stolen_binary.csv")
X = df_treino.drop(coluna_target, axis=1)
y = df_treino[coluna_target]

knn.fit(X, y)
dtree = dtree.fit(X, y)


X_test = df.drop(coluna_target, axis=1)
y_test = df[coluna_target]
print("KNN Predicted: ", knn.predict(X_test))
print("Tree Predicted: ", dtree.predict(X_test))
print("Resposta correta: ", list(y_test))






