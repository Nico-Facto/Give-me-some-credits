from pandas import read_csv
import pandas
import os
import matplotlib.pyplot as plt
import sys
from bigml.api import BigML
from graph import rViz ###### importer le script graph.py #########

api = BigML('NICOFACTO', 'f1c450758df16e375da99c36e7094fb901644232', project='project/5d94a4095a213962af00009a')

###### Comparaison AUC/Ammont of data entre 2 models #########
def compar2model(load_set_train,load_set_test,model_One,model_two):
    ###### variable d'environement #########
    var = 0.1
    tablex = []
    tabley = []
    tabley2 = []
    ###### Routine Big ML #########
    for i in range(0,10):
        print("Step : ",i)

        origin_dataset = api.get_dataset(f"{load_set_train}")
        train_dataset = api.create_dataset(origin_dataset, {"name": "AmountData2", "sample_rate": var})
        fileTest = api.get_dataset(f"{load_set_test}")

        modvar = modelOperate(model_One,train_dataset)

        evaluation = api.create_evaluation(modvar,fileTest)
        api.ok(evaluation)
        auc = evaluation['object']['result']['model']['average_area_under_roc_curve']
        print(f"auc {model_One} : ", auc, "avec un split : ", var)

        tablex.append(var)
        tabley.append(auc)

        ###### Appel de la fonction create_"" #########
        modvar = modelOperate(model_two,train_dataset)

        evaluation = api.create_evaluation(modvar,fileTest)
        api.ok(evaluation)
        auc = evaluation['object']['result']['model']['average_area_under_roc_curve']
        print(f"auc {model_two} : ", auc, "avec un split : ", var)

        tabley2.append(auc)

        var += 0.1
        var = round(var,2)
    ###### Appel du module graphique #########
    rViz.graph_double(tablex,tabley,tabley2)
    rViz.extraction()        
        
def summuary(model_One,var_mod) : 
    ###### Methode publié sur discord par Christophe #########
    model = getModel(model_One,var_mod)
    importances = model['object']['importance']

    importances_named = dict()
    for column, importance in importances.items():
        column_name = model['object'][f'{var_mod}']['fields'][column]['name']
        importances_named[column_name] = [importance * 100]
    df = pandas.DataFrame.from_dict(importances_named, orient='index')
    df = df.sort_values(0, ascending=False)
    df.plot(kind='bar', color='green', legend=False)
    plt.draw()
    plt.show()

def modelOperate(mod,train_dataset) :
    if mod == 'ensemble':
        modvar = api.create_ensemble(train_dataset, {"objective_field": "target","name": "test_auc_curve"}) 
    elif mod == 'model':
        modvar = api.create_model(train_dataset, {"objective_field": "target","name": "test_auc_curve"})
    elif mod == 'deepnet': 
        modvar = api.create_deepnet(train_dataset, {"objective_field": "target","name": "test_auc_curve"}) 
    elif mod == 'linear': 
        modvar = api.create_linear_regression(train_dataset, {"objective_field": "target","name": "test_auc_curve"})      
    else :
        print("mod non pris en charge ! programme terminé !!")
        exit()
    return modvar    

def getModel(model_One,var_mod) :
    if var_mod == 'ensemble':
        setmod = api.get_ensemble(f"{model_One}") 
    elif var_mod == 'model':
        setmod = api.get_model(f"{model_One}")
    elif var_mod == 'deepnet': 
        setmod = api.get_deepnet(f"{model_One}") 
    elif var_mod == 'linear': 
        setmod = api.get_linear_regression(f"{model_One}")      
    else :
        print("mod non pris en charge ! programme terminé !!")
        exit()
    return setmod  

def voidUpdate():
    ###### gestion du script #########
    option_choisie = 0

    while option_choisie != 5 :
        
        affichage = """

                            Choisissez une option:

        \t1:  Auc/ammount Data           \t2: Summary Report

                            \t5: Terminer      

            """
        print(affichage)
        option_choisie = int(input("Choisir option : "))

        if option_choisie == 1: 

            load_set_train = str(input("Id du fichier train : "))
            load_set_test = str(input("Id du fichier test : "))
            model_One = str(input("model 1 : "))
            model_two = str(input("model 2 : "))
            compar2model(load_set_train,load_set_test,model_One,model_two)

        elif option_choisie == 2 :

            model_One = str(input("model ID : "))
            var_mod = str(input("model name : "))
            summuary(model_One,var_mod)

voidUpdate() 

print("Programme terminé")        

