from pandas import read_csv
import pandas
import os
import matplotlib.pyplot as plt
import argparse
import sys
from bigml.api import BigML

################# Option d'initialisation  du script ##################

batchForKaggle = False
createSet = False

affichage = """
                            Choisissez une option:

        ---Creation---                     ---Chargement---       

\t1: Créer nouveau dataset            \t2: Charger un dataset

                        \t3 code libre

    """

option_choisie = 0
print(affichage)
option_choisie = int(input("Choisir option : "))

if option_choisie == 1:

    createSet = True
    file = str(input("Nom du fichier full train : "))
    splitTrain = float(input("valeur split train : "))
    splitTest = float(input("valeur split test : "))

elif option_choisie == 2:
    createSet = False




mod = str(input("Model selectioné : "))
objectifField = str(input("Nom du champs objectif : "))
batchForKaggle = bool(input("Mode full test True or False ? ")) #faire entrée sans rien pour False


if batchForKaggle :
    fileTest = str(input("Nom du ficher full test : "))
else :
    fileTest = ("NULL")
    
export = str(input("Nom du fichier exporté : "))


api = BigML('NICOFACTO', 'f1c450758df16e375da99c36e7094fb901644232', project='project/5d94a4095a213962af00009a')
# api = BigML(project='project/5d94a4095a213962af00009a')
# A rentré dans le .yml pour projet et com ds le code.

print("programme initialisé")

if createSet :
    source = api.create_source(file)
    origin_dataset = api.create_dataset(source)
    train_dataset = api.create_dataset(origin_dataset, {"name": "VarTraining", "sample_rate": splitTrain})
    test_dataset = api.create_dataset(origin_dataset, {"name": "VarTest", "sample_rate": splitTest})
    file = train_dataset
    fileTest = test_dataset
    print("split ok")

    ##################### Meth pour split les full test en dev test et test_test(genre kaggle) ##############################

    # test_full = api.create_dataset(origin_dataset, {"name": "test_full"})
    # rest = 1-splitTrain
    # sep= rest/2
    # print("reste pour test : ",sep)
    # test_test = api.create_dataset(test_full, {"name": "dev_test", "sample_rate": sep})
    # test_test = api.create_dataset(test_full, {"name": "test_final", "sample_rate": sep}) 
                
else :
    train_dataset = api.get_dataset("dataset/5db6c8e3e47684746800c2e6")
    test_dataset = api.get_dataset("dataset/5db6c8e47811dd0557001103")
    file = train_dataset
    fileTest = test_dataset
    splitTrain = 0.8
    splitTest = 0.2
    print("dataset load ok") 

################ Prediction sur fichier train/validation ###################  

def predmeth1(file,splitTrain,splitTest,mod,objectifField,export) :

    if mod == 'ensemble':
        modvar = api.create_ensemble(file, {"objective_field": f'{objectifField}',"name": "Ensemble training"})  
    elif mod == 'model':
        modvar = api.create_model(file, {"objective_field": f'{objectifField}',"name": "model training"})
    elif mod == 'deepnet': 
        modvar = api.create_deepnet(file, {"objective_field": f'{objectifField}',"name": "deepnet training"}) 
    elif mod == 'linear': 
        modvar = api.create_linear_regression(file, {"objective_field": f'{objectifField}',"name": "linear training"})      
    else :
        print("mod non pris en charge ! programme terminé !!")
        
        
    batch_prediction = api.create_batch_prediction(modvar, fileTest,{"all_fields": True,"probabilities": True})

    evaluation = api.create_evaluation(modvar,fileTest)
    
    api.ok(evaluation)  
    # api.pprint(evaluation['object']['result'])
    # api.pprint(evaluation['object']['result']['Ensemble training']['accuracy'])
    # api.pprint(evaluation['object']['result']['Ensemble training']['average_area_under_roc_curve'])
  
    print("predict-lancée")

    api.ok(batch_prediction)
    api.download_batch_prediction(batch_prediction,filename=f"Pred_Files/{export}")

    print("prediction ok")

################ Prediction sur fichier test ###################

def predmeth1Kagg (objectifField,mod,file,fileTest,export) :

    source_test = api.create_source(fileTest)
    source = api.create_source(file)

    origin_dataset = api.create_dataset(source)
    test_testdataset = api.create_dataset(source_test)

    print("fichier ok")

    if mod == 'ensemble' :
        modvar = api.create_ensemble(origin_dataset, {"objective_field": f'{objectifField}',"name": "Ensemble full training"})  
    elif mod == 'model':
        modvar = api.create_model(origin_dataset, {"objective_field": f'{objectifField}',"name": "model full training"})
    elif mod == 'deepnet': 
        modvar = api.create_deepnet(origin_dataset, {"objective_field": f'{objectifField}',"name": "deep full training"}) 
    elif mod == 'linear': 
        modvar = api.create_linear_regression(origin_dataset, {"objective_field": f'{objectifField}',"name": "linear training"})  
    else :
        print("mod non pris en charge")

    batch_prediction = api.create_batch_prediction(modvar, test_testdataset,{"all_fields" : True,"probabilities" : True})

    print("predict-lancée")

    api.ok(batch_prediction)
    api.download_batch_prediction(batch_prediction,filename=f"Pred_Files/{export}")

    print("prediction ok")
    
################ Code libre #######################

def codelibre():

    var = 0.1

    tablex = []
    tabley = []

    for i in range(0,10):
        print("Step : ",i)
        origin_dataset = api.get_dataset("dataset/5db6c8e3e47684746800c2e6")
        train_dataset = api.create_dataset(origin_dataset, {"name": "AmountData", "sample_rate": var})
        modvar = api.create_ensemble(train_dataset, {"objective_field": "target","name": "test_auc_curve"})

        fileTest = api.get_dataset("dataset/5db6c8e47811dd0557001103")

        evaluation = api.create_evaluation(modvar,fileTest)
        api.ok(evaluation)
        auc = evaluation['object']['result']['model']['average_area_under_roc_curve']
        print("auc : ", auc, "avec un split : ", var)

        tablex.append(var)
        tabley.append(auc)

        var += 0.1
        var = round(var,2)

    print("first step done !!")

    var = 0.1
    tabley2 = []

    for i in range(0,10):  
        print("Step : ",i)

        origin_dataset = api.get_dataset("dataset/5db6c8e3e47684746800c2e6")
        train_dataset = api.create_dataset(origin_dataset, {"name": "AmountData", "sample_rate": var})
        modvar = api.create_deepnet(train_dataset, {"objective_field": "target","name": "test_auc_curve"})

        fileTest = api.get_dataset("dataset/5db6c8e47811dd0557001103")

        evaluation = api.create_evaluation(modvar,fileTest)
        api.ok(evaluation)
        auc = evaluation['object']['result']['model']['average_area_under_roc_curve']
        print("auc : ", auc, "avec un split : ", var)

        tabley2.append(auc)

        var += 0.1
        var = round(var,2)    

    print("Sec step done !!")

    plt.plot(tablex,tabley, "r--", label="Ensemble")
    plt.plot(tablex,tabley2, "b:o", label="Deep")
    plt.legend()
    plt.xlabel("Ammount of Data")
    plt.ylabel("A.U.C")
    plt.title("Evaluation")
    plt.grid(True)
    plt.draw()        
    print("Extraction commencé")

    isClean = False
    varName = "graph.png"
    add = 0
    while isClean != True :
        if not os.path.isfile(varName) :
            plt.savefig(f'{varName}', dpi=200)
            isClean = True
        else :
            split = varName.split(".")
            part_1 = split[0]+"_"+str(add)
            varName = ".".join([part_1,split[1]])
            add +=1  

        
def summuary() : 

    model = api.get_ensemble("ensemble/5db6e9abe47684746800c3c6")
    importances = model['object']['importance']

    importances_named = dict()
    for column, importance in importances.items():
        column_name = model['object']['ensemble']['fields'][column]['name']
        importances_named[column_name] = [importance * 100]
    df = pandas.DataFrame.from_dict(importances_named, orient='index')
    df = df.sort_values(0, ascending=False)
    df.plot(kind='bar', color='green', legend=False)
    plt.draw()
    plt.show()

########### Gestion du script #####################

def voidUpdate():

    if option_choisie == 3:
        codelibre()
        return
    if  batchForKaggle == False :
        predmeth1(file,splitTrain,splitTest,mod,objectifField,export)
    else :
        predmeth1Kagg(objectifField,mod,file,fileTest,export) 

voidUpdate() 

print("Programme terminé")
    
    
