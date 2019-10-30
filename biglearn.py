from pandas import read_csv
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
        print("mod non pris en charge")
        
    batch_prediction = api.create_batch_prediction(modvar, fileTest,{"all_fields": True,"probabilities": True})

    evaluation = api.create_evaluation(modvar,fileTest)
    
    api.ok(evaluation) ## ?? 
    # api.pprint(evaluation['object']['result'])
    # api.pprint(evaluation['object']['result']['Ensemble training']['accuracy'])
    # api.pprint(evaluation['object']['result']['Ensemble training']['average_area_under_roc_curve'])
  
    print("predict-lancée")

    api.ok(batch_prediction)
    api.download_batch_prediction(batch_prediction,filename=f"Pred_Files/{export}")

    print("prediction ok")

################ Prediction sur fichier test(ok pour Kaggle mais n'envoie pas le fichier) ###################

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
    
########### Gestion du script #####################

def voidUpdate():

    if  batchForKaggle == False :
        predmeth1(file,splitTrain,splitTest,mod,objectifField,export)
    else :
        predmeth1Kagg(objectifField,mod,file,fileTest,export) 

voidUpdate() 

print("Programme terminé")
    
    