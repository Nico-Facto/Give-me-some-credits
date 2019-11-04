'''BigMl module for Data/Ia Dev Script -- Nicolas Autexier -- contact = nicolas.atx@gmx.fr '''

from pandas import read_csv
import pandas
import os
import matplotlib.pyplot as plt
import sys
from bigml.api import BigML
from graph import rViz
from Facto import secureLog as SL

def initproject():
    pr = str(input("project/id : "))
    api = BigML(f'{SL.bigUseur}', f'{SL.bigApiKey}', project=f'{pr}')
    return api
api = initproject()

################ Prediction sur fichier train/validation ###################  

def predmeth1(file,fileTest,splitTrain,splitTest,mod,objectifField,export) :


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
    api.ok
    source = api.create_source(file)
    api.ok

    origin_dataset = api.create_dataset(source)
    api.ok
    test_testdataset = api.create_dataset(source_test)
    api.ok

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

########### creation de prediction #####################
 
class createNewPred() :

    @staticmethod
    def newDataSet():

        file = str(input("Nom du fichier full train : "))
        splitTrain = float(input("valeur split train : "))
        splitTest = float(input("valeur split test : "))
        mod = str(input("Model selectioné : "))
        objectifField = str(input("Nom du champs objectif : "))
        export = str(input("Nom du fichier exporté : "))

        source = api.create_source(f"{file}")
        api.ok
        origin_dataset = api.create_dataset(source)
        api.ok
        train_dataset = api.create_dataset(origin_dataset, {"name": "VarTraining", "sample_rate": splitTrain})
        api.ok
        test_dataset = api.create_dataset(origin_dataset, {"name": "VarTest", "sample_rate": splitTest})
        api.ok
        file = train_dataset
        fileTest = test_dataset
        print("split ok")

        predmeth1(file,fileTest,splitTrain,splitTest,mod,objectifField,export)

    @staticmethod
    def loaddDataSet() :

        n_dataset = str(input("train dataset/id : "))
        train_dataset = api.get_dataset(f"{n_dataset}")
        t_dataset = str(input("test dataset/id : "))
        test_dataset = api.get_dataset(f"{t_dataset}")

        mod = str(input("Model selectioné : "))
        objectifField = str(input("Nom du champs objectif : "))
        export = str(input("Nom du fichier exporté : "))

        file = train_dataset
        fileTest = test_dataset
        splitTrain = 0
        splitTest = 0
        print("dataset load ok") 

        predmeth1(file,fileTest,splitTrain,splitTest,mod,objectifField,export)

    @staticmethod
    def predOnProdSet() :
        
        file = str(input("Nom du fichier full train : "))
        fileTest = str(input("Nom du fichier full Prod : "))
        mod = str(input("Model selectioné : "))
        objectifField = str(input("Nom du champs objectif : "))
        export = str(input("Nom du fichier exporté : "))

        predmeth1Kagg(objectifField,mod,file,fileTest,export)

##################### Meth pour split les full test en dev test et test_test(genre kaggle) ##############################

# test_full = api.create_dataset(origin_dataset, {"name": "test_full"})
# rest = 1-splitTrain
# sep= rest/2
# print("reste pour test : ",sep)
# test_test = api.create_dataset(test_full, {"name": "dev_test", "sample_rate": sep})
# test_test = api.create_dataset(test_full, {"name": "test_final", "sample_rate": sep}) 


########### Analyse BigMl #####################
class analyserML() :
    
    @staticmethod
    def compar2model():
    ###### Comparaison AUC/Ammont of data entre 2 models #########
        ###### variable d'environement #########
        load_set_train = str(input("Id du fichier train : "))
        load_set_test = str(input("Id du fichier test : "))
        model_One = str(input("model 1 : "))
        model_two = str(input("model 2 : "))
        var = 0.1
        tablex = []
        tabley = []
        tabley2 = []
        ###### Routine Big ML #########
        for i in range(0,10):
            print("Step : ",i)
        
            origin_dataset = api.get_dataset(f"{load_set_train}")
            api.ok
            train_dataset = api.create_dataset(origin_dataset, {"name": "AmountData2", "sample_rate": var})
            api.ok
            fileTest = api.get_dataset(f"{load_set_test}")
            api.ok

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

    @staticmethod        
    def summuary() : 
    ###### Methode publié sur discord par Christophe #########
        model_One = str(input("model ID : "))
        var_mod = str(input("model name : "))
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


    
