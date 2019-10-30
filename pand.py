import sys
import argparse
import csv
from pandas import read_csv
import pandas
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Process analyse pred")

parser.add_argument('context', choices=['pand'], help='context dans lequel nous allons travailler')

subparsers = parser.add_subparsers(dest='Action',required=True)

parser_ope = subparsers.add_parser("ope")
parser_ope.add_argument('newcol')
parser_ope.add_argument('col1')
parser_ope.add_argument('signeope')
parser_ope.add_argument('col2')
parser_ope.add_argument('--file')
parser_ope.add_argument('--extract')

parser_regr = subparsers.add_parser("regr")
parser_regr.add_argument('colactu')
parser_regr.add_argument('colpred')
parser_regr.add_argument('--file')
parser_regr.add_argument('--extract')

parser_matrix = subparsers.add_parser("matrix")
parser_matrix.add_argument('colactu')
parser_matrix.add_argument('colpred')
parser_matrix.add_argument('--file')
parser_matrix.add_argument('--extract')

parser_seuil = subparsers.add_parser("seuil")
parser_seuil.add_argument('colactu')
parser_seuil.add_argument('colpred')
parser_seuil.add_argument('--file')
parser_seuil.add_argument('--extract')

parser_auc = subparsers.add_parser("auc")
parser_auc.add_argument('--file')
parser_auc.add_argument('--extract')

args = parser.parse_args() 
print(args)


df = read_csv(f'{args.file}')

def addition():
    if args.signeope == '+' :
        df[f'{args.newcol}'] = df[f'{args.col1}'] + df[f'{args.col2}']
    elif args.signeope =='-':
        df[f'{args.newcol}'] = df[f'{args.col1}'] - df[f'{args.col2}']
    elif args.signeope =='*':
        df[f'{args.newcol}'] = df[f'{args.col1}'] * df[f'{args.col2}']
    elif args.signeope =='div':
        df[f'{args.newcol}'] = df[f'{args.col1}'] / df[f'{args.col2}']

########## Fonction pour regréssion context #############

def regr():
    df['pourcerror'] = (df[f'{args.colpred}'] - df[f'{args.colactu}']) / df[f'{args.colactu}']
    df['errorabs'] = round(abs(df['pourcerror']), 2)
    df.loc[0,'mape'] = round(np.average(df['errorabs'])*100, 2)

def posneg():

    erreurpos = 0
    erreurneg = 0
    predparfaite = 0

    for i in df['pourcerror'] :
        if i>0:
            erreurpos +=1
        elif i<0:
            erreurneg +=1
        else :  
            predparfaite +=1
    print(f'On a {erreurpos} erreur positive, {erreurneg} erreur negative, {predparfaite} prediction parfaite')
    df.loc[0,'positive erreur'] = erreurpos
    df.loc[0,'negatif erreur'] = erreurneg
    df.loc[0,'prediction parfaite'] = predparfaite

########## Fonction pour classification context #############  

def matrice(row):
    if args.Action == "seuil" :
        colpred = 'seuil_pred'
    elif args.Action == "matrix":
        colpred =args.colpred

    if row[f'{args.colactu}'] == row[f'{colpred}'] and row[f'{args.colactu}'] == 0 :
        val = "TN"
    elif row[f'{args.colactu}'] == row[f'{colpred}'] and row[f'{args.colactu}'] == 1 :
        val = "TP"
    elif row[f'{args.colactu}'] > row[f'{colpred}']:
        val = "FN"
    else:
        val = "FP"
    return val

def topErr():
    filtered = df.loc[df['Error'].isin(["FN","FP"])]
    filemane = input("Nom du fichier avec extention :")
    filemane = str(filemane)
    print("Extraction commencé")
    filtered.to_csv(filemane, index=False)
    print("Extraction terminé")
    # filtered = (filtered.nlargest(100,('0 probability'))) #nsmalest

def auc():

    positive = (df['target'] == 1)
    count_pos=len(df.loc[positive])
    count_neg=len(df.loc[~positive])

    result = df[['target','1 probability']]
    threshold_list = result.sort_values(by='1 probability',ascending=False)['target'].values

    auc = 0.0
    P_cumul = 0
    for i in range(len(threshold_list)):
        if threshold_list[i] == 1:
            P_cumul += 1
        else:
            auc += P_cumul
            
    auc = auc/(count_pos*count_neg)

    print(f"La valeur de l'AUC est {auc}") 
    return auc 

   
def matrix(colpred):

    df[f'{colpred}'] = df[f'{colpred}'].map({1:2, 0:0})
    mat = df[f'{colpred}'] - df[f'{args.colactu}']

    trueNegative = 0
    falseNegative =0
    truePositif = 0
    falsePositive = 0  

    for i in mat:
        if i==2:
            falsePositive +=1
        elif i==0 :
            trueNegative +=1
        elif i == 1:
            truePositif+=1 
        elif i ==-1:
            falseNegative +=1

    df.loc[0,'truePositif'] = truePositif
    df.loc[0,'falseNegative'] = falseNegative
    df.loc[0,'trueNegative'] = trueNegative
    df.loc[0,'falsePositive'] = falsePositive
    prec = (trueNegative+truePositif)/(truePositif+falseNegative+trueNegative+falsePositive)
    df.loc[0,'accuracy'] = round(prec,2)

    df[f'{colpred}'] = df[f'{colpred}'].map({2:1, 0:0})
    print(f'On a {truePositif} TP {falseNegative} FN {trueNegative} TN {falsePositive} FP et une precision de : {prec}')

def matcout(coutTP,coutFN,coutTN,coutFP):
    df.loc[0,'cout tp'] =  round(df.loc[0,'truePositif'] * coutTP,2) 
    df.loc[0,'cout fn'] =  round(df.loc[0,'falseNegative'] * coutFN,2)
    df.loc[0,'cout tn'] =  round(df.loc[0,'trueNegative'] * coutTN,2)
    df.loc[0,'cout fp'] =  round(df.loc[0,'falsePositive'] * coutFP,2)
    res = (df['cout tp']) + (df['cout fn']) + (df['cout tn']) + (df['cout fp'])
    df['resultat'] = res
    resret = df.loc[0,'resultat']
    return resret

def seuil():
    countSeuil = 0
    bestResult = 0
    newResult = 0
    varSeuil = 0.26
    optiSeuil = 0.1

 
    coutTP = float(input("saisir cout TP :"))
    coutFN = float(input("saisir cout FN :"))
    coutTN = float(input("saisir cout TN :"))
    coutFP = float(input("saisir cout FP :"))
    modGraph = bool(input("Voulez vous un graphique du seuil ?"))
    print(modGraph)
    
    if modGraph :
        tablx = []
        tably = []

    while countSeuil <= 10:
        print("count : ",countSeuil)
        varSeuil =round(varSeuil,2)
        print("valeur seuil : ",varSeuil)
        countline = 0
        for i in df['1 probability'] :
            if i < varSeuil :
                df.loc[countline,'seuil_pred'] = 0
            else:
                df.loc[countline,'seuil_pred']= 1
            countline +=1    

        colpred = 'seuil_pred'
        matrix(colpred)
        newResult = matcout(coutTP,coutFN,coutTN,coutFP)
        print("Resultat : ",newResult)

        if modGraph :
            tablx.append(varSeuil)
            tably.append(newResult)
        
        df.loc[countSeuil,'varSeuil'] = varSeuil
        df.loc[countSeuil, 'newResult']= newResult

        if newResult > bestResult :
            optiSeuil = varSeuil
            bestResult = newResult
            print(f"Nouveau meilleur résultat de {bestResult}, avec un seuil opti de {optiSeuil}")

        countSeuil+=1    
        varSeuil += 0.01
        print("pass")

    print("finally") 

    countline = 0
    for i in df['1 probability'] :
        if i < optiSeuil :
            df.loc[countline,'seuil_pred'] = 0
        else: 
            df.loc[countline,'seuil_pred']= 1
        countline +=1     
    
    colpred = 'seuil_pred'
    matrix(colpred)
    matcout(coutTP,coutFN,coutTN,coutFP)
    df.loc[0,"seuil_opti"] = optiSeuil
    df['Error'] =df.apply(matrice,axis=1)

    print(f"Matrice de couts généré, resultat = {df.loc[0,'resultat']}")
    print(f"Le meilleur résultat est de {bestResult}, avec un seuil opti de {optiSeuil}")    
    
    if modGraph :
        graph_simple(tablx,tably,'Threshold evaluation','graphSeuilOpti')


############# Graphique ##########################

def graph_simple(x,y,titre,name):
    plt.plot(x,y)
    plt.title(f"{titre}")
    plt.draw()
    plt.savefig(f'{name}.png', dpi=200) 


def graph_double(x,y,y2,titre,name):
    plt.plot(x, y, "r--", label="Ensemble")
    plt.plot(x, y2, "b:o", label="Deep")
    plt.title(f"{titre}")
    plt.draw()
    plt.savefig(f'{name}.png', dpi=200)                

################## Gestion du script ##############################                  

def extraction() :
    print("Extraction commencé")
    df.to_csv(f'{args.extract}', index=False)
    print("Extraction terminé")

def voidUpdate() :

    if args.Action == 'ope':

        addition()
        print("calcul terminé")

    elif args.Action == 'regr':

        regr()

    elif args.Action == 'matrix':

        df['Error'] =df.apply(matrice,axis=1)
        matrix(args.colpred)
        coutTP = float(input("saisir cout TP :"))
        coutFN = float(input("saisir cout FN :"))
        coutTN = float(input("saisir cout TN :"))
        coutFP = float(input("saisir cout FP :"))
        matcout(coutTP,coutFN,coutTN,coutFP)

    elif args.Action == 'seuil' :

        seuil() 
    elif  args.Action == 'auc':
        auc()

    if args.extract:

        extraction()


    affichage = """
                                Choisissez une option:

            ---Classification---                     ---Regression---       

    \t1: Appliquer erreur majeur            \t2: Appliquer positive negatif analyse


                            \t3: Afficher l'opération
                            \t4: Extraire la sélection en mémoire
                            \t5: Terminer
        """
    option_choisie = 0

    while option_choisie != 5 :

        print(affichage)
        option_choisie = input("Choisir option : ")
        option_choisie = int(option_choisie)

        if option_choisie == 1:
            topErr()

        elif option_choisie ==2:
            posneg()

        elif option_choisie ==3:
            print(df)

        elif option_choisie == 4:
            try :
                filemane = input("Nom du fichier avec extention :")
                filemane = str(filemane)
                args.extract = filemane
                extraction()  
                args.extract = None
            except : 
                print("Extraction Impossible")       
                  
voidUpdate()

print("Programme terminé !")



    # tab1['Probability']=tab2['data insert']

    # Replace all NaN elements with 0s.
    
    # >>> df.fillna(0)
    #     A   B   C   D
    # 0   0.0 2.0 0.0 0
    # 1   3.0 4.0 0.0 1
    # 2   0.0 0.0 0.0 5
    # 3   0.0 3.0 0.0 4
    
    # We can also propagate non-null values forward or backward.
    
    # >>> df.fillna(method='ffill')
    #     A   B   C   D
    # 0   NaN 2.0 NaN 0
    # 1   3.0 4.0 NaN 1
    # 2   3.0 4.0 NaN 5
    # 3   3.0 3.0 NaN 4