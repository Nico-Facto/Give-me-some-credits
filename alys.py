'''Pandas module for Data/Ia Dev Script -- Nicolas Autexier -- contact = nicolas.atx@gmx.fr '''

import sys
import os
import csv
from pandas import read_csv
import pandas
import numpy as np
import matplotlib.pyplot as plt


############# Graphique ##########################

def graph_simple(x,y,titre,name):
    plt.grid(True)
    plt.plot(x,y,"r:o")
    plt.title(f"{titre}")
    plt.draw()
    plt.savefig(f"Rapport/{name}.png", dpi=200)
    plt.show() 
             
################## Gestion du script ##############################  

def matrice(row,output,pred):

    if row[f"{output}"] == row[f"{pred}"] and row[f"{output}"] == 0 :
        val = "TN"
    elif row[f"{output}"] == row[f"{pred}"] and row[f"{output}"] == 1 :
        val = "TP"
    elif row[f"{output}"] > row[f"{pred}"]:
        val = "FN"
    else:
        val = "FP"   
    return val


########## Class operation ############################

class ope():

    @staticmethod
    def operate(importfile):
        df = importfile

        newcol = str(input("non colonne resultat : "))
        col1 = str(input("non colonne 1 : "))
        signeope = str(input("signe opération (+ - * div) : "))
        col2 = str(input("non colonne 2 : "))

        if signeope == '+' :
            df[f'{newcol}'] = df[f'{col1}'] + df[f'{col2}']
        elif signeope =='-':
            df[f'{newcol}'] = df[f'{col1}'] - df[f'{col2}']
        elif signeope =='*':
            df[f'{newcol}'] = df[f'{col1}'] * df[f'{col2}']
        elif signeope =='div':
            df[f'{newcol}'] = df[f'{col1}'] / df[f'{col2}']
        return df    

    @staticmethod
    def codelibre(importfile):
        df = importfile
        print("NONE")
        return df

########## Class regréssion context #############

class regression():

    @staticmethod
    def regr(importfile):
        df = importfile
        colpred = str(input("nom colonne prediction : "))
        colactu = str(input("nom colonne output : "))

        df['pourcerror'] = (df[f'{colpred}'] - df[f'{colactu}']) / df[f'{colactu}']
        df['errorabs'] = round(abs(df['pourcerror']), 2)
        df.loc[0,'mape'] = round(np.average(df['errorabs'])*100, 2)
        print("MAPE = ",df.loc[0,'mape'])
        return df

    @staticmethod
    def posneg(importfile):
        df = importfile
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
        print(f'On a {erreurpos} erreurs positive, {erreurneg} erreurs negative, {predparfaite} prediction parfaite')
        df.loc[0,'positive erreur'] = erreurpos
        df.loc[0,'negatif erreur'] = erreurneg
        df.loc[0,'prediction parfaite'] = predparfaite 
        return df       

    @staticmethod
    def seuil_vs_error(importfile):
        df = importfile
        seuil = float(input("valeur du seuil :"))

        count_error_min = 0
        count_error_maj = 0
        for i in df["errorabs"]:
            if i < seuil :
                count_error_min +=1
            else :
                count_error_maj +=1

        print(f"il y'a {count_error_min} erreurs inférieurs à {seuil}, il y'a {count_error_maj} erreurs supérieurs à {seuil}")
        p_sous_seuil = (count_error_min*100)/(count_error_min+count_error_maj) 
        p_sous_seuil = round(p_sous_seuil,4)
        print(f"erreur sous seuil = {p_sous_seuil}%")
        return p_sous_seuil

########## Class classification context #############  

class classification():
 
    @staticmethod
    def topErr(importfile):
        df = importfile
        try :
            filtered = df.loc[df['Error'].isin(["FN","FP"])]
        except :
            print("Erreur, la colonne 'Error' est surement absente du DataFrame")
            return
        print("Extraction commencé")
        filemane = str(input("Nom du fichier avec extention :"))
        print("Extraction commencé")
        filtered.to_csv(filemane, index=False)
        print("Extraction terminé")
        return filtered
        # filtered = (filtered.nlargest(100,('0 probability'))) #nsmalest
    
    @staticmethod
    def auc(importfile):
        df = importfile
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

        # ln1=df_val['target'].sum(axis =0)
        # df1 = df_val['1 probability'] * df_val['target']

        # df0 = df_val['target'] + 1
        # df0 = df0.apply(lambda x: 0 if x == 2 else 1)
        # ln2=df0.sum(axis =0)
        # df0 = df_val['1 probability'] * df0

        # nb1=0
        # for r1 in df1:
        #     if r1 > 0:
        #         for r2 in df0:
        #             if r1 > r2 and r2!=0:
        #                 nb1 = nb1 + 1
        # print("AUC= ", nb1/(ln1*ln2))
   
    @staticmethod    
    def matrix(importfile,coloutput,colpred):
        df = importfile
        df['Error'] =df.apply(matrice,axis=1,args=(coloutput,colpred))
        df[f'{colpred}'] = df[f'{colpred}'].map({1:2, 0:0})
        mat = df[f'{colpred}'] - df[f'{coloutput}']

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
        return df
 
    @staticmethod
    def matcout(importfile,coutTP,coutFN,coutTN,coutFP):
        df = importfile
        df.loc[0,'cout tp'] =  round(df.loc[0,'truePositif'] * coutTP,2) 
        df.loc[0,'cout fn'] =  round(df.loc[0,'falseNegative'] * coutFN,2)
        df.loc[0,'cout tn'] =  round(df.loc[0,'trueNegative'] * coutTN,2)
        df.loc[0,'cout fp'] =  round(df.loc[0,'falsePositive'] * coutFP,2)
        res = (df['cout tp']) + (df['cout fn']) + (df['cout tn']) + (df['cout fp'])
        df['resultat'] = res
        resret = df.loc[0,'resultat']
        return resret
  

class seuilOptimal() :
    
    @staticmethod
    def seuil(importfile):
        df = importfile
        me = classification()
        coloutput = str(input("nom colonne output : "))
        countSeuil = 0
        bestResult = 0
        newResult = 0
        varSeuil = 0.0
        optiSeuil = 0.1

        coutTP = float(input("saisir cout TP :"))
        coutFN = float(input("saisir cout FN :"))
        coutTN = float(input("saisir cout TN :"))
        coutFP = float(input("saisir cout FP :"))
        modGraph = bool(input("Voulez vous un graphique du seuil ?"))
        print(modGraph)

        inpSeuilMini = float(input("saisir le seuil de depart : "))
        inpSeuilMax = float(input("saisir le seuil de fin : "))
        inpSeuilParse = float(input("saisir le seuil d'analyse 0.1 ou 0.05 ou ... : "))
        varSeuil = inpSeuilMini
        
        if modGraph :
            tablx = []
            tably = []

        while varSeuil <= inpSeuilMax:
            print("")
            print("count : ",countSeuil)
            varSeuil =round(varSeuil,2)
            print("valeur seuil : ",varSeuil)
        
            s_0 = []
            for i in df['1 probability'] :
                if i < varSeuil :
                    s_0.append(0) #df.loc[countline,'seuil_pred'] = 0
                else:
                    s_0.append(1) #df.loc[countline,'seuil_pred']= 1
        

            df['seuil_pred'] = s_0
            colpred = 'seuil_pred'
            me.matrix(df,coloutput,colpred)
            newResult = me.matcout(df,coutTP,coutFN,coutTN,coutFP)
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
            varSeuil += inpSeuilParse

        print("finally") 

        s_0 = []
        for i in df['1 probability'] :
            if i < optiSeuil :
                s_0.append(0) #df.loc[countline,'seuil_pred'] = 0
            else:
                s_0.append(1) #df.loc[countline,'seuil_pred']= 1     
        
        df['seuil_pred'] = s_0
        colpred = 'seuil_pred'
        me.matrix(df,coloutput,colpred)
        me.matcout(df,coutTP,coutFN,coutTN,coutFP)
        df.loc[0,"seuil_opti"] = optiSeuil
        df['Error']=df.apply(matrice,axis=1,args=(coloutput,colpred))
        
        print(f"Matrice de couts généré, resultat = {df.loc[0,'resultat']}")
        print(f"Le meilleur résultat est de {bestResult}, avec un seuil opti de {optiSeuil}")    
        
        if modGraph :
            graph_simple(tablx,tably,'Threshold evaluation','graphSeuilOpti')

######### safe save system ##########

class save():

    @staticmethod
    def extraction(importfile) :
        df = importfile
        varName = str(input("nom du fichier.ex : "))
        print("Extraction commencé")

        isClean = False
        add = 0

        while isClean != True :
            if not os.path.isfile(varName) :
                df.to_csv(f"Rapport/{varName}", index=False)
                isClean = True
            else :
                split = varName.split(".")
                part_1 = split[0]+"_"+str(add)
                varName = ".".join([part_1,split[1]])
                add +=1  
        
        print("Extraction terminé")


## End         

