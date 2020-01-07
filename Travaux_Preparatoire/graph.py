'''Visual module for Data/Ia Dev Script -- Nicolas Autexier -- contact = nicolas.atx@gmx.fr '''

from pandas import read_csv
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
import os


class rViz():

    @staticmethod
    def graph_simple(x,y):

        labelx = str(input("label axe X : "))
        labely = str(input("label axe y : "))
        labeltitle = str(input("Graph title : "))

        plt.grid(True)
        plt.plot(x,y, "r:o")
        plt.xlabel(f"{labelx}")
        plt.ylabel(f"{labely}")
        plt.title(f"{labeltitle}")
        plt.draw()

    @staticmethod     
    def graph_double(x,y,y2):

        labelx = str(input("label axe X : "))
        labely = str(input("label axe y : "))
        courbe1 = str(input("label plot 1 : "))
        courbe2 = str(input("label plot 2 : "))
        labeltitle = str(input("Graph title : "))

        plt.grid(True)
        plt.plot(x, y, "r:o", label=f"{courbe1}")
        plt.plot(x, y2, "b:o", label=f"{courbe2}")
        plt.legend()
        plt.xlabel(f"{labelx}")
        plt.ylabel(f"{labely}")
        plt.title(f"{labeltitle}")
        plt.draw()

    @staticmethod 
    def diagr_matrice_simple(importfile,TypeCA):

        df = read_csv(f'{importfile}')
        bars = ('TP','FN','TN','FP',)

        if not TypeCA :
            v1 = df.loc[0,'truePositif']
            v2 = df.loc[0,'falseNegative']
            v3 = df.loc[0,'trueNegative']
            v4 = df.loc[0,'falsePositive']
        else : 
            v1 = df.loc[0,'cout tp']
            v2 = df.loc[0,'cout fn']
            v3 = df.loc[0,'cout tn']
            v4 = df.loc[0,'cout fp']    


        height = [v1,v2,v3,v4]

        plt.grid(True)
        plt.bar(bars, height)
        plt.legend()

        plt.title("Positif Negatif evaluation")
        plt.draw()
       
    @staticmethod 
    def diagr_matrice_double(importfile,importfile2,TypeCA):

        df = importfile
        df2 = importfile2
        mod1 = str(input('nom valeur 1 : '))
        mod2 = str(input('nom valeur 2 : '))
        bars = ('TP','FN','TN','FP')

        if not TypeCA :
            v1 = df.loc[0,'truePositif']
            v2 = df.loc[0,'falseNegative']
            v3 = df.loc[0,'trueNegative']
            v4 = df.loc[0,'falsePositive']
            v5 = df2.loc[0,'truePositif']
            v6 = df2.loc[0,'falseNegative']
            v7 = df2.loc[0,'trueNegative']
            v8 = df2.loc[0,'falsePositive']
        else : 
            v1 = df.loc[0,'cout tp']
            v2 = df.loc[0,'cout fn']
            v3 = df.loc[0,'cout tn']
            v4 = df.loc[0,'cout fp']
            v5 = df2.loc[0,'cout tp']
            v6 = df2.loc[0,'cout fn']
            v7 = df2.loc[0,'cout tn']
            v8 = df2.loc[0,'cout fp']
                                            

        height = [v1,v2,v3,v4]
        height2 = [v5,v6,v7,v8]

        plt.grid(True)
        plt.bar(bars, height, label=f'{mod1}')
        plt.bar(bars, height2, label=f'{mod2}')
        plt.legend()

        plt.title("Positif Negatif evaluation")
        plt.draw()
        plt.show()
    
    @staticmethod
    def stack_plot(importfile,nombreCourbe) :
        df = importfile
        plt.style.use("fivethirtyeight")## use(str,dict,list)
        
        if nombreCourbe < 0 or nombreCourbe >4 :
            print("Nombre de colonne non pris en charge par la fonction  >1-4<")
        else:
            if nombreCourbe <2 :    
                axe_x = str(input('valeur x : '))
                xaxes = df[f'{axe_x}']
            if nombreCourbe <3 :    
                c1 = str(input('courbe 1 : '))
                valvar = df[f'{c1}']
            if nombreCourbe <4 : 
                c2 = str(input('courbe 2 : '))
                valvar1 = df[f'{c2}']
            if nombreCourbe <5 :    
                c3 = str(input('courbe 3 : '))
                valvar2 = df[f'{c3}']

        t_ = str(input('titre : '))
        plt.title(f"{t_}")

        labels = [f'{c1}',f'{c2}',f'{c3}']
        colors = ['#6d904f','#fc4f30', '#e5ae37']#bleu #008fd5
        plt.legend()
        plt.stackplot(xaxes, valvar, valvar1, valvar2, labels=labels, colors=colors)

        plt.draw()
        plt.show()


    @staticmethod 
    def extraction():
        print("Extraction commencé")

        isClean = False
        varName = "graph.png"
        add = 0

        while isClean != True :
            if not os.path.isfile(varName) :
                plt.savefig(f"Rapport/{varName}", dpi=200)
                isClean = True
            else :
                split = varName.split(".")
                part_1 = split[0]+"_"+str(add)
                varName = ".".join([part_1,split[1]])
                add +=1  
        
        print("Extraction terminé")
        plt.show()
    
    
    # r-- // r:o where r is variable color

    # If we have long labels, we cannot see it properly
    # names = ("very long group name 1","very long group name 2","very long group name 3","very long group name 4","very long group name 5")
    # plt.xticks(xax, names, rotation=90)
    
    # Thus we have to give more margin:
    # plt.subplots_adjust(bottom=0.4)
    
    # It's the same concept if you need more space for your titles
    # plt.title("This is\na very very\nloooooong\ntitle!")
    # plt.subplots_adjust(top=0.7)