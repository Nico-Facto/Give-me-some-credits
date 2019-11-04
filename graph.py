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
    def diagr_simple(importfile):

        df = read_csv(f'{importfile}')
        bars = ('TP','FN','TN','FP',)

        v1 = df.loc[0,'truePositif']
        v2 = df.loc[0,'falseNegative']
        v3 = df.loc[0,'trueNegative']
        v4 = df.loc[0,'falsePositive']

        height = [v1,v2,v3,v4]

        plt.grid(True)
        plt.bar(bars, height)
        plt.legend()

        plt.title("Positif Negatif evaluation")
        plt.draw()
       
    @staticmethod 
    def diagr_double(importfile,importfile2):

        df = importfile
        df2 = importfile2

        bars = ('TP','FN','TN','FP')
                                            
        v1 = df.loc[0,'truePositif']
        v2 = df.loc[0,'falseNegative']
        v3 = df.loc[0,'trueNegative']
        v4 = df.loc[0,'falsePositive']

        v5 = df2.loc[0,'truePositif']
        v6 = df2.loc[0,'falseNegative']
        v7 = df2.loc[0,'trueNegative']
        v8 = df2.loc[0,'falsePositive']

        height = [v1,v2,v3,v4]
        height2 = [v5,v6,v7,v8]

        plt.grid(True)
        plt.bar(bars, height, label="Ensemble")
        plt.bar(bars, height2, label="deep")
        plt.legend()

        plt.title("Positif Negatif evaluation")
        plt.draw()
        
    @staticmethod 
    def extraction():
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