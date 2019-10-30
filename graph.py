from pandas import read_csv
import matplotlib.pyplot as plt
import sys
import argparse
import numpy as np
import csv

parser = argparse.ArgumentParser(description="Process graph pred")

parser.add_argument('context', choices=['courbe','diagr'], help='context dans lequel nous allons travailler')

subparsers = parser.add_subparsers(dest='Action',required=True)

parser_simple = subparsers.add_parser("simple")
parser_simple.add_argument('--file')

parser_double = subparsers.add_parser("double")
parser_double.add_argument('--file')
parser_double.add_argument('--file2')

args = parser.parse_args() 
print(args)


def graph_simple(importfile):
    df = read_csv(f'{importfile}')
    x = df['varSeuil']
    y = df['newResult']
    plt.plot(x,y)
    plt.xlabel("Threshold")
    plt.ylabel("C.A")
    plt.title("Threshold evaluation")
    plt.draw()
    plt.savefig('Threshold.png', dpi=200) 


def graph_double(importfile,importfile2):

    df = read_csv(f'{importfile}')
    df2 = read_csv(f'{importfile2}')
    x = df['varSeuil']
    y = df['newResult']
    y2 = df2['newResult']
    plt.plot(x, y, "r--", label="Ensemble")
    plt.plot(x, y2, "b:o", label="Deep")
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylabel("C.A")
    plt.title("Threshold evaluation")
    plt.draw()
    plt.savefig('ThresholdCompar.png', dpi=200)


def diagr_simple(importfile):

    df = read_csv(f'{importfile}')
    bars = ('TP','FN','TN','FP',)
                                                                                
    v1 = df.loc[0,'truePositif']
    v2 = df.loc[0,'falseNegative']
    v3 = df.loc[0,'trueNegative']
    v4 = df.loc[0,'falsePositive']

    height = [v1,v2,v3,v4]

    plt.bar(bars, height)
    plt.legend()

    plt.title("Positif Negatif evaluation")
    plt.draw()
    plt.savefig('posnegCompar.png', dpi=200)



def diagr_double(importfile,importfile2):

    df = read_csv(f'{importfile}')
    df2 = read_csv(f'{importfile2}')

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

    plt.bar(bars, height, label="Ensemble")
    plt.bar(bars, height2, label="deep")
    plt.legend()

    plt.title("Positif Negatif evaluation")
    plt.draw()
    plt.savefig('posnegCompar.png', dpi=200)



def voidUpdate() :
    if args.context == 'courbe':
        if args.Action == 'simple':
            graph_simple(args.file)
        elif args.Action == 'double':  
            graph_double(args.file,args.file2)  

    if args.context == 'diagr':
        if args.Action == 'simple':
            diagr_simple(args.file)   
        elif args.Action == 'double':
            diagr_double(args.file,args.file2)   

voidUpdate()

print("Programme terminé !!")


    # If we have long labels, we cannot see it properly
    # names = ("very long group name 1","very long group name 2","very long group name 3","very long group name 4","very long group name 5")
    # plt.xticks(xax, names, rotation=90)
    
    # Thus we have to give more margin:
    # plt.subplots_adjust(bottom=0.4)
    
    # It's the same concept if you need more space for your titles
    # plt.title("This is\na very very\nloooooong\ntitle!")
    # plt.subplots_adjust(top=0.7)