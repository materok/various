import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from math import ceil
from helper.utils import dayAndMonthToBin, convertDayToBin, SavePlot


def CheckSign(money,use):

    posSign=["gehalt","einzahlung","gutschrift"]
    for i in range(len(money)):
        if use[i] not in posSign:
            money[i]*=-1


def CheckFinances(day,month,year,money,use,show=False,savepng=False):

    #~ plots:
        #~ *einnahmen vs monat
        #~ *ausgaben vs monat
        #~ *gesamtwert vs monat
        #~ *ausgaben vs tag
        #~ *piechart ausgaben
        #~ *pichart einnahmen
    income = money[np.where(money>0)]
    expense = money[np.where(money<0)]
    x = dayAndMonthToBin(day,month,year)
    xIn = x[np.where(money>0)]
    xEx = x[np.where(money<0)]
    x2 = month
    x2In = month[np.where(money>0)]
    x2Ex = month[np.where(money<0)]
    listIn=np.array([],'d')
    listEx=np.array([],'d')
    possibleLabels = ['January', 'Febuary', 'March', 'April',
                      'May','June','July','August','September',
                      'Oktober','November', 'December']
    labels = []
    x3 = np.unique(x2)
    for i in x3:
        listIn=np.append(listIn,np.sum(income[np.where(x2In==i)]))
        listEx=np.append(listEx,np.sum(expense[np.where(x2Ex==i)]))
        labels.append(possibleLabels[int(i)-1])
    plt.figure(figsize=(10,10))
    plt.plot(x3, listIn, linestyle="",marker="x")
    plt.xlabel("Nummer des Monats")
    plt.xticks(x3, labels, rotation='vertical')
    plt.ylabel("Einkommen in Euro")
    SavePlot("%i/income"%(year),savepng)
    plt.figure(figsize=(10,10))
    plt.plot(x3, abs(listEx), linestyle="",marker="x")
    plt.xlabel("Nummer des Monats")
    plt.xticks(x3, labels, rotation='vertical')
    plt.ylabel("Ausgaben in Euro")
    SavePlot("%i/expense"%(year),savepng)
    plt.figure(figsize=(10,10))
    plt.plot(x3, listIn+listEx, linestyle="",marker="x")
    plt.xlabel("Nummer des Monats")
    plt.xticks(x3, labels, rotation='vertical')
    plt.ylabel("Bilanz in Euro")
    SavePlot("%i/bilanz"%(year),savepng)

    posSign=["gehalt","einzahlung","gutschrift"]
    sizesIn={}
    sizesOut={}
    for i in x3:
        tmpSetIn={}
        tmpSetOut={}
        for j in use:
            if j in posSign: tmpSetIn[j]=0
            else: tmpSetOut[j]=0
        for j in np.where(x2In==i):
            for h in j:
                tmpSetIn[use[np.where(money>0)][h]]+=income[h]
        for j in np.where(x2Ex==i):
            for h in j:
                tmpSetOut[use[np.where(money<0)][h]]+=abs(expense[h])
        sizesIn[i]=tmpSetIn
        sizesOut[i]=tmpSetOut
    for i in x3:
        totalIn = sum(sizesIn[i].values())
        plt.figure(figsize=(10,10))
        plt.pie(list(sizesIn[i].values()), labels=list(sizesIn[i].keys()),
                autopct=lambda p: '{:.1f}'.format(p * totalIn / 100), startangle=90)
        SavePlot("%i/pieIn"%(year)+str(int(i))+"abs",savepng)
        plt.figure(figsize=(10,10))
        plt.pie(list(sizesIn[i].values()), labels=list(sizesIn[i].keys()),
                autopct="%1.1f%%", startangle=90)
        SavePlot("%i/pieIn"%(year)+str(int(i))+"rel",savepng)
        totalEx = sum(sizesOut[i].values())
        plt.figure(figsize=(10,10))
        plt.pie(list(sizesOut[i].values()), labels=list(sizesOut[i].keys()),
                autopct=lambda p: '{:.1f}'.format(p * totalEx / 100), startangle=90)
        SavePlot("%i/pieEx"%(year)+str(int(i))+"abs",savepng)
        plt.figure(figsize=(10,10))
        plt.pie(list(sizesOut[i].values()), labels=list(sizesOut[i].keys()),
                autopct="%1.1f%%", startangle=90)
        SavePlot("%i/pieEx"%(year)+str(int(i))+"rel",savepng)
        categorised=categorise(sizesOut[i])
        plt.figure(figsize=(10,10))
        plt.pie(list(categorised.values()), labels=list(categorised.keys()),
                autopct=lambda p: '{:.1f}'.format(p * totalEx / 100), startangle=90)
        SavePlot("%i/pieEx"%(year)+str(int(i))+"abs_categorised",savepng)
        plt.figure(figsize=(10,10))
        plt.pie(list(categorised.values()), labels=list(categorised.keys()),
                autopct="%1.1f%%", startangle=90)
        SavePlot("%i/pieEx"%(year)+str(int(i))+"rel_categorised",savepng)

def categorise(dict):
    categories={}
    used=[]
    with open("../../data/categories.txt") as f:
        for line in f:
            line = line.replace("\n","")
            items = line.split(" ")
            categories[items[0]]=0
            for item in items[1:]:
                if item in dict.items():
                    categories[items[0]]+=dict[item]
                if item in used:
                    print( item, " is already in used... this is doublecounting" )
                used.append(item)
    for item in dict.keys():
        if item not in used:
            categories[item]=dict[item]
    return categories

if __name__=="__main__":
    pass
