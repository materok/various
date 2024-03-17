from helper.laufen import data
from helper.garminData import garminData
#import numpy as np
from helper.utils import *

#def main(lastyear=2023,short=False):
def main(lastyear=2024,short=True):

    #print("ok")
    #d.drawYears=[lastyear]
    #d.drawYears=[i for i in range(2016,2022)]
    #print max(d.run[2019]["velocity"]), 60./max(d.run[2019]["velocity"])
    #print( max(d.run[lastyear]["velocity"]), "%i:%2.0f"%(int(60./max(d.run[lastyear]["velocity"])),(60./max(d.run[lastyear]["velocity"])-int(60./max(d.run[lastyear]["velocity"])))*60))
    #print( (d.run[lastyear]["velocity"][-1]), "%i:%2.0f"%(int(60./(d.run[lastyear]["velocity"][-1])),(60./(d.run[lastyear]["velocity"][-1])-int(60./(d.run[lastyear]["velocity"][-1])))*60))
    #d=data(2017,2018)
    #print(d.weight)
    #print(d.bodycomposition)
    #print(d.run[2016]["bin"])
    #if short:
        #d.makePlotsShort()
    #else:
        #d.makePlots()
    #d.makeRoutePlots()

    d=data(2016,lastyear,short)
    #d=garminData(2016,lastyear,short)
    d.drawYears=[lastyear]
    #d.drawYears=[i for i in range(2016,2022)]
    print( max(d.run[lastyear]["velocity"]), "%i:%2.0f"%(int(60./max(d.run[lastyear]["velocity"])),(60./max(d.run[lastyear]["velocity"])-int(60./max(d.run[lastyear]["velocity"])))*60))
    print( (d.run[lastyear]["velocity"][-1]), "%i:%2.0f"%(int(60./(d.run[lastyear]["velocity"][-1])),(60./(d.run[lastyear]["velocity"][-1])-int(60./(d.run[lastyear]["velocity"][-1])))*60))
    if short:
        d.makePlotsShort()
    else:
        d.makePlots()

if __name__=="__main__":
    main()
