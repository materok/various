import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from math import ceil
import warnings
from helper.utils import *
from os import path
from os import makedirs

class garminData:
    first_year=-1
    last_year=-1
    drawYears=[]
    def __init__(self,first_year,last_year,short):
        print("ensure there is more than 1 entry in each file")
        self.first_year=first_year
        self.last_year=last_year
        self.short=short
        for year in range(first_year, last_year+1):
            filename=f"../../data/dataFromGarmin{year}.txt"
            if path.exists(filename):
                self.loadRunningData(filename, year)
            filename=f"../../data/statsFromGarmin{year}.txt"
            if path.exists(filename):
                self.loadWeightData(filename, year)
            print("todo: self.loadCompositionData(filename, year)")
            filename=f"../../data/gym{year-2000}.txt"
            if path.exists(filename) and year>=2023: self.loadGymData(filename, year)
            if not short:
                filename=f"../../data/hrDataFromGarmin{year}.txt"
                if path.exists(filename):
                    self.loadHRData(filename, year)
    weight={}
    def loadWeightData(self, filename, year=2019):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gew, day, month = np.genfromtxt(filename, missing_values=",", filling_values = -1,
                                            usecols=(0,1,2), unpack=True, invalid_raise=False)
            self.weight[year]={}
            self.weight[year]["weight"]=gew/10
            fillEmpty(day)
            fillEmpty(month)
            self.weight[year]["day"]=day
            self.weight[year]["month"]=month
            bins=dayAndMonthToBin(day,month,year)
            self.weight[year]["bin"]=bins
    bodycomposition={}
    def loadCompositionData(self, filename, year=2019):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bodycomposition[year]={}
            try:
                day,month,fat,water,muscle,bone= np.genfromtxt(filename, missing_values=",", filling_values = -1,
                                                               usecols=(1,2,3,4,5,6), unpack=True, invalid_raise=False)
            except:
                return
            try:
                fillEmpty(day)
                fillEmpty(month)
            except:
                return
            self.bodycomposition[year]["day"]=day
            self.bodycomposition[year]["month"]=month
            bins=dayAndMonthToBin(day,month,year)
            self.bodycomposition[year]["bin"]=bins
            self.bodycomposition[year]["water"]=water/10.
            self.bodycomposition[year]["muscle"]=muscle/10.
            self.bodycomposition[year]["fat"]=fat/10.
            self.bodycomposition[year]["bone"]=bone/10.
    run={}
    def loadRunningData(self, filename, year=2019):
        t, dist, vel, altitude = [], [], [], []
        bpm, bpm_max=[], []
        day, month = [], []
        route = []
        mins,sec,dist, bpm, bpm_max,altitude,day,month,vo2max= np.genfromtxt(filename, missing_values=",", filling_values = -1, unpack=True)
        t=mins+sec/60.
        dist/=1000
        vel=calcVelo(dist,t/60)
        self.run[year]={}
        self.run[year]["time"]=t
        self.run[year]["distance"]=dist
        self.run[year]["velocity"]=vel
        self.run[year]["bpm"]=bpm
        self.run[year]["maxbpm"]=bpm_max
        self.run[year]["vo2max"]=vo2max
        self.run[year]["altitude"]=altitude
        fillEmpty(month)
        if year==2016:
            bins=day
            self.run[year]["bin"]=bins
            self.run[year]["day"]=[dayToMonth(i,year)[0] for i in bins]
            self.run[year]["month"]=[dayToMonth(i,year)[1] for i in bins]
        else:
            bins=dayAndMonthToBin(day,month,year)
            self.run[year]["bin"]=bins
            self.run[year]["day"]=day
            self.run[year]["month"]=month

    HR={}
    def loadHRData(self, filename, year=2019):
        rhr, minHR, maxHR, day, month= np.genfromtxt(filename, missing_values=",", filling_values = -1, unpack=True)
        self.HR[year]={}
        fillEmpty(day)
        fillEmpty(month)
        self.HR[year]["day"]=day
        self.HR[year]["month"]=month
        bins=dayAndMonthToBin(day,month,year)
        self.HR[year]["bin"]=bins
        self.HR[year]["rhr"]=rhr
        self.HR[year]["minHR"]=minHR
        self.HR[year]["maxHR"]=maxHR

    gym={}
    def loadGymData(self, filename, year=2019):
        day, month, boolean= np.genfromtxt(filename, missing_values=",", filling_values = -1, unpack=True)
        self.gym[year]={}
        fillEmpty(day)
        fillEmpty(month)
        self.gym[year]["day"]=day
        self.gym[year]["month"]=month
        bins=dayAndMonthToBin(day,month,year)
        self.gym[year]["bin"]=bins
        self.gym[year]["boolean"]=boolean

    def makePlots(self):
        for year in self.drawYears:
            print("starting with year:",year)
            where="%i/"%(year)
            if not path.isdir("../../plots/"+where): makedirs("../../plots/"+where)
            wbins=self.weight[year]["bin"]
            weights=self.weight[year]["weight"]
            MakeDeltaPlot(wbins,weights,year,where,savepng=True)
            rbins=self.run[year]["bin"]
            MakeStats(self, year)
            time=self.run[year]["time"]
            vel=self.run[year]["velocity"]
            dist=self.run[year]["distance"]
            bpm=self.run[year]["bpm"]
            maxbpm=self.run[year]["maxbpm"]
            MakeKMHPlot(rbins,vel,dist,where, savepng=True)
            MakeKMHPlot(rbins,3600./vel,dist,where, savepng=True,timeBool=True)
            try:
                #MakeKMH_BPMPlot(rbins,vel,bpm,where, savepng=True)
                #MakeKMH_BPMPlot(rbins,vel,maxbpm,where, savepng=True, maxBPM=True)
                MakeKMH_BPMPlot(vel,bpm,where, savepng=True)
                MakeKMH_BPMPlot(vel,maxbpm,where, savepng=True, maxBPM=True)
                MakeBPMPlots(rbins,bpm,where, year=year)
                MakeBPMPlots(rbins,maxbpm,where, option="max",year=year)
            except:
                pass
            MakeCumulPlot(rbins,dist,year,where)
            MakeKMPlots(rbins,time,vel,year,where)
            plotTimePerKM(vel,rbins,year,where)
            plotVel(vel,rbins,year,where)
            plotVel(vel,rbins,year,where)
            plotLBLequiv(vel,rbins,year,where)
            #if year>2016:
                #cbins=self.bodycomposition[year]["bin"]
                #water=self.bodycomposition[year]["water"]
                #muscle=self.bodycomposition[year]["muscle"]
                #fat=self.bodycomposition[year]["fat"]
                #bone=self.bodycomposition[year]["bone"]
                #try:
                    #MakeComposition(cbins,fat,water,muscle,bone,year,where,savepng=True)
                #except:
                    #print("didn't plot composition, too few entries")
            try:
                altitude=self.run[year]["altitude"]
                plotVeloHeight(altitude,vel,where)
                plotBPMHeight(altitude,bpm,where)
                plotBPMHeight(altitude,maxbpm,where,option="max")
            except:
                pass
            try:
                PlotRHR(self,year)
            except:
                    pass
            MakeGymPlot(self,year)
            MakeMonthlyGymPlot(self,year)
            MakeMonthCumulPlot(self,onlyYear=year)
        MakeCombinedStats(self)
        MakeLongStatsPlot(self)
        rbins=np.array([])
        vel=np.array([])
        dist=np.array([])
        for year in range(self.first_year,self.last_year+1):
            if year not in self.weight:
                continue
            else:
                rbins=np.append(rbins,self.run[year]["bin"])
                vel=np.append(vel,self.run[year]["velocity"])
                dist=np.append(dist,self.run[year]["distance"])
        MakeKMHPlot(rbins,vel,dist,"",savepng=True)
        MakeKMHPlot(rbins,3600./vel,dist,"", savepng=True,timeBool=True)
        MakeLongCumulPlot(self)
        MakeMonthCumulPlot(self)
        MakeLongRHRPlot(self)
        self.makeRoutePlots()

    def makePlotsShort(self):
        for year in self.drawYears:
            print("starting with year:",year)
            where="%i/"%(year)
            if not path.isdir("../../plots/"+where): makedirs("../../plots/"+where)
            rbins=self.run[year]["bin"]
            MakeStats(self, year)
            time=self.run[year]["time"]
            vel=self.run[year]["velocity"]
            dist=self.run[year]["distance"]
            MakeKMHPlot(rbins,vel,dist,where, savepng=True)
            MakeKMHPlot(rbins,3600./vel,dist,where, savepng=True,timeBool=True)
            MakeCumulPlot(rbins,dist,year,where)
            plotTimePerKM(vel,rbins,year,where)
            MakeMonthCumulPlot(self,onlyYear=year)
            MakeGymPlot(self,year)
            MakeMonthlyGymPlot(self,year)
        MakeCombinedStats(self)
        MakeLongStatsPlot(self)
        rbins=np.array([])
        vel=np.array([])
        dist=np.array([])
        for year in range(self.first_year,self.last_year+1):
            if year not in self.weight:
                continue
            else:
                rbins=np.append(rbins,self.run[year]["bin"])
                vel=np.append(vel,self.run[year]["velocity"])
                dist=np.append(dist,self.run[year]["distance"])
        MakeKMHPlot(rbins,vel,dist,"",savepng=True)
        MakeKMHPlot(rbins,3600./vel,dist,"", savepng=True,timeBool=True)
        MakeLongCumulPlot(self)
        MakeMonthCumulPlot(self)

if __name__=="__main__":
    pass
