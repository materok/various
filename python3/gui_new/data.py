import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from math import ceil
import warnings
from utils import *

class data:
    first_year=-1
    last_year=-1
    drawYears=[]
    def __init__(self,first_year,last_year):
        print("ensure there is more than 1 entry in each file")
        self.first_year=first_year
        self.last_year=last_year
        for year in range(first_year, last_year+1):
            filename="../../dataLight"+str(year-2000)+".txt"
            self.loadRunningData(filename, year)
            filename="../../stats"+str(year-2000)+".txt"
            self.loadWeightData(filename, year)
            if year>=2017: self.loadCompositionData(filename, year)
            filename="../../mood"+str(year-2000)+".txt"
            if year>=2019: self.loadMoodData(filename, year)
            filename="../../ubung"+str(year-2000)+".txt"
            if year>=2019: self.loadTrainingData(filename, year)
            filename="../../sleep"+str(year-2000)+".txt"
            if year>=2019: self.loadSleepData(filename, year)
            filename="../../restingHeartrate"+str(year-2000)+".txt"
            if year>=2019: self.loadHRData(filename, year)
    weight={}
    def loadWeightData(self, filename, year=2019):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if year==2016:
                gew, day = np.genfromtxt(filename, missing_values=",", filling_values = -1,
                                        usecols=(0,1), unpack=True, invalid_raise=False)
                self.weight[year]={}
                self.weight[year]["weight"]=gew
                self.weight[year]["day"]=day-1
                self.weight[year]["bin"]=day-1
            else:
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
            day,month,fat,water,muscle,bone= np.genfromtxt(filename, missing_values=",", filling_values = -1,
                                                           usecols=(1,2,3,4,5,6), unpack=True, invalid_raise=False)
            self.bodycomposition[year]={}
            fillEmpty(day)
            fillEmpty(month)
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
        if year==2016:
            t_no_formatting,dist,bpm,bpm_max,day= np.genfromtxt(filename, unpack=True)
            t_for_velo,t = tConvert(t_no_formatting)
            vel=calcVelo(dist, t_for_velo)
        if year==2017:
            mins,sec,dist,bpm,bpm_max,day,month= np.genfromtxt(filename, missing_values=",", filling_values = -1, unpack=True)
            t=mins+sec/60.
            vel=calcVelo(dist,t/60)
        if year>=2018:
            mins,sec,dist,bpm,bpm_max,altitude,day,month= np.genfromtxt(filename, missing_values=",", filling_values = -1, unpack=True)
            t=mins+sec/60.
            vel=calcVelo(dist,t/60)
        if year!=2016:
            dist/=10.
            vel/=10.
        self.run[year]={}
        self.run[year]["time"]=t
        self.run[year]["distance"]=dist
        self.run[year]["velocity"]=vel
        self.run[year]["bpm"]=bpm
        self.run[year]["maxbpm"]=bpm_max
        if altitude!=[]:
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
    mood={}
    def loadMoodData(self, filename, year=2019):
        day, month, average, sad, boring, angry, active, happy, unsure, sick= np.genfromtxt(filename, unpack=True)
        self.mood[year]={}
        self.mood[year]["day"]=day.astype("int")
        self.mood[year]["month"]=month.astype("int")
        self.mood[year]["average"]=average.astype("bool")
        self.mood[year]["sad"]=sad.astype("bool")
        self.mood[year]["boring"]=boring.astype("bool")
        self.mood[year]["angry"]=angry.astype("bool")
        self.mood[year]["active"]=active.astype("bool")
        self.mood[year]["happy"]=happy.astype("bool")
        self.mood[year]["unsure"]=unsure.astype("bool")
        self.mood[year]["sick"]=sick.astype("bool")
        colorcode=[]
        moodDict=self.mood[year]
        for i in range(len(day)):
            colorlist=np.array([],"i")
            if moodDict["average"][i]: colorlist=np.append(colorlist,0)
            if moodDict["sad"][i]: colorlist=np.append(colorlist,1)
            if moodDict["boring"][i]: colorlist=np.append(colorlist,2)
            if moodDict["angry"][i]: colorlist=np.append(colorlist,3)
            if moodDict["active"][i]: colorlist=np.append(colorlist,4)
            if moodDict["happy"][i]: colorlist=np.append(colorlist,5)
            if moodDict["unsure"][i]: colorlist=np.append(colorlist,6)
            if moodDict["sick"][i]: colorlist=np.append(colorlist,7)
            colorcode.append(colorlist)
        self.mood[year]["colorcode"]=colorcode
    training={}
    def loadTrainingData(self, filename, year=2019):
        day, month, pushup= np.genfromtxt(filename, unpack=True)
        self.training[year]={}
        fillEmpty(day)
        fillEmpty(month)
        self.training[year]["day"]=day
        self.training[year]["month"]=month
        bins=dayAndMonthToBin(day,month,year)
        self.training[year]["bin"]=bins
        self.training[year]["pushup"]=pushup.astype("int")
    sleep={}
    def loadSleepData(self, filename, year=2019):
        day, month, toBedH, toBedM, awakeH, awakeM, outBedH, outBedM= np.genfromtxt(filename, unpack=True)
        self.sleep[year]={}
        fillEmpty(day)
        fillEmpty(month)
        self.sleep[year]["day"]=day
        self.sleep[year]["month"]=month
        bins=dayAndMonthToBin(day,month,year)
        self.sleep[year]["bin"]=bins
        self.sleep[year]["toBed"]=toBedH.astype("int")+toBedM.astype("float")/60.
        self.sleep[year]["awake"]=awakeH.astype("int")+awakeM.astype("float")/60.
        self.sleep[year]["outBed"]=outBedH.astype("int")+outBedM.astype("float")/60.
        sleeping=self.sleep[year]["awake"]-self.sleep[year]["toBed"]
        for i in range(len(sleeping)):
            if sleeping[i] <0: sleeping[i] +=24
        self.sleep[year]["sleeping"]=sleeping
        lying=self.sleep[year]["outBed"]-self.sleep[year]["awake"]
        for i in range(len(lying)):
            if lying[i] <0: lying[i] +=24
        self.sleep[year]["lying"]=lying
        total=self.sleep[year]["outBed"]-self.sleep[year]["toBed"]
        for i in range(len(total)):
            if total[i] <0: total[i] +=24
        self.sleep[year]["total"]=total
    HR={}
    def loadHRData(self, filename, year=2019):
        rhr, day, month= np.genfromtxt(filename, missing_values=",", filling_values = -1, unpack=True)
        self.HR[year]={}
        fillEmpty(day)
        fillEmpty(month)
        self.HR[year]["day"]=day
        self.HR[year]["month"]=month
        bins=dayAndMonthToBin(day,month,year)
        self.HR[year]["bin"]=bins
        self.HR[year]["rhr"]=rhr

    def makePlots(self):
        year=2019
        where="%i/"%(year)
        for year in self.drawYears:
            print("starting with year:",year)
            where="%i/"%(year)
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
            MakeKMH_BPMPlot(rbins,vel,bpm,where, savepng=True)
            MakeKMH_BPMPlot(rbins,vel,maxbpm,where, savepng=True, maxBPM=True)
            MakeBPMPlots(rbins,bpm,where, year=year)
            MakeBPMPlots(rbins,maxbpm,where, option="max",year=year)
            MakeCumulPlot(rbins,dist,year,where)
            MakeKMPlots(rbins,time,vel,year,where)
            plotTimePerKM(vel,rbins,year,where)
            plotVel(vel,rbins,year,where)
            plotVel(vel,rbins,year,where)
            plotLBLequiv(vel,rbins,year,where)
            if year>2016:
                cbins=self.bodycomposition[year]["bin"]
                water=self.bodycomposition[year]["water"]
                muscle=self.bodycomposition[year]["muscle"]
                fat=self.bodycomposition[year]["fat"]
                bone=self.bodycomposition[year]["bone"]
                MakeComposition(cbins,fat,water,muscle,bone,year,where,savepng=True)
            if year>2017:
                altitude=self.run[year]["altitude"]
                plotVeloHeight(altitude,vel,where)
                plotBPMHeight(altitude,bpm,where)
                plotBPMHeight(altitude,maxbpm,where,option="max")
            if year>2018:
                makeMoodPlot(self,year)
                PlotPushups(self,year)
                PlotSleep(self,year)
                PlotRHR(self,year)
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


if __name__=="__main__":
    pass
