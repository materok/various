import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from math import ceil
from laufen import dayAndMonthToBin, convertDayToBin, SavePlot

def GetSleepTimes(day,month,year,hBed,mBed,hAlarm,mAlarm,hUp,mUp,show=False,savepng=False):

    x = dayAndMonthToBin(day,month,year)

    possibleLabels = ['January', 'Febuary', 'March', 'April',
                      'May','June','July','August','September',
                      'Oktober','November', 'December']
    labels=[]
    x_ticks=[]
    for i in range(len(month)):
        if (month[i]==month[i-1]) and i != 0:
            labels.append('')
        else:
            labels.append(possibleLabels[int(month[i])-1])
            x_ticks.append(day[i])

    for i in np.where(hBed>20):
        hBed[i]-=24
    timeUntilAlarm=(hAlarm+mAlarm/60.)-(hBed+mBed/60.)
    delay=(hUp+mUp/60.)-(hAlarm+mAlarm/60.)

    plt.figure(figsize=(10,10))
    plt.plot(x, timeUntilAlarm, linestyle="",marker="x")
    plt.xlabel("month")
    plt.ylabel("time until alarm")
    plt.xticks(x, labels, rotation='vertical')
    SavePlot(x,year,"timeUntilAlarm",savepng)
    plt.figure(figsize=(10,10))
    plt.plot(x, delay, linestyle="",marker="x")
    plt.xlabel("month")
    plt.ylabel("delay")
    plt.xticks(x, labels, rotation='vertical')
    SavePlot(x,year,"timeDelay",savepng)
    plt.figure(figsize=(10,10))
    plt.plot(x, delay+timeUntilAlarm, linestyle="",marker="x")
    plt.xlabel("month")
    plt.ylabel("delay")
    plt.xticks(x, labels, rotation='vertical')
    SavePlot(x,year,"timeSlept",savepng)

if __name__=="__main__":
    pass
