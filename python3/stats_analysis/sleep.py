import numpy as np
from helper.sleep import GetSleepTimes
from helper.laufen import fillEmpty

def main(version=17):

    if version==17:
        year=17
        day,month,hBed,mBed,hAlarm,mAlarm,hUp,mUp= np.genfromtxt('../../data/sleep.txt',missing_values=",", filling_values = -1, unpack=True)
        fillEmpty(month)
        #print(day,month)
        GetSleepTimes(day,month,year,hBed,mBed,hAlarm,mAlarm,hUp,mUp,savepng=True)

if __name__=="__main__":
    #main(16)
    main()
