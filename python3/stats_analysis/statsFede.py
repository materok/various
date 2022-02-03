import numpy as np
from helper.laufen import MakeStats,MakeStats17,fillEmpty

def main(version=18):

    if version==16:
        t5,km5,bpm,bpm_max,day= np.genfromtxt('../../dataLight.txt', unpack=True)
        gew,day1= np.genfromtxt('../../stats.txt', unpack=True)
        MakeStats(day1,gew,2016,day,show=True,savepng=True)
    elif version==17:
        mins,sec,km5,bpm,bpm_max,day,month= np.genfromtxt('../../dataLight17.txt',missing_values=",", filling_values = -1, unpack=True)
        t5 =mins+sec/60.
        km5/=10.
        fillEmpty(sec)
        t5 =mins+sec/60.
        fillEmpty(month)
        gew17,day17,month17= np.genfromtxt('../../statsFede.txt',missing_values=",", filling_values = -1, unpack=True)
        gew17/=10.
        fillEmpty(month17)
        #MakeStats17(day17,month17,gew17,2017,show=True,Fede=True)
        MakeStats17(day17,month17,gew17,2017,Fede=True)
    elif version==18:
        #mins,sec,km5,bpm,bpm_max,day,month= np.genfromtxt('../../dataLight17.txt',missing_values=",", filling_values = -1, unpack=True)
        #t5 =mins+sec/60.
        #km5/=10.
        #fillEmpty(sec)
        #t5 =mins+sec/60.
        #fillEmpty(month)
        gew17,day17,month17= np.genfromtxt('../../statsFede18.txt',missing_values=",", filling_values = -1, unpack=True)
        gew17/=10.
        fillEmpty(month17)
        #MakeStats17(day17,month17,gew17,2017,show=True,Fede=True)
        MakeStats17(day17,month17,gew17,2017,Fede=True,savepng=True)

if __name__=="__main__":
    #main(16)
    main(17)
    main()
