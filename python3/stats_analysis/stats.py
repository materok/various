import numpy as np
from helper.laufen import MakeStats,MakeStats17,fillEmpty

def main(version=17):

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
        gew17,day17,month17= np.genfromtxt('../../stats17.txt',missing_values=",", filling_values = -1, unpack=True)
        gew17p2,day17p2,month17p2,fett,wasser,muskel,knochen= np.genfromtxt('../../stats17p2.txt',missing_values=",", filling_values = -1, unpack=True)
        gew17=np.append(gew17,gew17p2)
        gew17/=10.
        fett/=10.
        wasser/=10.
        muskel/=10.
        knochen/=10.
        day17=np.append(day17,day17p2)
        month17=np.append(month17,month17p2)
        fillEmpty(month17)
        fillEmpty(month17p2)
        MakeStats17(day17,month17,gew17,2017,show=True)

if __name__=="__main__":
    #main(16)
    main()
