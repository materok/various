import numpy as np
from helper.bilanz import *
from helper.utils import fillEmpty, dayToMonth

def main(version=17):

    if version==17:
        year=2017

        money,use,day17,month17 = np.genfromtxt('../../data/bilanz.txt',missing_values=",", filling_values = -1, unpack=True)
        use = (np.loadtxt("../../data/bilanz.txt",dtype="str"))[:,1]
        fillEmpty(day17)
        fillEmpty(month17)
        #~ print money
        #~ print use
        CheckSign(money,use)
        #~ print money
        CheckFinances(day17,month17,year,money,use,savepng=True)


if __name__=="__main__":
    main()
    #main(16)
