
def MakeCombinedStats(day,month,y,year,runDay=[],runMonth=[],height=1.70,height_err=0.01,show=False,savepng=False):

    plt.figure(figsize=(20,10))
    xRun=np.zeros(len(y))
    x=dayAndMonthToBin(day,month,year)
    runX=dayAndMonthToBin(runDay,runMonth,year)
    if year==2017:
        runX=runDay
    yRun=np.zeros(len(y))
    yerrRun=np.zeros(len(y))
    yerr=np.zeros(len(y))
    y_bmi=np.zeros(len(y))
    yerr_bmi=np.zeros(len(y))
    xRun_temp=np.zeros(len(x))
    yRun_temp=np.zeros(len(y))
    yerrRun_temp=np.zeros(len(y))
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
    counter=0
    for i in range(len(y)):
        if i < 5:
            yerr[i]=var(y[:5])**.5
        else:
            yerr[i]=var(y[i-5:i])**.5
            if yerr[i]<0.1:
                yerr[i]=0.1
        if int(x[i]) in [int(entry) for entry in runX]:
            counter+=1
            xRun_temp[i]=x[i]
            yRun_temp[i]=y[i]
            yerrRun_temp[i]=yerr[i]
        y_bmi[i]=y[i]/(height**2)
        yerr_bmi[i]=y_bmi[i]*np.sqrt((yerr[i]/y[i])**2+(2*height_err/height)**2)
    xRun=np.array([],'d')
    yRun=np.array([],'d')
    yRun_bmi=np.array([],'d')

    for i in range(len(xRun_temp)):
        if xRun_temp[i]>0:
            xRun=np.append(xRun,xRun_temp[i])
            yRun=np.append(yRun,yRun_temp[i])
            yRun_bmi=np.append(yRun_bmi,yRun_temp[i]/(height**2))
    gew,day1= np.genfromtxt('../../stats.txt', unpack=True)
    t5,km5,bpm,bpm_max,day= np.genfromtxt('../../dataLight.txt', unpack=True)
    yerr1=np.zeros(len(gew))
    for i in range(len(gew)):
        if i < 5:
            yerr1[i]=var(gew[:5])**.5
        else:
            yerr1[i]=var(gew[i-5:i])**.5
            if yerr1[i]<0.1:
                yerr1[i]=0.1
    xRun1=np.array([],'d')
    yRun1=np.array([],'d')
    for i in range(len(xRun_temp)):
        if xRun_temp[i]>0:
            xRun=np.append(xRun,xRun_temp[i])
            yRun=np.append(yRun,yRun_temp[i])
    #~ print yerr1
    plt.figure(figsize=(10,10))
    plt.subplot(111)
    plt.errorbar(x, y, xerr=0.25, yerr=yerr, fmt='o',zorder=1)
    plt.errorbar(day1, gew, xerr=0.25, yerr=yerr1, fmt='o',zorder=2)
    plt.plot(xRun, yRun, 'rs',zorder=5)
    plt.xlabel("month")
    plt.xticks(x, labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.175)
    plt.ylabel("weight in kg")
    SavePlot(x,year,"singleCombinedStats",savepng)
