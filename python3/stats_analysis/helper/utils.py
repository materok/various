import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from math import ceil

def moment(array, mu, sigma, n):
    nom=0.
    for item in array:
        nom += ( (item-mu) / sigma )**n
    return nom/len(array)

def mean(array):
    res=moment(array,0,1,1)
    return res

def var(array):
    res=moment(array,moment(array,0,1,1),1,2)*len(array)/(len(array)-1)
    return res

def delta(arr):
    arr_new=np.array([],'d')
    for i in range(1,len(arr)):
        arr_new= np.append(arr_new, arr[i]-arr[i-1] )
    return arr_new

def fillEmpty(array):
    if len(array)>0:
        temp=array[0]
        for i,item in enumerate(array):
            if item==-1:
                array[i]=temp
            else:
                temp=item

def dayToMonth(day,year):
    # [january,febuary,march,april,may,june,july,august,september,oktober,november,december]
    month=[31,28,31,30,31,30,31,31,30,31,30,31]
    if year%4==0:
        month[1]=month[1]+1
    i=1
    counter=1
    j=0
    while i<int(day):
        counter+=1
        if counter==month[j]+1:
            j+=1
            counter=1
        i+=1
    return counter,j+1

def dayAndMonthToBin(day,month,year):
    # [january,febuary,march,april,may,june,july,august,september,oktober,november,december]
    monthDict={1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    dayDict={}
    if year%4==0:
        monthDict[2]+=1
    for key,val in monthDict.items():
        if key ==1:
            dayDict[key]=0
        else:
            dayDict[key]=dayDict[key-1]+monthDict[key-1]
    binNumbers=np.array([],'d')
    for entryDay,entryMonth in zip(day,month):
        binNumbers=np.append(binNumbers,entryDay+dayDict[entryMonth])
    return binNumbers

def convertDayToBin():
    # [january,febuary,march,april,may,june,july,august,september,oktober,november,december]
    month=[31,28,31,30,31,30,31,31,30,31,30,31]
    if year%4==0:
        month[1]=month[1]+1
    month=makeCumul(month)
    binNumber=np.array([],'d')
    for i,j in izip(days,in_month):
        binNumber=np.append(binNumber, j+i-1)
    return binNumber

def makeLabels(x,year):
    xmin=dayAndMonthToBin([1],[dayToMonth(min(x),year)[1]],year)
    xAxis=np.linspace(int(xmin),int(max(x)),int(max(x)-xmin))
    possibleLabels = ['January', 'Febuary', 'March', 'April',
                      'May','June','July','August','September',
                      'Oktober','November', 'December']
    labels=[]
    j=0
    for i in xAxis:
        if dayToMonth(i,year)[0]==1:
            j=dayToMonth(i,year)[1]-1
            labels.append(possibleLabels[j])
        else:
            labels.append('')
    return xAxis,labels

def tConvert(time):

    time_new=np.array([],'d')
    time_min=np.array([],'d')
    for time_old in time:
        min=int(time_old)
        sec=(time_old-min)
        hour=(min+sec)/60
        timeInMin=(min+sec)
        time_new=np.append(time_new,hour)
        time_min=np.append(time_min,timeInMin)
    return time_new,time_min

def calcVelo(distance,time): #distance and time are arrays

    #distance in km, time in h => vel in km/h
    for entry in time:
        if entry==0:
            entry=1
            print("entry was 0 and has been set to 1")
    vel=distance/time
    return vel

def calcWeightUncert(y):
    yerr=np.zeros(len(y))
    for i in range(len(y)):
        if i < 5:
            yerr[i]=var(y[:5])**.5
        else:
            yerr[i]=var(y[i-4:i+1])**.5
            if yerr[i]<0.1:
                yerr[i]=0.1
    return yerr

def makeCumul(distance): #distance is an array

    distanceCumul=np.array([],'d')
    cumul=0
    for dist in distance:
        cumul+=dist
        distanceCumul=np.append(distanceCumul,cumul)
    return distanceCumul

def percentage(day,month=[],year=2017): #day is an array

    n=0.
    perc_arr=np.array([],'d')
    if year==2016:
        for run in day:
            #print(run)
            n+=1
            perc=n/run*100
            perc_arr=np.append(perc_arr,perc)
    else:
        binNumbers=dayAndMonthToBin(day,month,year)
        for run in binNumbers:
            n+=1
            perc=n/run*100.
            perc_arr=np.append(perc_arr,perc)
    return perc_arr

def TimeError(time):

    const=20./60.
    errors=np.array([],'d')
    for value in time:
        errors=np.append(errors,const)
    return errors

def VelError(velo):

    const=50./2100.
    errors=np.array([],'d')
    for value in velo:
        errors=np.append(errors,value*const)
    return errors

def prepRunArray(dataClass,year,x,y):
    tmp_xRun=dataClass.run[year]["bin"]
    tmp_yRun=np.zeros(len(tmp_xRun))
    for i in range(len(tmp_xRun)):
        searchArray=[int(entry) for entry in x]
        if int(tmp_xRun[i]) in searchArray:
            tmp_yRun[i]=y[searchArray.index(int(tmp_xRun[i]))]
            tmp_xRun[i]=x[searchArray.index(int(tmp_xRun[i]))]
    xRun=np.array([],'d')
    yRun=np.array([],'d')
    for i in range(len(tmp_xRun)):
        if tmp_yRun[i]!=0:
            xRun=np.append(xRun,tmp_xRun[i])
            yRun=np.append(yRun,tmp_yRun[i])
    return xRun,yRun

def prepPushupArray(dataClass,year,x,y):
    tmp_xU=dataClass.training[year]["bin"]
    tmp_yU=np.zeros(len(tmp_xU))
    for i in range(len(tmp_xU)):
        searchArray=[int(entry) for entry in x]
        if int(tmp_xU[i]) in searchArray:
            tmp_yU[i]=y[searchArray.index(int(tmp_xU[i]))]
            tmp_xU[i]=x[searchArray.index(int(tmp_xU[i]))]
    xU=np.array([],'d')
    yU=np.array([],'d')
    for i in range(len(tmp_xU)):
        if tmp_yU[i]!=0:
            xU=np.append(xU,tmp_xU[i])
            yU=np.append(yU,tmp_yU[i])
    return xU,yU

def DrawWeightInYear(dataClass, year, zorder=1, drawLine=False, fmt="o", drawAll=True, fullYear=False):
    if year not in dataClass.weight:
        return
    x=dataClass.weight[year]["bin"]
    y=dataClass.weight[year]["weight"]
    if fullYear:
        if x[-1]<364:
            x=np.append(x,365)
            y=np.append(y,y[-1])
    yerr=calcWeightUncert(y)
    xRun,yRun=prepRunArray(dataClass,year,x,y)
    plt.errorbar(x, y, xerr=0.25, yerr=yerr, fmt=fmt,zorder=zorder)
    zorder+=1
    if drawAll and not dataClass.short:
        if year>2018 and year<2022:
            xPushup,yPushup=prepPushupArray(dataClass,year,x,y)
            plt.errorbar(xPushup, yPushup, xerr=0, yerr=0, fmt='D',zorder=zorder)
            zorder+=1
    if drawAll:
        zorder+=1
        plt.errorbar(xRun, yRun, xerr=0, yerr=0, fmt='s',zorder=zorder)
    if drawLine:
        for i in range(int(min(y/(1.7**2))),int(max(y/(1.7**2)))+2):
            plt.plot((min(x),max(x)),(i*(1.7**2),i*(1.7**2)),  color = 'k')
    plt.xlabel("month")
    plt.ylabel("weight in kg")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.175)
    return x, y, yerr, xAxis, labels

def DrawBMIInYear(dataClass, year, height=1.7, heightErr=0.01, zorder=1):
    if year not in dataClass.weight:
        return
    x=dataClass.weight[year]["bin"]
    y=dataClass.weight[year]["weight"]
    yerr=calcWeightUncert(y)
    rel_h_err=heightErr/height
    for i, _ in enumerate(yerr):
        yerr[i]=y[i]*np.sqrt((yerr[i]/y[i])**2+(2*rel_h_err)**2)
    yerr/=(height**2)
    y/=(height**2)
    xRun,yRun=prepRunArray(dataClass,year,x,y)
    plt.errorbar(x, y, xerr=0.25, yerr=yerr, fmt='o',zorder=zorder)
    zorder+=1
    plt.errorbar(xRun, yRun, xerr=0, yerr=0, fmt='s',zorder=zorder)
    zorder+=1
    for i in range(int(min(y)),int(max(y))+2):
        plt.plot((min(x),max(x)),(i,i),  color = 'k')
    plt.xlabel("month")
    plt.ylabel(r"BMI in kg/cm$^2$")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.175)
    y*=(height**2)

def MakeCombinedStats(dataClass):
    plt.figure(figsize=(10,10))
    zorder=1
    yearForLabel=2018
    xAxisDict={}
    labelsDict={}
    inRange=False
    for year in range(dataClass.first_year,dataClass.last_year+1):
        x, y, yerr, xAxis, labels=DrawWeightInYear(dataClass, year, zorder=1, drawLine=False, fmt="o", drawAll=False)
        xAxisDict[year]=xAxis
        labelsDict[year]=labels
        if year==yearForLabel:
            inRange=True
    if inRange==False:
        maxlen=0
        for year in range(dataClass.first_year,dataClass.last_year+1):
            if maxlen<len(xAxisDict[year]):
                maxlen=len(xAxisDict[year])
                yearForLabel=year
    plt.xticks(xAxisDict[yearForLabel], labelsDict[yearForLabel], rotation='vertical')
    SavePlot("combinedStats")
    plt.close()

def addX(xs):
    x=np.array([0])
    for i in xs:
        x=np.append(x,i+x[-1])
    return x[1:]

def add2X(x1s,x2s):
    x1=np.array([0])
    x2=np.array([0])
    for i,j in zip(x1s,x2s):
        x2=np.append(x2,j+x1[-1])
        x1=np.append(x1,i+x1[-1])
    return x1[1:], x2[1:]

def addY(ys):
    y=np.array([])
    for i in ys:
        y=np.append(y,i)
    return y

def MakeLongStatsPlot(dataClass):
    plt.figure(figsize=(20,20))
    zorder=1

    xs=[]; ys=[]; yerrs=[]; xRuns=[]; yRuns=[]; xAxes=[]; labels=[];
    #years=[]; x_years=[]
    for year in range(dataClass.first_year,dataClass.last_year+1):
        if year not in dataClass.weight:
            continue
        #years.append(year)
        fullYear=True
        if year==dataClass.last_year: fullYear=False
        x, y, yerr, xAxis, labelsYear= DrawWeightInYear(dataClass, year, fullYear=fullYear)
        xRun,yRun=prepRunArray(dataClass,year,x,y)
        xs.append(x); ys.append(y); yerrs.append(yerr)
        xRuns.append(xRun); yRuns.append(yRun);
        xAxes.append(xAxis); labels.append(labelsYear)
        #if len(x_years)==0:
            #x_years.append(xAxis[0])
        #else:
            #x_years.append(xAxis[0]+xAxes[-1])
    x,xRun=add2X(xs,xRuns)
    y=addY(ys)
    yRun=addY(yRuns)
    yerr=calcWeightUncert(y)
    xAxis=addX(xAxes)
    label=[]
    for l in labels:
        for i in l:
            label.append(i)
    plt.clf()
    plt.errorbar(x, y, xerr=0.25, yerr=yerr, fmt='o',zorder=zorder)
    plt.errorbar(xRun, yRun, xerr=0, yerr=0, fmt='o',zorder=zorder)
    #for i,year in enumerate(years):
        #print(x_years[i])
        #plt.vlines(x_years[i], min(y), max(y), label=str(year))
    plt.xticks(xAxis, label, rotation='vertical')
    plt.ylabel("weight in kg")
    plt.plot((min(x),max(x)),(min(y),min(y)),  color = 'k')
    plt.plot((min(x),max(x)),(y[-1],y[-1]),  color = 'red')

    SavePlot("longStats",tight=False)
    """
    if fit:
        if True:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score

            polynomial_features = PolynomialFeatures(degree=9,
                                                     include_bias=False)
                                                     # include_bias=True)
            linear_regression = LinearRegression()
            pipeline = Pipeline([("polynomial_features", polynomial_features),
                                 ("linear_regression", linear_regression)])
            pipeline.fit(x[:, np.newaxis], y)
            # Evaluate the models using crossvalidation
            scores = cross_val_score(pipeline, x[:, np.newaxis], y,
                                     scoring="neg_mean_squared_error", cv=10)

            X_test = np.linspace(x[0],x[-1] ,x[-1]-x[0] )
            plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
            SavePlot("longStatsFit",tight=False)

        if False:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import KBinsDiscretizer
            enc = KBinsDiscretizer(n_bins=20, encode='onehot')
            X_binned = enc.fit_transform(x[:, np.newaxis])
            line = np.linspace(x[0],x[-1]+100 ,x[-1]-x[0]+100 ).reshape(-1, 1)
            line_binned = enc.transform(line)
            reg = LinearRegression().fit(X_binned, y)
            plt.plot(line, reg.predict(line_binned), linewidth=2, color='green',
                     linestyle='-', label='linear regression')
            SavePlot("longStatsFit",tight=False)
        if False:
            from sklearn.tree import DecisionTreeRegressor
            line = np.linspace(x[0], x[-1], 1000, endpoint=False).reshape(-1, 1)
            X=x.reshape(-1,1)
            # reg = DecisionTreeRegressor(min_samples_split=10, max_depth=10, min_weight_fraction_leaf=0.01, random_state=0).fit(X, y)
            reg = DecisionTreeRegressor(min_samples_split=10, random_state=0).fit(X, y)
            line = np.linspace(x[0], x[-1]+1000, x[-1]-x[0]+1000, endpoint=False).reshape(-1, 1)
            plt.plot(line, reg.predict(line), linewidth=2, color='red',
                     label="decision tree")
            SavePlot("longStatsFit",tight=False)
    """
    plt.close()

def MakeLongCumulPlot(dataClass, bike=False):
    plt.figure(figsize=(20,20))
    zorder=1

    xs=[]; ys=[]; cumulys=[]; xAxes=[]; labels=[]
    addMe=0
    for year in range(dataClass.first_year,dataClass.last_year+1):
        if year not in dataClass.run:
            continue
        if year > dataClass.first_year:
            addMe+=dayAndMonthToBin([31],[12],year)[0]
        xs.append(dataClass.run[year]["bin"]+addMe); ys.append(dataClass.run[year]["distance"]);
        x_for_axis=dataClass.run[year]["bin"]
        if x_for_axis[0]!=1.0: x_for_axis=np.insert(x_for_axis,0,1.0)
        if x_for_axis[-1]!=dayAndMonthToBin([31],[12],year) and year != dataClass.last_year: x_for_axis=np.append(x_for_axis,dayAndMonthToBin([31],[12],year))
        xAxis, labelsYear = makeLabels(x_for_axis,year)
        #print(xs[-1])
        #print(x_for_axis)
        cumulys.append(makeCumul(dataClass.run[year]["distance"]))
        xAxes.append(xAxis); labels.append(labelsYear)
    x=addY(xs)
    #print(x)
    y=addY(cumulys)
    toty=addY(ys)
    toty=makeCumul(toty)
    xAxis=addX(xAxes)
    label=[]
    for l in labels:
        for i in l:
            label.append(i)
    plt.clf()
    plt.errorbar(x, y, xerr=0, yerr=0, fmt='o',zorder=zorder)
    label=[]
    for l in labels:
        for i in l:
            label.append(i)
    plt.xticks(xAxis, label, rotation='vertical')
    plt.title("gelaufene Strecke")
    plt.xlabel("Monat")
    plt.ylabel("Strecke in km")
    saveName="longCumul"
    if bike: saveName="bike_"+saveName
    SavePlot(saveName,tight=False)
    plt.clf()
    plt.errorbar(x, toty, xerr=0, yerr=0, fmt='o',zorder=zorder)
    plt.xticks(xAxis, label, rotation='vertical')
    saveName="longTotalCumul"
    if bike: saveName="bike_"+saveName
    SavePlot(saveName,tight=False)
    plt.close()
    print(y[-1])
    print(toty[-1])

def MakeMonthCumulPlot(dataClass, onlyYear=None, bike=False):
    plt.figure(figsize=(20,20))
    zorder=1

    possibleLabels = ['January', 'Febuary', 'March', 'April',
                      'May','June','July','August','September',
                      'Oktober','November', 'December']
    counter=0
    data={}
    label=[]; xs=[]; ys=[]
    lastMonth=0
    for year in range(dataClass.first_year,dataClass.last_year+1):
        if year not in dataClass.run:
            continue
        if onlyYear is not None and year != onlyYear:
            continue
        data[year]={}
        for i in range(1,13):
            data[year][i]=0
            label.append(possibleLabels[i-1])
        for d, mon, dist in zip(dataClass.run[year]["day"],dataClass.run[year]["month"],dataClass.run[year]["distance"]):
            data[year][mon]+=dist
        xs.append(addY(data[year]))
        for i in range(1,13):
            ys.append(data[year][i])
        if year!=dataClass.last_year:
            lastMonth+=12
        else:
            lastMonth+=dataClass.run[year]["month"][-1]
    lastMonth=int(lastMonth)
    x=addY(xs)
    y=addY(ys)
    toty=(makeCumul(y))
    xAxis=[i for i in range(len(x))]
    plt.clf()
    plt.errorbar(xAxis[:lastMonth], y[:lastMonth], xerr=0, yerr=0, fmt='o',zorder=zorder)
    plt.xticks(xAxis[:lastMonth], label[:lastMonth], rotation='vertical')
    plt.title("gelaufene Strecke")
    plt.xlabel("Monat")
    plt.ylabel("Strecke in km")
    outDir=""
    if onlyYear is not None: outDir="%i/"%onlyYear
    if bike: outDir+="bike_"
    SavePlot(outDir+"monthCumul",tight=False)
    plt.clf()
    plt.errorbar(xAxis[:lastMonth], toty[:lastMonth], xerr=0, yerr=0, fmt='o',zorder=zorder)
    plt.xticks(xAxis[:lastMonth], label[:lastMonth], rotation='vertical')
    SavePlot(outDir+"monthTotalCumul",tight=False)
    plt.close()
    print(y[lastMonth-1])
    #print(toty[-1])

def MakeLongRHRPlot(dataClass):
    plt.figure(figsize=(20,20))
    zorder=1

    xs=[]; ys=[]; xRuns=[]; yRuns=[]; xAxes=[]; labels=[]
    for year in range(dataClass.first_year,dataClass.last_year+1):
        if year not in dataClass.HR:
            continue
        x, y, xAxis, labelsYear= PlotRHR(dataClass, year, False)
        xs.append(x); ys.append(y);
        xAxes.append(xAxis); labels.append(labelsYear)
    x=addX(xs)
    y=addY(ys)
    xAxis=addX(xAxes)
    label=[]
    for l in labels:
        for i in l:
            label.append(i)
    plt.clf()
    plt.plot(x, y, linestyle='', marker="+", zorder=zorder)
    plt.xticks(xAxis, label, rotation='vertical')
    plt.ylabel("resting heartrate")
    SavePlot("longRHR",tight=False)
    plt.close()

def MakeStats(dataClass,year):
    plt.figure(figsize=(10,10))
    x, y, yerr, xAxis, labels= DrawWeightInYear(dataClass, year, zorder=1, fmt=",")
    plt.plot((min(x),max(x)),(min(y),min(y)),  color = 'k')
    plt.plot((min(x),max(x)),(y[-1],y[-1]),  color = 'red')
    SavePlot("%i/singleStats"%(year))
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        polynomial_features = PolynomialFeatures(degree=5,
                                                 include_bias=False)
                                                 # include_bias=True)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(x[:, np.newaxis], y)
        # Evaluate the models using crossvalidation
        scores = cross_val_score(pipeline, x[:, np.newaxis], y,
                                 scoring="neg_mean_squared_error", cv=10)

        X_test = np.linspace(x[0],x[-1]+3,x[-1]-x[0]+3)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
        SavePlot("%i/singleStatsFit"%(year),tight=False)
    except:
        print ("couldnt fit")
    """
    plt.close()
    plt.figure(figsize=(20,10))
    plt.subplot(121)
    DrawWeightInYear(dataClass, year, zorder=1, drawLine=True)
    plt.subplot(122)
    DrawBMIInYear(dataClass, year, zorder=1)
    SavePlot("%i/Stats"%(year))
    plt.close()

def MakeKMPlots(day,time,velo,year=2016,where="",show=False):

    plt.figure(figsize=(30,10))
    plt.subplot(131)
    xAxis,labels=makeLabels(day,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.errorbar(day,time,xerr=np.zeros(len(day)),yerr=TimeError(time), fmt='o')
    plt.xlabel("Monat")
    plt.ylabel("Zeit in minuten")

    plt.subplot(132)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.errorbar(day,velo,xerr=np.zeros(len(day)),yerr=VelError(velo), fmt='o')
    plt.xlabel("Monat")
    plt.ylabel("Geschwindigkeit in km/h")

    plt.subplot(133)
    deltaVel=delta(velo)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(day[1:len(day)],deltaVel, linestyle='', marker="+")
    plt.xlabel("Monat")
    plt.ylabel("Differenz von Geschwindigkeit in km/h")
    SavePlot(where+"kmPlots")
    if show==True: plt.show()
    plt.close()

def MakeCumulPlot(day,distance,year=2016,where="",show=False, bike=False):

    import datetime
    today = datetime.date.today()
    lastRun = datetime.date(year,dayToMonth(day[-1],year)[1],dayToMonth(day[-1],year)[0])
    diff = today -lastRun

    cumulDist= makeCumul(distance)

    if diff.days!=0:
        if (day[-1]+diff.days)<365:
            day=np.append(day,day[-1]+diff.days)
        else:
            day=np.append(day,364)
        cumulDist=np.append(cumulDist,cumulDist[-1])

    plt.figure(figsize=(10,10))
    xAxis,labels=makeLabels(day,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(day,cumulDist,marker="+")
    maxVal=(int(day[-1]/7.)+1)*20
    plt.title("gelaufene Strecke")
    plt.xlabel("Monat")
    plt.ylabel("Strecke in km")
    if bike: where+="bike_"
    SavePlot(where+"cumul")
    if show==True: plt.show()
    plt.close()

def MakePercPlot(day,month,year=2016,show=False):

    plt.figure(figsize=(10,10))
    perc=percentage(day,month,year)
    xAxis,labels=makeLabels(day,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(day,perc,marker="+")
    plt.title("Lauf Prozent")
    plt.xlabel("Monat")
    plt.ylabel("Prozent")
    SavePlot("perc")
    if show==True: plt.show()
    plt.close()

def MakeKMHPlot(day,velo,dist,where,savepng=False,show=False, timeBool=False, bike=False):

    plt.figure(figsize=(8, 8))
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # now determine nice limits by hand:
    #binwidth = .25
    binwidth = 1
    binwidthy=binwidth
    lowx = max(int(min(dist))-binwidth,0)
    # highx = max(dist)+1+binwidth
    highx = int(max(dist))+binwidth
    lowy = int(min(velo))-binwidthy
    highy = int(max(velo))+1+binwidthy
    if timeBool:
        binwidthy*=10
        lowy = int(min(velo)/10)*10-binwidthy
        highy = int(max(velo)/10)*10+10+binwidthy

    axScatter.set_xlim((lowx, highx))
    axScatter.set_ylim((lowy, highy))

    binsx = np.arange(lowx, highx+binwidth, binwidth)
    binsy = np.arange(lowy, highy+binwidthy, binwidthy)

    # setup colorcoding
    hist, xedges, yedges = np.histogram2d(dist, velo,(binsx, binsy))
    #xidx = np.clip(np.digitize(velo, binsx), 0, hist.shape[0]-1)
    #yidx = np.clip(np.digitize(dist, binsy), 0, hist.shape[1]-1)
    #colors = hist[xidx-1, yidx-1]
    # the scatter plot:
    #axScatter.scatter(dist, velo,c=colors)
    axScatter.scatter(dist, velo)
    #axScatter.scatter(velo, dist)

    #setup gridspacing

    from matplotlib.ticker import MultipleLocator
    spacing = .25
    minorLocator = MultipleLocator(spacing)
    axScatter.yaxis.set_minor_locator(minorLocator)
    axScatter.xaxis.set_minor_locator(minorLocator)

    #axScatter.grid(color='r', linestyle='--', linewidth=.5, which = 'minor')
    axScatter.grid(color='r', linestyle='-', linewidth=.5)
    axHistx.hist(dist, bins=binsx)
    axHisty.hist(velo, bins=binsy, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.set_xlabel("distance in km")
    if not timeBool: axScatter.set_ylabel("velocity in km/h")
    else:
        axScatter.set_ylabel("time per km in min:sec")
        import time
        import matplotlib.ticker as ticker
        formatter = ticker.FuncFormatter(lambda ms, y: time.strftime('%M:%S', time.gmtime(ms)))
        axScatter.yaxis.set_major_formatter(formatter)
    axHistx.set_ylabel("#entries")
    axHisty.set_xlabel("#entries")
    if bike: where += "bike_"
    if timeBool: SavePlot(where+"kmh2",savepng=savepng,tight=False)
    else: SavePlot(where+"kmh",savepng=savepng,tight=False)
    if show==True: plt.show()
    plt.close()

def MakeKMH_BPMPlot(velo_orig,bpm_orig,where,savepng=False,show=False, maxBPM=False, timeBool=False, bike=False):

    plt.figure(figsize=(8, 8))
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # now determine nice limits by hand:
    #binwidth = .25
    binwidth = 1
    binwidthy=binwidth

    bpm=[x for x in bpm_orig if x>0]
    velo=[y for x,y in zip(bpm_orig,velo_orig) if x>0]
    lowx = int(min(bpm))-binwidth
    highx = max(bpm)+1+binwidth
    lowy = int(min(velo))-binwidthy
    highy = int(max(velo))+1+binwidthy
    if timeBool:
        binwidthy*=10
        lowy = int(min(velo)/10)*10-binwidthy
        highy = int(max(velo)/60)*60+10+binwidthy

    axScatter.set_xlim((lowx, highx))
    axScatter.set_ylim((lowy, highy))

    binsx = np.arange(lowx, highx+binwidth, binwidth)
    binsy = np.arange(lowy, highy+binwidthy, binwidthy)

    # setup colorcoding
    hist, xedges, yedges = np.histogram2d(bpm, velo,(binsx, binsy))
    xidx = np.clip(np.digitize(velo, binsx), 0, hist.shape[0]-1)
    yidx = np.clip(np.digitize(bpm, binsy), 0, hist.shape[1]-1)
    colors = hist[xidx-1, yidx-1]
    # the scatter plot:
    axScatter.scatter(bpm, velo,c=colors)
    #axScatter.scatter(velo, dist)

    #setup gridspacing

    from matplotlib.ticker import MultipleLocator
    spacing = .25
    minorLocator = MultipleLocator(spacing)
    axScatter.yaxis.set_minor_locator(minorLocator)
    axScatter.xaxis.set_minor_locator(minorLocator)

    #axScatter.grid(color='r', linestyle='--', linewidth=.5, which = 'minor')
    axScatter.grid(color='r', linestyle='-', linewidth=.5)
    axHistx.hist(bpm, bins=binsx)
    axHisty.hist(velo, bins=binsy, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.set_xlabel("average heart rate in bpm")
    if maxBPM: axScatter.set_xlabel("maximal heart rate in bpm")
    if not timeBool: axScatter.set_ylabel("velocity in km/h")
    else:
        axScatter.set_ylabel("time per km in min:sec")
        import time
        import matplotlib.ticker as ticker
        formatter = ticker.FuncFormatter(lambda ms, y: time.strftime('%M:%S', time.gmtime(ms)))
        axScatter.yaxis.set_major_formatter(formatter)
    axHistx.set_ylabel("#entries")
    axHisty.set_xlabel("#entries")
    adder=""
    if maxBPM: adder="max"
    if bike: where += "bike_"
    if timeBool: SavePlot(where+"kmh2_bpm"+adder,savepng=savepng,tight=False)
    else: SavePlot(where+"kmh_bpm"+adder,savepng=savepng,tight=False)
    if show==True: plt.show()
    plt.close()

def MakeBPMPlots(day_orig,bpm_orig,where,option="avg",year=2016,show=False, bike=False):

    plt.figure(figsize=(10,10))
    bpm=[x for x in bpm_orig if x>0]
    day=[y for x,y in zip(bpm_orig,day_orig) if x>0]
    xAxis,labels=makeLabels(day,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(day,bpm,linestyle="",marker="x")
    adder=""
    if option=="avg":
        pass
    elif option=="max":
        adder="max. "
    else: print("this option is not supported")
    plt.title(adder+"Herzfrequenz")
    plt.xlabel("Monat")
    plt.ylabel(adder+ "Herzfrequenz in bpm")
    if bike: where += "bike_"
    SavePlot(where+adder.split(".")[0]+"bpm")
    if show==True: plt.show()
    plt.close()

def MakeDeltaPlots(day,month,weight,year=2016,show=False,savepng=False):

    binNumbers=dayAndMonthToBin(day,month,year)
    nWeeks=int(ceil(int(binNumbers[-1]/7)))
    x= np.ones(nWeeks)
    deltaWeightMinMax = np.ones(nWeeks)
    deltaWeight0m1 = np.ones(nWeeks)
    WeightAvg = np.ones(nWeeks)
    WeightAvgError = np.ones(nWeeks)
    lastIndex=0
    for i in range(nWeeks):
        x[i]=i+1
        lowBorder=i*7
        upBorder=(i+1)*7
        tmpArray=np.array([],'d')
        while lastIndex<len(binNumbers) and int(binNumbers[lastIndex])>lowBorder and int(binNumbers[lastIndex])<=upBorder:
            tmpArray=np.append(tmpArray,weight[lastIndex])
            #~ print(binNumbers[lastIndex],lowBorder,upBorder,lastIndex)
            lastIndex+=1
        deltaWeightMinMax[i]=max(tmpArray)-min(tmpArray)
        deltaWeight0m1[i]=tmpArray[0]-tmpArray[-1]
        arrayMean=mean(tmpArray)
        WeightAvg[i]=arrayMean
        WeightAvgError[i]=var(tmpArray)

    plt.figure(figsize=(10,10))
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(x,deltaWeightMinMax,linestyle="",marker="x")
    plt.title("$\Delta$ weight (max-min)")
    plt.xlabel("Monat")
    plt.ylabel("$\Delta$ weight in kg")
    SavePlot("deltaweightMinMax",savepng)
    if show==True: plt.show()
    plt.close()
    plt.figure(figsize=(10,10))
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(x,deltaWeight0m1,linestyle="",marker="x")
    plt.title("$\Delta$ weight (first minus last day of the week)")
    plt.xlabel("Monat")
    plt.ylabel("$\Delta$ weight in kg")
    SavePlot("deltaweightFirstLast",savepng)
    if show==True: plt.show()
    plt.close()
    plt.figure(figsize=(10,10))
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(x,WeightAvgError,linestyle="",marker="x")
    plt.title("variance of weight")
    plt.xlabel("Monat")
    plt.ylabel("weight variance in kg")
    SavePlot("weightVar",savepng)
    if show==True: plt.show()
    plt.close()
    plt.figure(figsize=(10,10))
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.errorbar(x,WeightAvg,xerr=0,yerr=WeightAvgError,linestyle="",marker="x")
    plt.title("average weight")
    plt.xlabel("Monat")
    plt.ylabel("average weight in kg")
    SavePlot("avgWeight",savepng)
    if show==True: plt.show()
    plt.close()

def MakeDeltaPlot(bins,weight,year=2016,where="",show=False,savepng=False):

    x=bins
    weightDelta = delta(weight)
    weightDelta = np.append(weightDelta,0)
    nWeeks=int(ceil((x[-1])/7))
    #print(nWeeks)
    weightDeltaRebin = np.zeros(nWeeks)
    xRebin = np.ones(nWeeks)
    i=0
    for pos,weight in zip(x,weightDelta):
        curWeek=i+1
        weightDeltaRebin[i]+=weight
        if int(pos/7)==curWeek: i+=1
        # print(xRebin)
        # print(i)
        xRebin[i]=i+1

    plt.figure(figsize=(10,10))
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(x,weightDelta,linestyle="",marker="x")
    plt.title("$\Delta$ weight")
    plt.xlabel("Monat")
    plt.ylabel("$\Delta$ weight in kg")
    plt.grid(True)
    SavePlot(where+"deltaweight")
    plt.close()
    plt.figure(figsize=(10,10))
    maxWeeks=365/7.

    names=["January","Febuary","March","April","May","June","July","August","September","Oktober","November","December"]
    monthDict={1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    xAxis=[1./7]
    labels=[names[0]]
    for key,val in zip(monthDict.keys(),monthDict.values()):
        if key!=12 and (val/7.+xAxis[-1]<nWeeks):
            xAxis.append(val/7.+xAxis[-1])
            labels.append(names[key])
    xAxis[0]=1
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(xRebin,weightDeltaRebin,linestyle="",marker="o")
    plt.title("$\Delta$ weight")
    plt.xlabel("Monat")
    plt.ylabel("$\Delta$ weight in kg")
    plt.grid(True)
    SavePlot(where+"deltaweightRebin",savepng)
    if show==True: plt.show()
    plt.close()

def MakeComposition(bins,fett,wasser,muskel,knochen,year,where="",show=False,savepng=True):

    #x=dayAndMonthToBin(day,month,year)

    plt.figure(figsize=(10,10))
    xAxis,labels=makeLabels(bins,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(bins,fett,linestyle="",marker="x")
    plt.xlabel("Monat")
    plt.ylabel("Fett in %")
    SavePlot(where+"body1_",savepng)
    plt.close()
    plt.figure(figsize=(10,10))
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(bins,wasser,linestyle="",marker="x")
    plt.xlabel("Monat")
    plt.ylabel("Wasser in %")
    SavePlot(where+"body2_",savepng)
    plt.close()
    plt.figure(figsize=(10,10))
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(bins,muskel,linestyle="",marker="x")
    plt.xlabel("Monat")
    plt.ylabel("Muskel in %")
    SavePlot(where+"body3_",savepng)
    plt.close()
    plt.figure(figsize=(10,10))
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.plot(bins,knochen,linestyle="",marker="x")
    plt.xlabel("Monat")
    plt.ylabel("Knochen in %")
    SavePlot(where+"body4_",savepng)
    plt.close()
    labels = 'Fett', 'Wasser', 'Muskel', 'Knochen'
    sizes = [fett[-1], wasser[-1], muskel[-1], knochen[-1]]
    total = sum(sizes)
    plt.figure(figsize=(10,10))
    plt.pie(sizes, labels=labels, autopct=lambda p: '{:.1f}'.format(p * total / 100), startangle=90)
    SavePlot(where+"body5_",savepng)
    plt.close()
    plt.figure(figsize=(10,10))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    SavePlot(where+"body6_",savepng)
    if show==True: plt.show()
    plt.close()


def SavePlot(title,savepng=True,tight=True):
    title=title.replace("_","")
    if tight: plt.tight_layout()
    plt.savefig("../../plots/"+title+".pdf")
    if savepng: plt.savefig("../../plots/"+title+".png")

def fitIt(gew,day,month,year,where="",thresh=69.5):
    y=gew
    plt.figure(figsize=(20,10))
    #x=dayAndMonthToBin(day,month,year)
    yRun=np.zeros(len(y))
    yerrRun=np.zeros(len(y))
    yerr=np.zeros(len(y))
    y_bmi=np.zeros(len(y))
    yerr_bmi=np.zeros(len(y))
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
            yerr[i]=var(y[i-4:i+1])**.5
            if yerr[i]<0.1:
                yerr[i]=0.1
    from scipy.optimize import curve_fit
    def fit_func(x,a,b):
        return a*x+b
    fitStart=0
    fitEnd=int(x[-1])
    if year==17:
        fitStart=np.where(y == y.max())[0][0]
        fitEnd=np.where(y < thresh )[0][0]
    popt, pcov = curve_fit(fit_func, x[fitStart:fitEnd], y[fitStart:fitEnd])
    print(popt)
    plt.close()
    plt.figure(figsize=(10,10))
    plt.errorbar(x, y, xerr=0.25, yerr=yerr, fmt='o',zorder=1)
    minIndices=np.where(y == y.min())
    plt.plot(x[minIndices], y[minIndices], 'h',zorder=6,color="green")
    fitX=np.linspace(fitStart,fitEnd,100)
    plt.plot(fitX, fit_func(fitX,*popt), '',zorder=6,color="yellow", label='fit')

    from scipy.stats import chisquare
    print( chisquare(y[fitStart:fitEnd],fit_func(x[fitStart:fitEnd],*popt)) )

    plt.xlabel("month")
    plt.xticks(x, labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.175)
    plt.ylabel("weight in kg")#
    #plt.show()
    plt.tight_layout()
    SavePlot(where+"singeStatFit")
    plt.close()

def plotTimePerKM(vel,day,year,where="", bike=False):
    import time
    import matplotlib.ticker as ticker
    """
    make a function that shows time per km in min:seconds correctly
    axis labels: every 10 seconds
    stepsize: 1./6
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xticks.html
    xticks(np.arange(int(min), int(max), step=1./6))
    min/km = 60/v[km/h]
    """
    y=60.*60./vel
    #plt.figure(figsize=(20,10))
    x=day
    yRun=np.zeros(len(y))
    yerrRun=np.zeros(len(y))
    yerr=np.zeros(len(y))

    plt.figure(figsize=(10,10))
    plt.errorbar(x, y, xerr=0.25, yerr=yerr, fmt='o',zorder=1)

    formatter = ticker.FuncFormatter(lambda ms, y: time.strftime('%M:%S', time.gmtime(ms)))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)

    plt.xlabel("month")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.175)
    plt.ylabel("time per kilometer [min:sec]")#
    #plt.show()
    if bike: where += "bike_"
    SavePlot(where+"timePerKilometer")
    plt.close()

def plotVel(vel,day,year,where="", bike=False):
    import matplotlib.ticker as ticker
    y=vel
    #plt.figure(figsize=(20,10))
    x=day
    yRun=np.zeros(len(y))
    yerrRun=np.zeros(len(y))
    yerr=np.zeros(len(y))

    plt.figure(figsize=(10,10))
    plt.errorbar(x, y, xerr=0.25, yerr=yerr, fmt='o',zorder=1)

    plt.xlabel("month")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.175)
    plt.ylabel("velocity [km/h]")#
    #plt.show()
    if bike: where += "bike_"
    SavePlot(where+"vel")
    plt.close()

def plotLBLequiv(vel,day,year,where=""):
    import time
    import matplotlib.ticker as ticker
    """
    make a function that shows time per km in min:seconds correctly
    axis labels: every 10 seconds
    stepsize: 1./6
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xticks.html
    xticks(np.arange(int(min), int(max), step=1./6))
    min/km = 60/v[km/h]
    """
    y=60.*60./vel*5.555
    #plt.figure(figsize=(20,10))
    x=day
    yRun=np.zeros(len(y))
    yerrRun=np.zeros(len(y))
    yerr=np.zeros(len(y))

    plt.figure(figsize=(10,10))
    plt.errorbar(x, y, xerr=0.25, yerr=yerr, fmt='o',zorder=1)

    formatter = ticker.FuncFormatter(lambda ms, y: time.strftime('%M:%S', time.gmtime(ms)))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)

    plt.xlabel("month")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.175)
    plt.ylabel("time for LBL [min:sec]")#
    #plt.show()
    SavePlot(where+"LBLequiv")
    plt.close()

def plotPushups(number,day,year):
    y=number
    #plt.figure(figsize=(20,10))
    x=day
    yRun=np.zeros(len(y))
    yerrRun=np.zeros(len(y))
    yerr=np.zeros(len(y))

    plt.figure(figsize=(10,10))
    plt.errorbar(x, y, xerr=0.25, yerr=yerr, fmt='o',zorder=1)

    plt.xlabel("month")
    plt.ylabel("number of pushups")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.175)
    #plt.show()
    plt.tight_layout()
    plt.savefig("../../plots/pushups.pdf")
    plt.savefig("../../plots/pushups.png")
    plt.close()

def plotVeloHeight(hoehe,vel5,where="", bike=False):
    x=[x for x in hoehe if x>0]
    y=[y for x,y in zip(hoehe,vel5) if x>0]

    plt.figure(figsize=(10,10))
    plt.plot(x, y, 'rs')

    plt.xlabel("overcome altitude")
    plt.ylabel("velocity")
    #xAxis,labels=makeLabels(x,year)
    #plt.xticks(xAxis, labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.175)
    #plt.show()
    if bike: where += "bike_"
    SavePlot(where+"altitudeVelo")
    plt.close()

def plotBPMHeight(hoehe,bpm,where,option="", bike=False):
    #x=hoehe
    #y=bpm
    x=[]
    y=[]
    for i,j in zip(hoehe,bpm):
        if j>0 and i>0:
            x.append(i)
            y.append(j)

    plt.figure(figsize=(10,10))
    plt.plot(x, y, 'rs')

    plt.xlabel("overcome altitude")
    plt.ylabel("bpm")
    if option=="max":
        plt.ylabel("max bpm")
    plt.subplots_adjust(bottom=0.175)
    plt.tight_layout()
    if bike: where += "bike_"
    if option=="":
        SavePlot(where+"altitudeBPM")
    elif option=="max":
        SavePlot(where+"altitudeMaxBPM")
    plt.close()


def makeMoodPlot(dataClass,year=2019):
    plt.figure(figsize=(10,10))
    ax=plt.gca()
    colors=["cyan","blue","lightgray","crimson",
            "lawngreen","gold","gray","black"]
    for i, j, k in zip(dataClass.mood[year]["month"],dataClass.mood[year]["day"],dataClass.mood[year]["colorcode"]):
        if len(k)==1:
            ax.add_patch(matplotlib.patches.Rectangle(xy=(i-1,j-1),width=1,height=1,color=colors[k[0]]))
        elif len(k)==2:
            array1=[[i-1,j-1],[i,j-1],[i,j]]
            array2=[[i-1,j-1],[i-1,j],[i,j]]
            ax.add_patch(plt.Polygon(xy=array1,color=colors[k[0]]))
            ax.add_patch(plt.Polygon(xy=array2,color=colors[k[1]]))
        elif len(k)==3:
            array1=[[i-(1./2.),j-1],[i,j-1],[i,j-(1./2.)]]
            array2=[[i-1,j-(1./2.)],[i-1,j],[i-(1./2.),j]]
            ax.add_patch(matplotlib.patches.Rectangle(xy=(i-1,j-1),width=1,height=1,color=colors[k[1]]))
            ax.add_patch(plt.Polygon(xy=array1,color=colors[k[0]]))
            ax.add_patch(plt.Polygon(xy=array2,color=colors[k[2]]))
        elif len(k)>=4:
            array1=[[i-1,j-1],[i,j-1],[i,j]]
            array2=[[i-1,j-1],[i-1,j],[i,j]]
            array3=[[i-(1./2.),j-1],[i,j-1],[i,j-(1./2.)]]
            array4=[[i-1,j-(1./2.)],[i-1,j],[i-(1./2.),j]]
            ax.add_patch(plt.Polygon(xy=array1,color=colors[k[0]]))
            ax.add_patch(plt.Polygon(xy=array2,color=colors[k[1]]))
            ax.add_patch(plt.Polygon(xy=array3,color=colors[k[2]]))
            ax.add_patch(plt.Polygon(xy=array4,color=colors[k[3]]))
    #for i in range(1,13):
        #for j in range(1,32):
            #import random
            #ax.add_patch(matplotlib.patches.Rectangle(xy=(i-1,j-1),width=1,height=1,color=random.choice(colors)))
    xAxis=np.linspace(0,15,15)
    xLabels = ['January', 'Febuary', 'March', 'April',
                      'May','June','July','August','September',
                      'Oktober','November', 'December','','','']
    plt.xticks(xAxis, xLabels, rotation='vertical')
    plt.xlim((0,14))
    plt.ylim((0,31))
    plt.ylim((0,31))
    moods=["average", "sad", "boring", "angry", "active", "happy", "unsure", "sick"]
    for i, mood in enumerate(moods):
        plt.text(12.5,19-i,mood,color=colors[i])
    #ax.legend(["average", "sad", "boring", "angry", "active", "happy", "unsure", "sick"],loc="right")
    plt.subplots_adjust(bottom=0.175)
    SavePlot("%i/mood"%(year),tight=False)

def PlotPushups(dataClass,year=2016,show=False):

    plt.figure(figsize=(10,10))
    x=dataClass.training[year]["bin"]
    y=dataClass.training[year]["pushup"]
    plt.plot(x,y,linestyle='',marker="+")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.xlabel("Monat")
    plt.ylabel("pushups")
    SavePlot("%i/pushups"%(year))
    if show==True: plt.show()
    plt.close()

def PlotSleep(dataClass,year=2016,show=False):

    plt.figure(figsize=(10,10))
    x=dataClass.sleep[year]["bin"]
    y=dataClass.sleep[year]["sleeping"]
    plt.plot(x,y,linestyle='',marker="+")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.xlabel("Monat")
    plt.ylabel("sleeping hours")
    SavePlot("%i/sleeping"%(year))
    if show==True: plt.show()
    plt.close()
    plt.figure(figsize=(10,10))
    y=dataClass.sleep[year]["lying"]
    plt.plot(x,y,linestyle='',marker="+")
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.xlabel("Monat")
    plt.ylabel("time lying in bed")
    SavePlot("%i/lyingInBed"%(year))
    if show==True: plt.show()
    plt.close()
    plt.figure(figsize=(10,10))
    y=dataClass.sleep[year]["total"]
    plt.plot(x,y,linestyle='',marker="+")
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.xlabel("Monat")
    plt.ylabel("total time in bed")
    SavePlot("%i/totalBed"%(year))
    if show==True: plt.show()
    plt.close()

def PlotRHR(dataClass,year=2016,save=True,show=False):

    plt.figure(figsize=(10,10))
    x=dataClass.HR[year]["bin"]
    y=dataClass.HR[year]["rhr"]
    plt.plot(x,y,linestyle='',marker="+")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.xlabel("Monat")
    plt.ylabel("resting heartrate")
    if save==True: SavePlot("%i/rhr"%(year))
    if show==True: plt.show()
    plt.close()
    return x, y, xAxis, labels

def plotRouteVsLength(dataClass,year=2016,save=True,show=False):

    plt.figure(figsize=(10,10))
    x=dataClass.run[year]["route"]
    y=dataClass.run[year]["distance"]
    plt.plot(x,y,linestyle='',marker="+")
    xAxis,labels=getRouteLabels(x)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.xlabel("routes")
    plt.ylabel("distance")
    if save==True: SavePlot("%i/route_length"%(year))
    if show==True: plt.show()
    plt.close()

def plotRouteVsNumber(dataClass,year=2016,save=True,show=False):

    plt.figure(figsize=(10,10))
    inp=dataClass.run[year]["route"]
    dic_inp={}
    for i in inp:
        if i not in dic_inp.keys():
            dic_inp[i]=0
        else:
            dic_inp[i]+=1
    x=np.array([],dtype=int)
    y=np.array([],dtype=int)
    for key in dic_inp.keys():
        x=np.append(x,key)
    for val in dic_inp.values():
        y=np.append(y,val)
    plt.plot(x,y,linestyle='',marker="+")
    xAxis,labels=getRouteLabels(x)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.xlabel("routes")
    plt.ylabel("number of runs")
    if save==True: SavePlot("%i/route_number"%(year))
    if show==True: plt.show()
    plt.close()

def getRouteLabels(routeNrs):
    #print(np.genfromtxt("../../data/Strecken.txt",dtype=str))
    route=np.genfromtxt("../../data/Strecken.txt",dtype=str,delimiter=" ")
    routeNr=np.array([],dtype=int)
    routeName=np.array([],dtype=str)
    for i in route:
        routeNr=np.append(routeNr,int(i[0]))
        routeName=np.append(routeName,i[1])
    return routeNr,routeName

def MakeGymPlot(dataClass,year=2023,save=True,show=False):
    plt.figure(figsize=(10,10))
    x=dataClass.gym[year]["bin"]
    y=dataClass.gym[year]["boolean"]
    plt.plot(x,y,linestyle='',marker="+")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.xlabel("Monat")
    plt.ylabel("Been to Gym")
    if save==True: SavePlot("%i/gym"%(year))
    if show==True: plt.show()
    plt.close()

def MakeMonthlyGymPlot(dataClass,year=2023,save=True,show=False):
    plt.figure(figsize=(10,10))
    x=np.array([],dtype=int)
    y=np.array([],dtype=int)
    s=1
    for n in range(len(dataClass.gym[year]["boolean"])):
        i,j=dataClass.gym[year]["month"][n],dataClass.gym[year]["boolean"][n]
        if len(x)==0 or i!=x[-1]:
            x=np.append(x,i)
        if n==len(dataClass.gym[year]["month"])-1:
            s+=j
            y=np.append(y,s)
        elif  dataClass.gym[year]["month"][n]==dataClass.gym[year]["month"][n+1]:
            s+=j
        else:
            y=np.append(y,s)
            s=1
    possibleLabels = ['January', 'Febuary', 'March', 'April',
                      'May','June','July','August','September',
                      'Oktober','November', 'December']

    plt.plot(x,y,linestyle='',marker="+")
    xAxis,labels=makeLabels(x,year)
    lastMonth=int(dataClass.gym[year]["month"][-1])
    plt.xticks(x[:lastMonth], possibleLabels[:lastMonth], rotation='vertical')
    plt.xlabel("Monat")
    plt.ylabel("Been to Gym")
    if save==True: SavePlot("%i/gymMonthly"%(year))
    if show==True: plt.show()
    plt.close()

def makeVo2MaxPlot(dataClass,year=2023,save=True,show=False):

    plt.figure(figsize=(10,10))
    x=[x for x, y in zip(dataClass.run[year]["bin"], dataClass.run[year]["vo2max"]) if y!=-1]
    y=[y for y in dataClass.run[year]["vo2max"] if y!=-1]
    plt.plot(x,y,linestyle='',marker="+")
    xAxis,labels=makeLabels(x,year)
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.xlabel("Monat")
    plt.ylabel("VO2 Max")
    SavePlot("%i/vo2max"%(year))
    if show==True: plt.show()
    plt.close()

if __name__=="__main__":
    pass
