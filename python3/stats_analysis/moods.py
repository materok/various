import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def SavePlot(title,savepng=True,tight=True):
    title=title.replace("_","")
    if tight: plt.tight_layout()
    plt.savefig("../../plots/"+title+".pdf")
    if savepng: plt.savefig("../../plots/"+title+".png")

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

class data:
    def __init__(self,first_year,last_year):
        print("ensure there is more than 1 entry in each file")
        self.first_year=first_year
        self.last_year=last_year
        for year in range(first_year, last_year+1):
            filename="../../data/mood"+str(year-2000)+".txt"
            if year>=2019: self.loadMoodData(filename)
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

def main(version=17):

    d=data(2019,2020)
    makeMoodPlot(data,2019)


if __name__=="__main__":
    main()
