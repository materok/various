import json
import glob
from datetime import timedelta, datetime
import os
from tqdm import tqdm

def reformatRuns():
    print("reformatRuns")
    for fileName in tqdm(glob.glob("garminData/activities/*json")):
        with open(fileName, "r") as f:
            for line in f:
                jsonFile=json.loads(line.replace("\n",""))
        if jsonFile["activityType"]["typeKey"] == "running" or jsonFile["activityType"]["typeKey"] == "treadmill_running":
            activityID=jsonFile["activityId"]
            distance=round(jsonFile["distance"])
            distance=f'{distance:5.0f}'.replace(" 0"," ")
            duration=jsonFile["duration"]
            mins=f'{int(duration//60):3.0f}'.replace(" 0"," ")
            secs=f'{round(duration-duration//60*60):2.0f}'
            avgHR=int(jsonFile["averageHR"])
            maxHR=int(jsonFile["maxHR"])
            elevationGain=jsonFile["elevationGain"]
            if elevationGain is None:
                elevationGain=-1
            elevationGain=f'{int(round(elevationGain)):3.0f}'
            currentYear=datetime.now().year-2000
            runDate=datetime.strptime(jsonFile["startTimeGMT"], "%Y-%m-%d %H:%M:%S")
            currentYear=runDate.year
            runMonth=f'{runDate.month:2.0f}'.replace(" 0"," ")
            runDay=f'{runDate.day:2.0f}'.replace(" 0", " ")
            vo2max=jsonFile["vO2MaxValue"]
            if vo2max is None:
                vo2max=-1
            else:
                vo2max=int(vo2max)
            if os.path.exists(f'../../data/dataFromGarmin{currentYear}.txt'):
                with open(f'../../data/dataFromGarmin{currentYear}.txt', "r") as f:
                    for line in f:
                        alreadyInFile=str(activityID)==line.split("activityID: ")[-1].replace("\n","")
                        if alreadyInFile:
                            break
            else:
                alreadyInFile=False
            if not alreadyInFile:
                with open(f'../../data/dataFromGarmin{currentYear}.txt', "a") as f:
                    f.write(f'{mins} {secs} {distance} {avgHR} {maxHR} {elevationGain} {runDay} {runMonth} {vo2max}#activityID: {activityID}\n')
        elif jsonFile["activityType"]["typeKey"] == "cycling":
            pass
            #print("this is a cycling activity")
        else:
            pass
            #print(jsonFile["activityType"]["typeKey"])

def reformatWeight():
    print("reformatWeight")
    for fileName in tqdm(glob.glob("garminData/weight/*json")):
        with open(fileName, "r") as f:
            for line in f:
                jsonFile=json.loads(line.replace("\n",""))
        dateStr=jsonFile["startDate"]
        date=datetime.strptime(dateStr, "%Y-%m-%d")
        if not date.date()==datetime.now().date():
            weight=jsonFile["totalAverage"]["weight"]
            if weight is not None:
                weight=int(weight/100)
                currentYear=datetime.now().year-2000
                currentYear=date.year
                month=f'{date.month:2.0f}'.replace(" 0"," ")
                day=f'{date.day:2.0f}'.replace(" 0", " ")
                if os.path.exists(f'../../data/statsFromGarmin{currentYear}.txt'):
                    with open(f'../../data/statsFromGarmin{currentYear}.txt', "r") as f:
                        for line in f:
                            alreadyInFile=dateStr==line.split("date: ")[-1].replace("\n","")
                            if alreadyInFile:
                                break
                else:
                    alreadyInFile=False
                if not alreadyInFile:
                    with open(f'../../data/statsFromGarmin{currentYear}.txt', "a") as f:
                        f.write(f'{weight} {day} {month}#date: {dateStr}\n')

def reformatRHR():
    print("reformatRHR")
    for fileName in tqdm(glob.glob("garminData/dailySummaries/*json")):
        with open(fileName, "r") as f:
            for line in f:
                jsonFile=json.loads(line.replace("\n",""))
        dateStr=jsonFile['calendarDate']
        date=datetime.strptime(dateStr, "%Y-%m-%d")
        if not date.date()==datetime.now().date():
            currentYear=date.year
            month=f'{date.month:2.0f}'.replace(" 0"," ")
            day=f'{date.day:2.0f}'.replace(" 0", " ")
            minHR=jsonFile["minHeartRate"]
            maxHR=jsonFile["maxHeartRate"]
            rhr=jsonFile["restingHeartRate"]
            if os.path.exists(f'../../data/hrDataFromGarmin{currentYear}.txt'):
                with open(f'../../data/hrDataFromGarmin{currentYear}.txt', "r") as f:
                    for line in f:
                        alreadyInFile=dateStr==line.split("date: ")[-1].replace("\n","")
                        if alreadyInFile:
                            break
            else:
                alreadyInFile=False
            if not alreadyInFile:
                with open(f'../../data/hrDataFromGarmin{currentYear}.txt', "a") as f:
                    f.write(f'{rhr} {minHR} {maxHR} {day} {month}#date: {dateStr}\n')

def reformat():
    reformatRuns()
    reformatWeight()
    reformatRHR()

if __name__=="__main__":
    import downloadData
    downloadData.downloadData()
    reformat()
