def setupConfig(configFileName, dirName):
    import json
    import os
    assert os.path.exists(configFileName), "Config file doesnt exist"
    with open(configFileName, "r") as f:
        jsonFile=json.load(f)
    for variable in ["user", "password"]:
        assert os.path.exists(f'{dirName}/{variable}.txt'), f'file {dirName}/{variable}.txt doesnt exist'
        with open(f'{dirName}/{variable}.txt',"r") as f:
            for line in f:
                jsonFile["credentials"][variable]=line.replace("\n","")
    return jsonFile

def getPW(jsonFile):
    return jsonFile["credentials"]["password"]
def getUser(jsonFile):
    return jsonFile["credentials"]["user"]
def getDate(jsonFile):
    return jsonFile["data"]["start_date"]
def isEnabled(jsonFile, statName):
    return jsonFile["enabled_stats"][statName]
def getCount(jsonFile, allActivities=False):
    if allActivities:
        return jsonFile["data"]["download_latest_activities"]
    else:
        return jsonFile["data"]["download_all_activities"]

if __name__=="__main__":
    import os
    jsonFile=setupConfig(os.path.join(os.getcwd(), "..")+"/GarminConnectConfig.json", "../garminData")
    getUser(jsonFile)
    getPW(jsonFile)
    getDate(jsonFile)
    isEnabled(jsonFile, "dailySummary")
    getCount(jsonFile, False)
    getCount(jsonFile, True)
