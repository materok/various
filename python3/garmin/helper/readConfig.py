def setupConfig(dirName):
    import json
    import os
    assert os.path.exists(os.getcwd()+"/GarminConnectConfig.json"), "Config file doesnt exist"
    with open(os.getcwd()+"/GarminConnectConfig.json", "r") as f:
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
    return jsonFile["data"]["monitoring_start_date"]

if __name__=="__main__":
    jsonFile=setupConfig("../../../data/garmin")
    getUser(jsonFile)
    getPW(jsonFile)
    print(getDate(jsonFile))
