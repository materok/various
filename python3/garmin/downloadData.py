import datetime
from login import login
import cloudscraper
from idbutils import RestClient, RestException
from helper.readConfig import setupConfig, getDate, isEnabled, getCount
from time import sleep
from tqdm import tqdm
import os
#import tempfile
import re
#import zipfile

def downloadData(downloadAll=False):

    session=cloudscraper.CloudScraper()
    restClient=RestClient(session, 'connect.garmin.com', 'modern', aditional_headers={'NK': 'NT'})
    ssoClient=RestClient(session, 'sso.garmin.com', 'sso', aditional_headers={'NK': 'NT'})
    basedir="garminData/"
    jsonFile=setupConfig("GarminConnectConfig.json", basedir)
    isAnyEnabled = True in [isEnabled(jsonFile, metric) for metric in jsonFile["enabled_stats"]]
    if isAnyEnabled:
        if login(session, restClient, ssoClient, jsonFile):
            if isEnabled(jsonFile, "dailySummary"): getDailySummary(jsonFile, basedir, restClient, downloadAll)
            if isEnabled(jsonFile, "weight"): getWeight(jsonFile, basedir, restClient, downloadAll)
            if isEnabled(jsonFile, "activities"):
                activityClient = RestClient.inherit(restClient, "proxy/activity-service/activity")
                downloadClient = RestClient.inherit(restClient, "proxy/download-service/files")
                directory=basedir+"activities/"
                count=getCount(jsonFile, downloadAll)
                get_activities(restClient, activityClient, downloadClient, directory, count)

def determineDatesToDownload(jsonFile, basedir, subdir, downloadAll):
    startDate=datetime.datetime.strptime(getDate(jsonFile), "%d/%m/%Y")
    if not downloadAll and os.path.exists(f'{basedir}/{subdir}/lastDate.txt'):
        with open(f'{basedir}/{subdir}/lastDate.txt', "r") as f:
            for line in f:
                startDate=datetime.datetime.strptime(line.replace("\n", ""), "%d/%m/%Y")-datetime.timedelta(days=1)
    return [startDate+i*datetime.timedelta(days=1) for i in range(0,(datetime.datetime.now()-startDate).days+1)]

def writeLastDate(basedir, subdir, date):
    with open(f'{basedir}/{subdir}/lastDate.txt', "w+") as f:
        f.write(date.strftime("%d/%m/%Y"))

def getDailySummary(jsonFile, basedir, restClient, downloadAll):
    subdir="dailySummaries"
    datesToDownload=determineDatesToDownload(jsonFile, basedir, subdir, downloadAll)
    if len(datesToDownload)!=0:
        with open(f'{basedir}displayName.txt', "r") as f:
            for line in f:
                displayName=line
        for date in tqdm(datesToDownload, desc=f'downloading {subdir}'):
            if downloadDailySummary(restClient, date, displayName, basedir, subdir):
                writeLastDate(basedir, subdir, date)
            sleep(0.25)

def downloadDailySummary(restClient, date, displayName, basedir, subdir, overwrite=False):
    date_str = date.strftime('%Y-%m-%d')
    summaryParams = {
        'calendarDate': date_str,
        '_': str(dt_to_epoch_ms(date_to_dt(date)))
    }
    url = f'proxy/usersummary-service/usersummary/daily/{displayName}'
    json_filename = f'{basedir}/{subdir}/daily_summary_{date_str}'
    try:
        restClient.download_json_file(url, json_filename, overwrite, summaryParams)
        return True
    except RestException as e:
        print("Exception getting daily summary: %s", e)
        return False


def getWeight(jsonFile, basedir, restClient, downloadAll):
    subdir="weight"
    datesToDownload=determineDatesToDownload(jsonFile, basedir, subdir, downloadAll)
    if len(datesToDownload)!=0:
        for date in tqdm(datesToDownload, desc=f'downloading {subdir}'):
            if downloadWeight(restClient, date, basedir, subdir):
                writeLastDate(basedir, subdir, date)
            sleep(0.25)

def downloadWeight(restClient, date, basedir, subdir, overwrite=False):
    date_str = date.strftime('%Y-%m-%d')
    weightParams = {
        'startDate' : date_str,
        'endDate'   : date_str,
        '_'         : str(dt_to_epoch_ms(date_to_dt(date)))
    }
    url = "proxy/weight-service/weight/dateRange"
    json_filename = f'{basedir}/{subdir}/weight_{date_str}'
    try:
        restClient.download_json_file(url, json_filename, overwrite, weightParams)
        return True
    except RestException as e:
        print("Exception getting weight: %s", e)
        return False


def getActivitySummaries(restClient, start, count):
    activitySummaryParams = {
        'start' : str(start),
        "limit" : str(count)
    }
    url = "proxy/activitylist-service/activities/search/activities"
    try:
        response = restClient.get(url, params=activitySummaryParams)
        return response.json()
    except RestException as e:
        print("Exception getting activity summary: %s", e)

#def downloadActivityDetails(activityClient, directory, activity_id_str, overwrite):
    #json_filename = f'{directory}/activity_details_{activity_id_str}'
    #try:
        #activityClient.download_json_file(activity_id_str, json_filename, overwrite)
    #except RestException as e:
        #print("Exception getting daily summary %s", e)

#def downloadActivityFile(downloadClient, tempDir, activity_id_str):
    #print("ok")
    #zip_filename = f'{tempDir}/activity_{activity_id_str}.zip'
    #url = f'activity/{activity_id_str}'
    #try:
        #downloadClient.download_binary_file(url, zip_filename)
    #except RestException as e:
        #print("Exception downloading activity file: %s", e)

def get_activities(restClient, activityClient, downloadClient, directory, count, overwrite=False):
    """Download activities files from Garmin Connect and save the raw files."""
    #tempDir=tempfile.mkdtemp()
    activities = getActivitySummaries(restClient, 0, count)
    #for activity in tqdm(activities or [], unit='activities'):
        #activity_id_str = str(activity['activityId'])
        #activity_name_str = printable(activity['activityName'])
        #json_filename = f'{directory}/activity_{activity_id_str}.json'
        #if not os.path.isfile(json_filename) or overwrite:
            #downloadActivityDetails(activityClient, directory, activity_id_str, overwrite)
            #restClient.save_json_to_file(json_filename, activity)
            #if not os.path.isfile(f'{directory}/{activity_id_str}.fit') or overwrite:
                #downloadActivityFile(downloadClient, tempDir, activity_id_str)
                # pause for a second between every page access
        #sleep(1)
    #unzip_files(tempDir, directory)

#def unzip_files(tempDir, outdir):
    #"""Unzip and downloaded zipped files into the directory supplied."""
    #for filename in os.listdir(tempDir):
        #match = re.search(r'.*\.zip', filename)
        #if match:
            #full_pathname = f'{tempDir}/{filename}'
            #with zipfile.ZipFile(full_pathname, 'r') as files_zip:
                #try:
                    #files_zip.extractall(outdir)
                #except Exception as e:
                    #print('Failed to unzip %s to %s: %s', full_pathname, outdir, e)

def date_to_dt(date):
    """Given a datetime date object, return a date time datetime object for the given date at 00:00:00."""
    return datetime.datetime.combine(date, datetime.time.min)

def dt_to_epoch_ms(dt):
    """Convert a datetime object to milliseconds since the epoch."""
    return int((dt - datetime.datetime.fromtimestamp(0)).total_seconds() * 1000)

def printable(string_in):
    """Return a string with only prinatable characters given a string with potentially unprintable characters."""
    if string_in is not None:
        return filter(lambda x: x in string.printable, string_in)

if __name__== "__main__":
    downloadData()
