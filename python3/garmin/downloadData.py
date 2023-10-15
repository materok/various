import datetime
from login import login, login_garth
import cloudscraper
from idbutils import RestException
from garth import Client as GarthClient
from helper.readConfig import setupConfig, getDate, isEnabled, getCount
from helper.readConfig import getUser, getPW, getDomain
from time import sleep
from tqdm import tqdm
import os
import re
import json

#https://github.com/tcgoetz/GarminDB

def downloadData(downloadAll=False):

    basedir="garminData/"
    jsonFile=setupConfig("GarminConnectConfig.json", basedir)

    session=cloudscraper.CloudScraper()

    garth=GarthClient(session=session)
    garth.configure(domain=getDomain(jsonFile))
    garth.login(getUser(jsonFile), getPW(jsonFile))
    garmin_headers = {'NK': 'NT'}

    isAnyEnabled = True in [isEnabled(jsonFile, metric) for metric in jsonFile["enabled_stats"]]
    if isAnyEnabled:
        garmin_headers['Authorization'] = str(garth.oauth2_token)

        if isEnabled(jsonFile, "dailySummary"): getDailySummary(jsonFile, basedir, garth, downloadAll)
        if isEnabled(jsonFile, "weight"): getWeight(jsonFile, basedir, garth, downloadAll)
        if isEnabled(jsonFile, "activities"):
            directory=basedir+"activities/"
            count=getCount(jsonFile, downloadAll)
            get_activities(garth, directory, count)

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



def getDailySummary(jsonFile, basedir, garth, downloadAll):
    subdir="dailySummaries"
    datesToDownload=determineDatesToDownload(jsonFile, basedir, subdir, downloadAll)
    if len(datesToDownload)!=0:
        with open(f'{basedir}displayName.txt', "r") as f:
            for line in f:
                displayName=line
        for date in tqdm(datesToDownload, desc=f'downloading {subdir}'):
            if downloadDailySummary(garth, date, displayName, basedir, subdir):
                writeLastDate(basedir, subdir, date)
            sleep(0.25)

def downloadDailySummary(garth, date, displayName, basedir, subdir, overwrite=False):
    date_str = date.strftime('%Y-%m-%d')
    summaryParams = {
        'calendarDate': date_str,
        '_': str(dt_to_epoch_ms(date_to_dt(date)))
    }
    url = f'usersummary-service/usersummary/daily/{displayName}'
    json_filename = f'{basedir}/{subdir}/daily_summary_{date_str}'
    try:
        save_json_to_file(json_filename, garth.connectapi(url, params=summaryParams), overwrite)
        return True
    except RestException as e:
        print("Exception getting daily summary: %s", e)
        return False


def save_json_to_file(filename, json_data, overwrite=False):
    """Save JSON formatted data to a file."""
    full_filename = f'{filename}.json'
    exists = os.path.isfile(full_filename)
    if not exists or overwrite:
        with open(full_filename, 'w') as file:
            file.write(json.dumps(json_data))


def getWeight(jsonFile, basedir, garth, downloadAll):
    subdir="weight"
    datesToDownload=determineDatesToDownload(jsonFile, basedir, subdir, downloadAll)
    if len(datesToDownload)!=0:
        for date in tqdm(datesToDownload, desc=f'downloading {subdir}'):
            if downloadWeight(garth, date, basedir, subdir):
                writeLastDate(basedir, subdir, date)
            sleep(0.25)

def downloadWeight(garth, date, basedir, subdir, overwrite=False):
    date_str = date.strftime('%Y-%m-%d')
    weightParams = {
        'startDate' : date_str,
        'endDate'   : date_str,
        '_'         : str(dt_to_epoch_ms(date_to_dt(date)))
    }
    url = "weight-service/weight/dateRange"
    json_filename = f'{basedir}/{subdir}/weight_{date_str}'
    try:
        save_json_to_file(json_filename, garth.connectapi(url, params=weightParams), overwrite)
        return True
    except RestException as e:
        print("Exception getting weight: %s", e)
        return False


def getActivitySummaries(garth, start, count):
    activitySummaryParams = {
        'start' : str(start),
        "limit" : str(count)
    }
    url = "activitylist-service/activities/search/activities"
    try:
        return garth.connectapi(url, params=activitySummaryParams)
    except RestException as e:
        print("Exception getting activity summary: %s", e)

def get_activities(restClient, directory, count, overwrite=False):
    """Download activities files from Garmin Connect and save the raw files."""
    activities = getActivitySummaries(restClient, 0, count)
    for activity in tqdm(activities or [], unit='activities'):
        activity_id_str = str(activity['activityId'])
        activity_name_str = printable(activity['activityName'])
        json_filename = f'{directory}/activity_{activity_id_str}.json'
        if not os.path.isfile(json_filename) or overwrite:
            save_json_to_file(json_filename, activity)
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
