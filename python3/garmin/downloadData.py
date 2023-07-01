import datetime
from login import login
import cloudscraper
from idbutils import RestClient, RestException
from mm.readConfig import setupConfig, getDate
from time import sleep
from tqdm import tqdm
import os

def downloadData(downloadAll=False):

    session=cloudscraper.CloudScraper()
    restClient=RestClient(session, 'connect.garmin.com', 'modern', aditional_headers={'NK': 'NT'})
    ssoClient=RestClient(session, 'sso.garmin.com', 'sso', aditional_headers={'NK': 'NT'})
    basedir="garminData/"
    jsonFile=setupConfig(basedir)
    startDate=datetime.datetime.strptime(getDate(jsonFile), "%d/%m/%Y")
    if not downloadAll and os.path.exists(f'{basedir}/dailySummaries/lastDate.txt'):
        with open(f'{basedir}/dailySummaries/lastDate.txt', "r") as f:
            for line in f:
                startDate=datetime.datetime.strptime(line.replace("\n", ""), "%d/%m/%Y")-datetime.timedelta(days=1)
    datesToDownload = [startDate+i*datetime.timedelta(days=1) for i in range(0,(datetime.datetime.now()-startDate).days+1)]
    if len(datesToDownload)!=0:
        if login(session, restClient, ssoClient, jsonFile):
            with open(f'{basedir}displayName.txt', "r") as f:
                for line in f:
                    displayName=line
        for date in tqdm(datesToDownload, desc="downloading dailySummaries"):
            get_summary_day(restClient, date, displayName, basedir)
            with open(f'{basedir}/dailySummaries/lastDate.txt', "w+") as f:
                f.write(date.strftime("%d/%m/%Y"))
            sleep(0.25)

def get_summary_day(restClient, date, displayName, basedir, overwrite=False):
    print("get_summary_day: %s", date)
    date_str = date.strftime('%Y-%m-%d')
    summaryParams = {
        'calendarDate': date_str,
        '_': str(dt_to_epoch_ms(date_to_dt(date)))
    }
    url = f'proxy/usersummary-service/usersummary/daily/{displayName}'
    json_filename = f'{basedir}/dailySummaries/daily_summary_{date_str}'
    try:
        restClient.download_json_file(url, json_filename, overwrite, summaryParams)
    except RestException as e:
        print("Exception getting daily summary: %s", e)


def date_to_dt(date):
    """Given a datetime date object, return a date time datetime object for the given date at 00:00:00."""
    return datetime.datetime.combine(date, datetime.time.min)

def dt_to_epoch_ms(dt):
    """Convert a datetime object to milliseconds since the epoch."""
    return int((dt - datetime.datetime.fromtimestamp(0)).total_seconds() * 1000)

if __name__== "__main__":
    downloadData()
