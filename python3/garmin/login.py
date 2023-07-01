import re
import cloudscraper
import json
from idbutils import RestClient, RestResponseException, RestException
from helper.setupParams import setupParams
from helper.readConfig import setupConfig, getUser, getPW


def login(session, restClient, ssoClient, jsonFile):

    params=setupParams(restClient, ssoClient)
    get_headers = {'Referer': params["source"]}

    ssoLogin='signin'
    try:
        response = ssoClient.get(ssoLogin, get_headers, params)
    except RestResponseException as e:
        print("Exception during login get: %s", e)
        RestClient.save_binary_file('login_get.html', e.response)
        return False
    found = re.search(r"name=\"_csrf\" value=\"(\w*)", response.text, re.M)
    if not found:
        print("_csrf not found: %s", response.status_code)
        RestClient.save_binary_file('login_get.html', response)
        return False


    data = {
        'username'  : getUser(jsonFile),
        'password'  : getPW(jsonFile),
        'embed'     : 'false',
        '_csrf'     : found.group(1)
    }
    post_headers = {
        'Referer'       : response.url,
        'Content-Type'  : 'application/x-www-form-urlencoded'
    }
    try:
        response = ssoClient.post(ssoLogin, post_headers, params, data)
    except RestException as e:
        root_logger.error("Exception during login post: %s", e)
        return False
    found = re.search(r"\?ticket=([\w-]*)", response.text, re.M)
    if not found:
        logger.error("Login ticket not found (%d).", response.status_code)
        RestClient.save_binary_file('login_post.html', response)
        return False
    foundParams = {
        'ticket' : found.group(1)
    }
    try:
        response = restClient.get(ssoLogin, params=foundParams)
    except RestException:
        print("Login get homepage failed (%d).", response.status_code)
        RestClient.save_binary_file('login_home.html', response)
        return False
    userPrefs = getJsonFromHTML(response.text, 'VIEWER_USERPREFERENCES')
    #if profile_dir:
        #restClient.save_json_to_file(f'{profile_dir}/profile.json', userPrefs)
    with open("garminData/displayName.txt","w+") as f:
        f.write(userPrefs['displayName'])
    #self.display_name = self.user_prefs['displayName']
    #self.social_profile = self.__get_json(response.text, 'VIEWER_SOCIAL_PROFILE')
    #self.full_name = self.social_profile['fullName']
    #root_logger.info("login: %s (%s)", self.full_name, self.display_name)
    return True

def getJsonFromHTML(page_html, key):
    found = re.search(key + r" = (\{.*\});", page_html, re.M)
    if found:
        json_text = found.group(1).replace('\\"', '"')
        return json.loads(json_text)

if __name__=="__main__":
    session=cloudscraper.CloudScraper()
    restClient=RestClient(session, 'connect.garmin.com', 'modern', aditional_headers={'NK': 'NT'})
    ssoClient=RestClient(session, 'sso.garmin.com', 'sso', aditional_headers={'NK': 'NT'})
    jsonFile=setupConfig("GarminConnectConfig.json", "garminData/")
    login(session, restClient, ssoClient, jsonFile)
