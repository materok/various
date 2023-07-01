import cloudscraper
from idbutils import RestClient
def setupParams(restClient, ssoClient):
    return  {
        'service'                           : restClient.url(),
        'webhost'                           : "https://connect.garmin.com",
        'source'                            : "https://connect.garmin.com/en-US/signin",
        'redirectAfterAccountLoginUrl'      : restClient.url(),
        'redirectAfterAccountCreationUrl'   : restClient.url(),
        'gauthHost'                         : ssoClient.url(),
        'locale'                            : 'en_US',
        'id'                                : 'gauth-widget',
        'cssUrl'                            : 'https://static.garmincdn.com/com.garmin.connect/ui/css/gauth-custom-v1.2-min.css',
        'privacyStatementUrl'               : '//connect.garmin.com/en-US/privacy/',
        'clientId'                          : 'GarminConnect',
        'rememberMeShown'                   : 'true',
        'rememberMeChecked'                 : 'false',
        'createAccountShown'                : 'true',
        'openCreateAccount'                 : 'false',
        'displayNameShown'                  : 'false',
        'consumeServiceTicket'              : 'false',
        'initialFocus'                      : 'true',
        'embedWidget'                       : 'false',
        'generateExtraServiceTicket'        : 'true',
        'generateTwoExtraServiceTickets'    : 'false',
        'generateNoServiceTicket'           : 'false',
        'globalOptInShown'                  : 'true',
        'globalOptInChecked'                : 'false',
        'mobile'                            : 'false',
        'connectLegalTerms'                 : 'true',
        'locationPromptShown'               : 'true',
        'showPassword'                      : 'true'
    }

if __name__=="__main__":
    session=cloudscraper.CloudScraper()
    restClient=RestClient(session, 'connect.garmin.com', 'modern', aditional_headers={'NK': 'NT'})
    ssoClient=RestClient(session, 'sso.garmin.com', 'sso', aditional_headers={'NK': 'NT'})
    setupParams(restClient, ssoClient)
