import tweepy

'''
Función utilizada para utilizar la Api de Twitter
'''
def get_auth():
    consumer_key = 'x'
    consumer_secret = 'x'
    access_token = 'x'
    access_token_secret = 'x'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    return auth