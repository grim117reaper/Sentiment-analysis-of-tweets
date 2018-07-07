#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

from pycorenlp import StanfordCoreNLP
import tweepy

from datetime import datetime, timedelta,timezone

consumer_key = 'KRDXSSEM1LP2lgBT1gqNKlm8h'
consumer_secret = 'hpLgyILXSMruIXT1qU8gJ9BdkW1OEdqTBlRc0z8eM9GTIO8gpl'

access_token = '980691074770386946-VWN666WzHwjgT8Z1UQU67TMvW6jFomX'
access_token_secret = 'otTq5F2qjbCzFC4o93bKAf35U0S4wKEafGJ6nM0i8kQBP'

auth = tweepy.OAuthHandler(consumer_key , consumer_secret)
auth.set_access_token(access_token , access_token_secret)
tw = open("t1.txt" , "w")

api = tweepy.API(auth,wait_on_rate_limit=True , wait_on_rate_limit_notify=True)
print('after auth')

search_query = input()
last_hour_date_time = datetime.now(timezone.utc)
last_hour_date_time = last_hour_date_time.replace(tzinfo=None)
last_hour_date_time = last_hour_date_time - timedelta(hours = 1)
average = []
#geo code of delhi , london , new york , mumbai
geoloc = ["28.7117544,77.4514209,100km","51.4592902,-0.2901918,100km","42.665396,-76.0788017,100km","19.0825223,72.7410985,100km"]
n = 0
for i in geoloc:
    search_query = tweepy.Cursor(api.search,q=search_query,geocode = i ,include_entities=True,lang="en").items()
    print ('after search')
    nlp = StanfordCoreNLP('http://localhost:9000')
    c = 0
    len_s = 0
    score_s = 0

    for tweet in search_query:
        if (tweet.created_at > last_hour_date_time):
            result = nlp.annotate(str(tweet.text),
                               properties={
                                   'annotators': 'sentiment',
                                   'outputFormat': 'json'
                               })
            c = c + 1
            for s in result["sentences"]:
                z = ("%d: '%s': %s %s" % (
                    s["index"],
                    " ".join([t["word"] for t in s["tokens"]]),
                    s["sentimentValue"], s["sentiment"]))
                z = z.encode('utf-8')
                tw.write(str(z))
                tw.write('\n\n')
                len_s = len_s + 1
                if (s["sentiment"] == 'Negative'):
                    score_s = score_s - (1 * int(s["sentimentValue"]))
                elif (s["sentiment"] == 'Neutral'):
                    score_s = score_s - (0 * int(s["sentimentValue"]))
                else:
                    score_s = score_s + (1 * int(s["sentimentValue"]))
        else:
            break

    if (c!= 0):
        print ("tweets retrived = " , c)
    else:
        print("no tweets retrieved")
        average.append(0)
        continue

    average.append(score_s / len_s)
    print("average score of geocode ",i,"= " , average[n])
    b = 'average sentiment score = %s ', str(average[n])
    tw.write(str(b))
    n = n + 1
    print('\n-------------\n--------------\n')


for i in range(0,4):
    b = 'average score of geolocation ' , geoloc[i] , 'is = ' , str(average[i])
    tw.write(str(b))
    tw.write('\n\n')

print('operation done')
tw.close()