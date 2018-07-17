#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

from pycorenlp import StanfordCoreNLP
import tweepy

import pickle
import nltk

from datetime import datetime, timedelta,timezone


#classification
classifier = pickle.load(open('trained/NAIVE.pickle', 'rb'))
word_features = pickle.load(open('trained/word_features.pickle', 'rb'))
def document_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

def predict_topic(s):
	token = nltk.word_tokenize(s.lower())
	return classifier.classify(document_features(token))
#classification ends


consumer_key = 'xxxxxxxxxxxxxxxxx'
consumer_secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

access_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
access_token_secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

auth = tweepy.OAuthHandler(consumer_key , consumer_secret)
auth.set_access_token(access_token , access_token_secret)


tw = open("tweets_sentiment_analysis.txt" , "w")


api = tweepy.API(auth,wait_on_rate_limit=True , wait_on_rate_limit_notify=True)
print('after auth')


query = input()
last_hour_date_time = datetime.now(timezone.utc)
last_hour_date_time = last_hour_date_time.replace(tzinfo=None)
last_hour_date_time = last_hour_date_time - timedelta(hours = 1)
average = []


#geo code of delhi , london , new york , mumbai
geoloc = ["28.7117544,77.4514209,100km","51.4592902,-0.2901918,100km","42.665396,-76.0788017,100km","19.0825223,72.7410985,100km"]
n = 0


for i in geoloc:
	search_query = tweepy.Cursor(api.search,q=query,geocode = i ,include_entities=True,lang="en").items()
	print ('after search')
	tw.write(str(i))
	tw.write('\n')
	tw.write('--------------------------------------------------------------------------------------------------')
	tw.write('\n')
	nlp = StanfordCoreNLP('http://localhost:9000')
	c = 0
	len_s = 0
	score_s = 0

	for tweet in search_query:
		if (tweet.created_at > last_hour_date_time):
			topic = predict_topic(str(tweet.text))
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
				tw.write('\n')
				z = "topic = %s" %(topic)
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
	b =("average sentiment score =  %s" %(str(average[n])))
	tw.write(str(b))
	tw.write('\n\n\n')
	n = n + 1
	print('\n-------------\n--------------\n')


n = len(geoloc)

tw.write('\n\n\n\n\n')
tw.write('---------------------------------------------------------------------')
for i in range(0,n):
	b = ('average score of geolocation ' + str(geoloc[i]) +' is ' + str(average[i]))
	tw.write(str(b))
	tw.write('\n\n')


print('operation done')


tw.close()
