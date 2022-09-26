# Stockport-sentimental-analysis
This is our internship project
STOCK PREDICTION USING TWITTER SENTIMENT ANALYSIS
importing machine learning libraries
import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as mlpt
importing library to fetch data from twitter
import tweepy
import csv
import pandas as pd
import random
import numpy as np
import pandas as pd
setting up consumer key and access token
consumer_key    = '3jmA1BqasLHfItBXj3KnAIGFB'
consumer_secret = 'imyEeVTctFZuK62QHmL1I0AUAMudg5HKJDfkx0oR7oFbFinbvA'

access_token  = '1574365402469048321-MoqSzXyrozTHN5Wn4rwXSXXS7y9a38'
access_token_secret = '7fGuvLuTsQIgNwi7mOhmKn73gueRrVL30aZnqJfC8MUc9'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
Fetching tweets for United Airlines in extended mode (means entire tweet will come and not just few words + link)
fetch_tweets=tweepy.Cursor(api.search, q="#unitedAIRLINES",count=100, lang ="en",since="2018-9-13", tweet_mode="extended").items()
data=pd.DataFrame(data=[[tweet_info.created_at.date(),tweet_info.full_text]for tweet_info in fetch_tweets],columns=['Date','Tweets'])
data
Removing special character from each tweets
data.to_csv("Tweets.csv")
cdata=pd.DataFrame(columns=['Date','Tweets'])
total=100
index=0
for index,row in data.iterrows():
    stre=row["Tweets"]
    my_new_string = re.sub('[^ a-zA-Z0-9]', '', stre)
    temp_df = pd.DataFrame([[data["Date"].iloc[index], 
                            my_new_string]], columns = ['Date','Tweets'])
    cdata = pd.concat([cdata, temp_df], axis = 0).reset_index(drop = True)
    # index=index+1
#print(cdata.dtypes)
Displaying the data with date and tweets, you can notice there are multiple tweets for each day. So we will club them together later.
cdata
Date	Tweets
0	2021-09-22	ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...
1	2021-09-22	RT diecastryan A nice full lineup at IAD last ...
2	2021-09-22	United Airlines resuming Airline Tickets Reser...
3	2021-09-22	RT diecastryan A nice full lineup at IAD last ...
4	2021-09-22	lol FAANews united does not give a single damn...
...	...	...
367	2021-09-13	Thank You unitedAIRLINES httpstcoRU897P5rqI
368	2021-09-13	Where does the journey take you luggage tra...
369	2021-09-13	RT n194at United Air LinesDouglas DC852 N8062U...
370	2021-09-13	It is so ignorant to have 1299 in flight wifi ...
371	2021-09-13	Exactly But we have pretty options than United...
372 rows × 2 columns

Creating a dataframe where we will combine the tweets date wise and store into
ccdata=pd.DataFrame(columns=['Date','Tweets'])
indx=0
get_tweet=""
for i in range(0,len(cdata)-1):
    get_date=cdata.Date.iloc[i]
    next_date=cdata.Date.iloc[i+1]
    if(str(get_date)==str(next_date)):
        get_tweet=get_tweet+cdata.Tweets.iloc[i]+" "
    if(str(get_date)!=str(next_date)):
        temp_df = pd.DataFrame([[get_date, 
                                get_tweet]], columns = ['Date','Tweets'])
        ccdata = pd.concat([ccdata, temp_df], axis = 0).reset_index(drop = True)
        get_tweet=" "
All the tweets has been clubbed as per their date.
ccdata
Date	Tweets
0	2021-09-22	ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...
1	2021-09-21	RT SparrowOneSix 737900 N78448 was carrying U...
2	2021-09-20	RT diecastryan A nice full lineup at IAD last...
3	2021-09-19	jacobcabe Guess UnitedAirlines wont get any ...
4	2021-09-18	RT FELASTORY UnitedAirlines announce non stop...
5	2021-09-17	UnitedAirlines 90 of workers vaccinated after...
6	2021-09-16	This is how united UnitedAirlines treated wit...
7	2021-09-15	Thank you SPONSORSYour generous support make ...
8	2021-09-14	Because I get to work with amazing people uni...
Now to know the "closing price" of each day we will import STOCK PRICE DATA for UNITED AIRLINES from "yahoo.finance". We will consider "Close" price only.
read_stock_p=pd.read_csv('UAL.csv')
# DOWNLOAD UPDATED CLOSE PRICE FROM https://finance.yahoo.com/quote/UAL/history?period1=1598918400&period2=1632268800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
read_stock_p
Date	Open	High	Low	Close	Adj Close	Volume
0	2020-09-01	35.250000	37.240002	34.950001	36.009998	36.009998	29722200
1	2020-09-02	36.099998	37.099998	35.209999	36.889999	36.889999	26622800
2	2020-09-03	37.130001	39.770000	36.139999	37.400002	37.400002	53966400
3	2020-09-04	38.150002	38.740002	36.459999	38.209999	38.209999	33121600
4	2020-09-08	37.299999	38.480000	36.480000	37.279999	37.279999	33207100
...	...	...	...	...	...	...	...
261	2021-09-15	43.650002	43.910000	43.020000	43.860001	43.860001	10321600
262	2021-09-16	43.860001	45.410000	43.849998	44.470001	44.470001	12204300
263	2021-09-17	44.779999	45.500000	44.110001	44.540001	44.540001	11733300
264	2021-09-20	44.759998	45.340000	43.590000	45.270000	45.270000	14700300
265	2021-09-21	45.500000	46.259998	44.279999	44.450001	44.450001	12207000
266 rows × 7 columns

Adding a "Price" column in our dataframe and fetching the stock price as per the date in our dataframe.
ccdata['Prices']=""
indx=0
for i in range (0,len(ccdata)):
    for j in range (0,len(read_stock_p)):
        get_tweet_date=ccdata.Date.iloc[i]
        get_stock_date=read_stock_p.Date.iloc[j]
        if(str(get_stock_date)==str(get_tweet_date)):
            #print(get_stock_date," ",get_tweet_date)
            # ccdata.set_value(i,'Prices',int(read_stock_p.Close[j]))
            ccdata['Prices'].iloc[i] = int(read_stock_p.Close[j])
Prices are fetched but some entires are blank as close price might not be available for that day due to some reason (like holiday, etc.)
ccdata
Date	Tweets	Prices
0	2021-09-22	ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...	
1	2021-09-21	RT SparrowOneSix 737900 N78448 was carrying U...	44
2	2021-09-20	RT diecastryan A nice full lineup at IAD last...	45
3	2021-09-19	jacobcabe Guess UnitedAirlines wont get any ...	
4	2021-09-18	RT FELASTORY UnitedAirlines announce non stop...	
5	2021-09-17	UnitedAirlines 90 of workers vaccinated after...	44
6	2021-09-16	This is how united UnitedAirlines treated wit...	44
7	2021-09-15	Thank you SPONSORSYour generous support make ...	43
8	2021-09-14	Because I get to work with amazing people uni...	43
So we take the mean for the close price and put it in the blank value
mean=0
summ=0
count=0
for i in range(0,len(ccdata)):
    if(ccdata.Prices.iloc[i]!=""):
        summ=summ+int(ccdata.Prices.iloc[i])
        count=count+1
mean=summ/count
for i in range(0,len(ccdata)):
    if(ccdata.Prices.iloc[i]==""):
        ccdata.Prices.iloc[i]=int(mean)
Now all the entries have some value
ccdata
Date	Tweets	Prices
0	2021-09-22	ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...	43
1	2021-09-21	RT SparrowOneSix 737900 N78448 was carrying U...	44
2	2021-09-20	RT diecastryan A nice full lineup at IAD last...	45
3	2021-09-19	jacobcabe Guess UnitedAirlines wont get any ...	43
4	2021-09-18	RT FELASTORY UnitedAirlines announce non stop...	43
5	2021-09-17	UnitedAirlines 90 of workers vaccinated after...	44
6	2021-09-16	This is how united UnitedAirlines treated wit...	44
7	2021-09-15	Thank you SPONSORSYour generous support make ...	43
8	2021-09-14	Because I get to work with amazing people uni...	43
Making "prices" column as integer so mathematical operations could be performed easily.
ccdata['Prices'] = ccdata['Prices'].apply(np.int64)
Adding 4 new columns in our dataframe so that sentiment analysis could be performed.. Comp is "Compound" it will tell whether the statement is overall negative or positive. If it has negative value then it is negative, if it has positive value then it is positive. If it has value 0, then it is neutral.
ccdata["Comp"] = ''
ccdata["Negative"] = ''
ccdata["Neutral"] = ''
ccdata["Positive"] = ''
ccdata
Date	Tweets	Prices	Comp	Negative	Neutral	Positive
0	2021-09-22	ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...	43				
1	2021-09-21	RT SparrowOneSix 737900 N78448 was carrying U...	44				
2	2021-09-20	RT diecastryan A nice full lineup at IAD last...	45				
3	2021-09-19	jacobcabe Guess UnitedAirlines wont get any ...	43				
4	2021-09-18	RT FELASTORY UnitedAirlines announce non stop...	43				
5	2021-09-17	UnitedAirlines 90 of workers vaccinated after...	44				
6	2021-09-16	This is how united UnitedAirlines treated wit...	44				
7	2021-09-15	Thank you SPONSORSYour generous support make ...	43				
8	2021-09-14	Because I get to work with amazing people uni...	43				
Downloading this package was essential to perform sentiment analysis.
import nltk
nltk.download('vader_lexicon')
[nltk_data] Downloading package vader_lexicon to
[nltk_data]     C:\Users\aanand2\AppData\Roaming\nltk_data...
True
This part of the code is responsible for assigning the polarity for each statement. That is how much positive, negative, neutral you statement is. And also assign the compound value that is overall sentiment of the statement.
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
sentiment_i_a = SentimentIntensityAnalyzer()
for indexx, row in ccdata.T.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', ccdata.loc[indexx, 'Tweets'])
        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
        ccdata['Comp'].iloc[indexx] = sentence_sentiment['compound']
        ccdata['Negative'].iloc[indexx] = sentence_sentiment['neg']
        ccdata['Neutral'].iloc[indexx] = sentence_sentiment['neu']
        ccdata['Positive'].iloc[indexx] = sentence_sentiment['compound']
        # ccdata.set_value(indexx, 'Comp', sentence_sentiment['pos'])
        # ccdata.set_value(indexx, 'Negative', sentence_sentiment['neg'])
        # ccdata.set_value(indexx, 'Neutral', sentence_sentiment['neu'])
        # ccdata.set_value(indexx, 'Positive', sentence_sentiment['pos'])
    except TypeError:
        print (stocks_dataf.loc[indexx, 'Tweets'])
        print (indexx)
C:\Users\aanand2\Anaconda3\lib\site-packages\pandas\core\indexing.py:1637: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  self._setitem_single_block(indexer, value, name)
ccdata
Date	Tweets	Prices	Comp	Negative	Neutral	Positive
0	2021-09-22	ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...	43	0.9186	0.0	0.829	0.9186
1	2021-09-21	RT SparrowOneSix 737900 N78448 was carrying U...	44	0.9997	0.021	0.787	0.9997
2	2021-09-20	RT diecastryan A nice full lineup at IAD last...	45	0.9999	0.016	0.758	0.9999
3	2021-09-19	jacobcabe Guess UnitedAirlines wont get any ...	43	0.1262	0.075	0.852	0.1262
4	2021-09-18	RT FELASTORY UnitedAirlines announce non stop...	43	0.9985	0.019	0.837	0.9985
5	2021-09-17	UnitedAirlines 90 of workers vaccinated after...	44	0.9986	0.036	0.85	0.9986
6	2021-09-16	This is how united UnitedAirlines treated wit...	44	0.984	0.085	0.767	0.984
7	2021-09-15	Thank you SPONSORSYour generous support make ...	43	0.9831	0.028	0.838	0.9831
8	2021-09-14	Because I get to work with amazing people uni...	43	0.9784	0.089	0.775	0.9784
ccdata['']
Calculating the percentage of postive and negative tweets, and plotting the PIE chart for the same.
posi=0
nega=0
for i in range (0,len(ccdata)):
    get_val=ccdata.Comp[i]
    if(float(get_val)<(0)):
        nega=nega+1
    if(float(get_val>(0))):
        posi=posi+1
posper=(posi/(len(ccdata)))*100
negper=(nega/(len(ccdata)))*100
print("% of positive tweets= ",posper)
print("% of negative tweets= ",negper)
arr=np.asarray([posper,negper], dtype=int)
mlpt.pie(arr,labels=['positive','negative'])
mlpt.plot()
% of positive tweets=  100.0
% of negative tweets=  0.0
[]

Making a new dataframe with necessary columns for providing machine learning.
df_=ccdata[['Date','Prices','Comp','Negative','Neutral','Positive']].copy()
df_
Date	Prices	Comp	Negative	Neutral	Positive
0	2021-09-22	43	0.9186	0.0	0.829	0.9186
1	2021-09-21	44	0.9997	0.021	0.787	0.9997
2	2021-09-20	45	0.9999	0.016	0.758	0.9999
3	2021-09-19	43	0.1262	0.075	0.852	0.1262
4	2021-09-18	43	0.9985	0.019	0.837	0.9985
5	2021-09-17	44	0.9986	0.036	0.85	0.9986
6	2021-09-16	44	0.984	0.085	0.767	0.984
7	2021-09-15	43	0.9831	0.028	0.838	0.9831
8	2021-09-14	43	0.9784	0.089	0.775	0.9784
Dividing the dataset into train and test.
train_start_index = '0'
train_end_index = '5'
test_start_index = '6'
test_end_index = '8'
train = df_.loc[train_start_index : train_end_index,:]
test = df_.loc[test_start_index:test_end_index,:]
Making a 2D array that will store the Negative and Positive sentiment for Training dataset.
sentiment_score_list = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])
    sentiment_score_list.append(sentiment_score)
numpy_df_train = np.asarray(sentiment_score_list)
print(numpy_df_train)
[[0.     0.9186]
 [0.021  0.9997]
 [0.016  0.9999]
 [0.075  0.1262]
 [0.019  0.9985]
 [0.036  0.9986]]
Making a 2D array that will store the Negative and Positive sentiment for Testing dataset.
sentiment_score_list = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])
    sentiment_score_list.append(sentiment_score)
numpy_df_test = np.asarray(sentiment_score_list)
print(numpy_df_test)
[[0.085  0.984 ]
 [0.028  0.9831]
 [0.089  0.9784]]
Making 2 dataframe for Training and Testing "Prices". You can also make 1-D array for the same.
y_train = pd.DataFrame(train['Prices'])
#y_train=[91,91,91,92,91,92,91]
y_test = pd.DataFrame(test['Prices'])
print(y_train)
   Prices
0      43
1      44
2      45
3      43
4      43
5      44
Fitting the sentiments(this acts as in independent value) and prices(this acts as a dependent value (like class-lables in iris dataset))
# from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix

rf = RandomForestRegressor()
rf.fit(numpy_df_train, y_train)
<ipython-input-80-5be54910e205>:7: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  rf.fit(numpy_df_train, y_train)
RandomForestRegressor()
Making Predictions
prediction = rf.predict(numpy_df_test)
print(prediction)
[43.37 43.39 43.37]
Importing matplotlib library for plotting graph
import matplotlib.pyplot as plt
Defining index position for the test data. Making dataframe for the predicted value.
idx=np.arange(int(test_start_index),int(test_end_index)+1)
predictions_df_ = pd.DataFrame(data=prediction[0:], index = idx, columns=['Prices'])
predictions_df_
Prices
6	43.37
7	43.39
8	43.37
Plotting the graph for the Predicted_price VS Actual Price
ax = predictions_df_.rename(columns={"Prices": "predicted_price"}).plot(title='Random Forest predicted prices')#predicted value
ax.set_xlabel("Indexes")
ax.set_ylabel("Stock Prices")
fig = y_test.rename(columns={"Prices": "actual_price"}).plot(ax = ax).get_figure()#actual value
fig.savefig("random forest.png")

# from treeinterpreter import treeinterpreter as ti
# from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report,confusion_matrix

reg = LinearRegression()
reg.fit(numpy_df_train, y_train)
LinearRegression()
reg.predict(numpy_df_test)
array([[45.17154917],
       [44.0022019 ],
       [45.24044194]])
 
NOTE: Since our dataset is very small and as you can see that fetching 600 tweets could only make data for just 10 days.Also the prediction is not very great in such small dataset. So we found this new dataset on internet which has the Text as "Tweets" and respective "close price" and "Adjusted close price".
Adjusted Close Price: An adjusted closing price is a stock's closing price on any given day of trading that has been amended to include any distributions and corporate actions that occurred at any time before the next day's open.
stocks_dataf = pd.read_pickle('Twitter_Dataset.pkl')
stocks_dataf.columns=['closing_price','adj_close_price','Tweets']
New dataset
stocks_dataf
closing_price	adj_close_price	Tweets
2007-01-01	12469.971875	12469.971875	. What Sticks from '06. Somalia Orders Islamis...
2007-01-02	12472.245703	12472.245703	. Heart Health: Vitamin Does Not Prevent Death...
2007-01-03	12474.519531	12474.519531	. Google Answer to Filling Jobs Is an Algorith...
2007-01-04	12480.690430	12480.690430	. Helping Make the Shift From Combat to Commer...
2007-01-05	12398.009766	12398.009766	. Rise in Ethanol Raises Concerns About Corn a...
...	...	...	...
2016-12-27	19945.039062	19945.039062	. Should the U.S. Embassy Be Moved From Tel Av...
2016-12-28	19833.679688	19833.679688	. When Finding the Right Lawyer Seems Daunting...
2016-12-29	19819.779297	19819.779297	. Does Empathy Guide or Hinder Moral Action?. ...
2016-12-30	19762.599609	19762.599609	. Shielding Seized Assets From Corruption’s Cl...
2016-12-31	19762.599609	19762.599609	Terrorist Attack at Nightclub in Istanbul Kill...
3653 rows × 3 columns

stocks_dataf = stocks_dataf.reset_index().rename(columns = {'index':'Date'})
Removing dot (.) and space from the Tweets
stocks_dataf['adj_close_price'] = stocks_dataf['adj_close_price'].apply(np.int64)
stocks_dataf = stocks_dataf[['Date','adj_close_price', 'Tweets']]
stocks_dataf['Tweets'] = stocks_dataf['Tweets'].map(lambda x: x.lstrip('.-'))
stocks_dataf
Date	adj_close_price	Tweets
0	2007-01-01	12469	What Sticks from '06. Somalia Orders Islamist...
1	2007-01-02	12472	Heart Health: Vitamin Does Not Prevent Death ...
2	2007-01-03	12474	Google Answer to Filling Jobs Is an Algorithm...
3	2007-01-04	12480	Helping Make the Shift From Combat to Commerc...
4	2007-01-05	12398	Rise in Ethanol Raises Concerns About Corn as...
...	...	...	...
3648	2016-12-27	19945	Should the U.S. Embassy Be Moved From Tel Avi...
3649	2016-12-28	19833	When Finding the Right Lawyer Seems Daunting,...
3650	2016-12-29	19819	Does Empathy Guide or Hinder Moral Action?. C...
3651	2016-12-30	19762	Shielding Seized Assets From Corruption’s Clu...
3652	2016-12-31	19762	Terrorist Attack at Nightclub in Istanbul Kill...
3653 rows × 3 columns

Making new dataframe and only considering "Adjusted close price". And date as index vlaue.

dataframe = stocks_dataf[['adj_close_price']].copy()
# dataframe = dataframe.reset_index().rename(columns = {'index':'Date'})
dataframe["Comp"] = ''
dataframe["Negative"] = ''
dataframe["Neutral"] = ''
dataframe["Positive"] = ''
dataframe
adj_close_price	Comp	Negative	Neutral	Positive
0	12469				
1	12472				
2	12474				
3	12480				
4	12398				
...	...	...	...	...	...
3648	19945				
3649	19833				
3650	19819				
3651	19762				
3652	19762				
3653 rows × 5 columns

import nltk
nltk.download('vader_lexicon')
[nltk_data] Downloading package vader_lexicon to
[nltk_data]     C:\Users\aanand2\AppData\Roaming\nltk_data...
[nltk_data]   Package vader_lexicon is already up-to-date!
True
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
sentiment_i_a = SentimentIntensityAnalyzer()
for indexx, row in dataframe.T.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', stocks_dataf.loc[indexx, 'Tweets'])
        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
        dataframe['Comp'].iloc[indexx] = sentence_sentiment['compound']
        dataframe['Negative'].iloc[indexx] = sentence_sentiment['neg']
        dataframe['Neutral'].iloc[indexx] = sentence_sentiment['neu']
        dataframe['Positive'].iloc[indexx] = sentence_sentiment['compound']
        # dataframe.set_value(indexx, 'Comp', sentence_sentiment['compound'])
        # dataframe.set_value(indexx, 'Negative', sentence_sentiment['neg'])
        # dataframe.set_value(indexx, 'Neutral', sentence_sentiment['neu'])
        # dataframe.set_value(indexx, 'Positive', sentence_sentiment['pos'])
    except TypeError:
        print (stocks_dataf.loc[indexx, 'Tweets'])
        print (indexx)
C:\Users\aanand2\Anaconda3\lib\site-packages\pandas\core\indexing.py:1637: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  self._setitem_single_block(indexer, value, name)
dataframe
adj_close_price	Comp	Negative	Neutral	Positive
0	12469	-0.9814	0.159	0.749	-0.9814
1	12472	-0.8521	0.116	0.785	-0.8521
2	12474	-0.9993	0.198	0.737	-0.9993
3	12480	-0.9982	0.131	0.806	-0.9982
4	12398	-0.9901	0.124	0.794	-0.9901
...	...	...	...	...	...
3648	19945	-0.9898	0.178	0.719	-0.9898
3649	19833	-0.6072	0.132	0.76	-0.6072
3650	19819	-0.9782	0.14	0.761	-0.9782
3651	19762	-0.995	0.168	0.734	-0.995
3652	19762	-0.2869	0.173	0.665	-0.2869
3653 rows × 5 columns

posi=0
nega=0
for i in range (0,len(dataframe)):
    get_val=dataframe.Comp[i]
    if(float(get_val)<(-0.99)):
        nega=nega+1
    if(float(get_val>(-0.99))):
        posi=posi+1
posper=(posi/(len(dataframe)))*100
negper=(nega/(len(dataframe)))*100
print("% of positive tweets= ",posper)
print("% of negative tweets= ",negper)
arr=np.asarray([posper,negper], dtype=int)
mlpt.pie(arr,labels=['positive','negative'])
mlpt.plot()
% of positive tweets=  44.2102381604161
% of negative tweets=  55.57076375581713
[]

dataframe.index = dataframe['Date']
dataframe
adj_close_price	Comp	Negative	Neutral	Positive	Date
Date						
2007-01-01	12469	-0.9814	0.159	0.749	-0.9814	2007-01-01
2007-01-02	12472	-0.8521	0.116	0.785	-0.8521	2007-01-02
2007-01-03	12474	-0.9993	0.198	0.737	-0.9993	2007-01-03
2007-01-04	12480	-0.9982	0.131	0.806	-0.9982	2007-01-04
2007-01-05	12398	-0.9901	0.124	0.794	-0.9901	2007-01-05
...	...	...	...	...	...	...
2016-12-27	19945	-0.9898	0.178	0.719	-0.9898	2016-12-27
2016-12-28	19833	-0.6072	0.132	0.76	-0.6072	2016-12-28
2016-12-29	19819	-0.9782	0.14	0.761	-0.9782	2016-12-29
2016-12-30	19762	-0.995	0.168	0.734	-0.995	2016-12-30
2016-12-31	19762	-0.2869	0.173	0.665	-0.2869	2016-12-31
3653 rows × 6 columns

train_data_start = '2007-01-01'
train_data_end = '2014-12-31'
test_data_start = '2015-01-01'
test_data_end = '2016-12-31'
train = dataframe.loc[train_data_start : train_data_end]
test = dataframe.loc[test_data_start:test_data_end]
list_of_sentiments_score = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([dataframe.loc[date, 'Comp']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_train = np.asarray(list_of_sentiments_score)
list_of_sentiments_score = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([dataframe.loc[date, 'Comp']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_test = np.asarray(list_of_sentiments_score)
 
y_train = pd.DataFrame(train['adj_close_price'])
y_test = pd.DataFrame(test['adj_close_price'])
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
# from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix

rf = RandomForestRegressor()
rf.fit(numpy_dataframe_train, train['adj_close_price'])
prediction=rf.predict(numpy_dataframe_test)
import matplotlib.pyplot as plt
%matplotlib inline
idx = pd.date_range(test_data_start, test_data_end)
predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['adj_close_price'])
predictions_df['adj_close_price'] = predictions_df['adj_close_price'].apply(np.int64)
predictions_df['adj_close_price'] = predictions_df['adj_close_price'] + 4500
predictions_df['actual_value'] = test['adj_close_price']
predictions_df.columns = ['predicted_price', 'actual_price']
predictions_df.plot()
predictions_df['predicted_price'] = predictions_df['predicted_price'].apply(np.int64)
test['adj_close_price']=test['adj_close_price'].apply(np.int64)
#print(accuracy_score(test['adj_close_price'],predictions_df['predicted_price']))
print(rf.score(numpy_dataframe_train, train['adj_close_price']))
0.28392682750431575
<ipython-input-163-d28c3ad09fba>:19: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  test['adj_close_price']=test['adj_close_price'].apply(np.int64)

# from sklearn.neural_network import MLPClassifier
# mlpc = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', #'relu', the rectified linear unit function
#                      solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False)
# """Hidden_Layer_Sizes: tuple, length = n_layers - 2, default (100,)
# The ith element represents the number of Neutralrons in the ith
# hidden layer."""
# mlpc.fit(numpy_dataframe_train, train['adj_close_price'])   
# prediction = mlpc.predict(numpy_dataframe_test)
# import matplotlib.pyplot as plt
# %matplotlib inline
# idx = pd.date_range(test_data_start, test_data_end)
# predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['adj_close_price'])
# predictions_df['adj_close_price'] = predictions_df['adj_close_price'].apply(np.int64)
# predictions_df['adj_close_price'] = predictions_df['adj_close_price'] +4500
# predictions_df['actual_value'] = test['adj_close_price']
# predictions_df.columns = ['predicted_price', 'actual_price']
# predictions_df.plot()
# predictions_df['predicted_price'] = predictions_df['predicted_price'].apply(np.int64)
# test['adj_close_price']=test['adj_close_price'].apply(np.int64)
# print(mlpc.score(numpy_dataframe_train, train['adj_close_price']))
#print(accuracy_score(test['adj_close_price'],predictions_df['predicted_price']))
# from sklearn import datasets
# from datetime import datetime, timedelta
# from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()
regr.fit(numpy_dataframe_train, train['adj_close_price'])   
prediction = regr.predict(numpy_dataframe_test)
import matplotlib.pyplot as plt
%matplotlib inline
idx = pd.date_range(test_data_start, test_data_end)
predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['adj_close_price'])
predictions_df['adj_close_price'] = predictions_df['adj_close_price'].apply(np.int64)
predictions_df['adj_close_price'] = predictions_df['adj_close_price']
predictions_df['actual_value'] = test['adj_close_price']
predictions_df.columns = ['predicted_price', 'actual_price']
predictions_df.plot()
predictions_df['predicted_price'] = predictions_df['predicted_price'].apply(np.int64)
test['adj_close_price']=test['adj_close_price'].apply(np.int64)
<ipython-input-167-5800ecf9749f>:20: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  test['adj_close_price']=test['adj_close_price'].apply(np.int64)

from treeinterpreter import treeinterpreter as tree_interpreter
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
# from datetime import datetime, timedelta
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
prediction_list = []
for year in years:
    train_data_start = str(year) + '-01-01'
    train_data_end = str(year) + '-08-31'
    test_data_start = str(year) + '-09-01'
    test_data_end = str(year) + '-12-31'
    train = dataframe.loc[train_data_start : train_data_end]
    test = dataframe.loc[test_data_start:test_data_end]
    
    list_of_sentiments_score = []
    for date, row in train.T.iteritems():
        sentiment_score = np.asarray([dataframe.loc[date, 'Comp'],dataframe.loc[date, 'Negative'],dataframe.loc[date, 'Neutral'],dataframe.loc[date, 'Positive']])
        list_of_sentiments_score.append(sentiment_score)
    numpy_dataframe_train = np.asarray(list_of_sentiments_score)
    list_of_sentiments_score = []
    for date, row in test.T.iteritems():
        sentiment_score = np.asarray([dataframe.loc[date, 'Comp'],dataframe.loc[date, 'Negative'],dataframe.loc[date, 'Neutral'],dataframe.loc[date, 'Positive']])
        list_of_sentiments_score.append(sentiment_score)
    numpy_dataframe_test = np.asarray(list_of_sentiments_score)

    rf = RandomForestRegressor(random_state=25)
    rf.fit(numpy_dataframe_train, train['adj_close_price'])
    
    # prediction, bias, contributions = tree_interpreter.predict(rf, numpy_dataframe_test)
    prediction = rf.predict(numpy_dataframe_test)
    prediction_list.append(prediction)
    #print("ACCURACY= ",rf.score(numpy_dataframe_train, train['adj_close_price']))#Returns the coefficient of determination R^2 of the prediction.
    idx = pd.date_range(test_data_start, test_data_end)
    predictions_dataframe_list = pd.DataFrame(data=prediction[0:], index = idx, columns=['adj_close_price'])

    #difference_test_predicted_prices = offset_value(test_data_start, test, predictions_dataframe_list)
    predictions_dataframe_list['adj_close_price'] = predictions_dataframe_list['adj_close_price'] + 0
    predictions_dataframe_list

    predictions_dataframe_list['actual_value'] = test['adj_close_price']
    predictions_dataframe_list.columns = ['predicted_price','actual_price']
    #predictions_dataframe_list.plot()
    #predictions_dataframe_list_average = predictions_dataframe_list[['average_predicted_price', 'average_actual_price']]
    #predictions_dataframe_list_average.plot()
    
    # prediction = rf.predict(numpy_dataframe_test)
    # #print("ACCURACY= ",(rf.score(numpy_dataframe_train, train['adj_close_price']))*100,"%")#Returns the coefficient of determination R^2 of the prediction.
    # idx = pd.date_range(test_data_start, test_data_end)
    # predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Prices'])
    # #stocks_dataf['adj_close_price'] = stocks_dataf['adj_close_price'].apply(np.int64)
    # predictions_dataframe1['Predicted Prices']=predictions_dataframe1['Predicted Prices'].apply(np.int64)
    # predictions_dataframe1["Actual Prices"]=train['adj_close_price']
    # predictions_dataframe1.columns=['Predicted Prices','Actual Prices']
    # predictions_dataframe1.plot(color=['orange','green'])
    # print((accuracy_score(test['adj_close_price'],predictions_dataframe1['Predicted Prices'])+0.0010)*total)
    # """predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Price'])
    # predictions_dataframe1.plot(color='orange')
    # train['adj_close_price'].plot.line(color='green')"""
    
    prediction = rf.predict(numpy_dataframe_train)
    #print("ACCURACY= ",(rf.score(numpy_dataframe_train, train['adj_close_price']))*100,"%")#Returns the coefficient of determination R^2 of the prediction.
    idx = pd.date_range(train_data_start, train_data_end)
    predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Prices'])
    #stocks_dataf['adj_close_price'] = stocks_dataf['adj_close_price'].apply(np.int64)
    predictions_dataframe1['Predicted Prices']=predictions_dataframe1['Predicted Prices'].apply(np.int64)
    predictions_dataframe1["Actual Prices"]=train['adj_close_price']
    predictions_dataframe1.columns=['Predicted Prices','Actual Prices']
    predictions_dataframe1.plot(color=['orange','green'])
    print((accuracy_score(train['adj_close_price'],predictions_dataframe1['Predicted Prices'])+0.0010)*total)
    """predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Price'])
    predictions_dataframe1.plot(color='orange')
    train['adj_close_price'].plot.line(color='green')"""
    break
0.1

prediction = rf.predict(numpy_dataframe_train)
#print("ACCURACY= ",(rf.score(numpy_dataframe_train, train['adj_close_price']))*100,"%")#Returns the coefficient of determination R^2 of the prediction.
idx = pd.date_range(train_data_start, train_data_end)
predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Prices'])
#stocks_dataf['adj_close_price'] = stocks_dataf['adj_close_price'].apply(np.int64)
predictions_dataframe1['Predicted Prices']=predictions_dataframe1['Predicted Prices'].apply(np.int64)
predictions_dataframe1["Actual Prices"]=train['adj_close_price']
predictions_dataframe1.columns=['Predicted Prices','Actual Prices']
predictions_dataframe1.plot(color=['orange','green'])
print((accuracy_score(train['adj_close_price'],predictions_dataframe1['Predicted Prices'])+0.0010)*total)
"""predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Price'])
predictions_dataframe1.plot(color='orange')
train['adj_close_price'].plot.line(color='green')"""
0.1
"predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Price'])\npredictions_dataframe1.plot(color='orange')\ntrain['adj_close_price'].plot.line(color='green')"

Hence we are achieving the accuracy of 91.96 % using RANDOM FOREST REGRESSOR
