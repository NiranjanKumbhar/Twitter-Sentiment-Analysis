#All generic and Project Specific Libraries
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import snscrape.modules.twitter as sntwitter
import re
import csv
import calendar
import nltk
import itertools
import threading
import time
import sys
import os
import gensim
import pyLDAvis
import pickle 
from textblob import TextBlob
from datetime import datetime
from collections import Counter
#Natural Language Processing Libraries
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
#WordCloud Libraries
from wordcloud import WordCloud  # Check for WordCloud2
from PIL import Image
#Panda Dataframe advance Library
from pandas_datareader import data as pdr
#Yahoo Finance library
import yfinance as yf
#Corpus Libraries
from gensim.corpora import Dictionary   
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from pprint import pprint
#pyLDA Model library
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis.gensim_models as gensimvis
#Model Building Libraries
from sklearn import linear_model
from sklearn.svm import SVR 
from sklearn.cluster import KMeans 
from sklearn.tree import DecisionTreeRegressor, export_graphviz 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor 
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score 
from sklearn.model_selection import train_test_split


#Global Variables used throughout the program
lemmatized_words=[]
hastags=[]
taskDone = False
#Downloading Latest Wordnet 
nltk.download('wordnet')

#GetSet Class for Accessing objects at dynamic exeuction for faster and parallel access
class getterSetter():
    
    #Directory Path
    def setPath(self,path):
        self.path = path
    def getPath(self):
        return self.path
    
    #LDA Dataset
    def set_LDA_Data(self,dataframe):
        self.dataframe = dataframe
    def get_LDA_Data(self):
        return self.dataframe

    #Sentiment Dataframe
    def setSentimentDF(self,df):
        self.sentiDF= df
    def setMarketDF(self,df):
        self.marketDF = df
    
    #Setting Top 10 Words list
    def setTop10words(self,words):
        self.top10words=words
    def getTop10words(self):
        return self.top10words

#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if taskDone:
            break
        sys.stdout.write('\rProcessing ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone with Analysis!     ')

class textProcessingMining(getterSetter):
    #Class constructor-inititalizing list values which will be used througout the class
    def __init__(self,filename,all_tweets,selectedDataList,excludeWords,getset):
        self.filename           = filename
        self.allTweetTexts      = all_tweets
        self.completeDataList   = selectedDataList
        self.excludeWords       = excludeWords
        self.getset             = getset

    #Method to seperate hashtags from provided string and return list of hashtags
    def extractHashTags(self,s):
        return [i  for i in s.split() if i.startswith("#") ]

    #Method to plot word cloud
    def plot_cloud(self,wordcloud):
        # Set figure size
        plt.figure(figsize=(40, 30))
        # Display image
        plt.imshow(wordcloud) 
        # No axis details
        plt.axis("off");

    def dataProcessing(self):

        #assigning stopwords which are fetched nltk
        stop_words = set(stopwords.words('english'))
        #Adding few more stopwords which can be possibly in the tweets
        new_stopwords = ["amp", "u", "ji"]
        new_stopwords.extend(self.excludeWords)
        
        #Adding new stopwords to to make new stopword list
        new_stopwords_list = stop_words.union(new_stopwords)
        #Working on text part of tweet
        for curr_tweet in self.allTweetTexts:

            #Data Preprocessing and Cleaning:

            # 1 . lowercasing sentences
            preprocessingtweet = curr_tweet.lower()

            # 2. Extracting Hastags before cleaning
            curr_hashtags = self.extractHashTags(preprocessingtweet)

            # if hashtags are retreived, first hashtag will be stored in the same list
            if (len(curr_hashtags) != 0):
                hastags.extend(curr_hashtags)

            # 3.  Removing links which are having http Or https
            preprocessingtweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", preprocessingtweet)
            preprocessingtweet = re.sub(r'[^\w\s]','',preprocessingtweet)            

            preprocessingtweet  = re.sub(r'\b\w{1,3}\b', '', preprocessingtweet)
            curr_tweet   = preprocessingtweet
            # 4.  Tokenizing sentences to create tokens
            tokenizedWords = word_tokenize(twpreprocessingtweeteets)

            # 5.  Remove punctuations
            #Creating Translation table using maketrans where we are replacing punctuations with None('')
            table = str.maketrans('', '', string.punctuation) 
            stripped = [w.translate(table) for w in tokenizedWords]

            # 6.  Removing numbers
            #Keeping text which are alphanumeric
            cleanedWords = [word for word in tokenizedWords if word.isalnum()]  # Test

            filtered_tweets = []
            # 7.  Removing Stopwords
            for word in cleanedWords:
                if word not in new_stopwords_list:
                    filtered_tweets.append(word)

            # 8. Lemmatization
            wordnet_lemmatizer = WordNetLemmatizer()
            for eachWord in filtered_tweets:  # MT need to check
                nounWord = wordnet_lemmatizer.lemmatize(eachWord, pos="n")  # noun
                verbWord = wordnet_lemmatizer.lemmatize(nounWord, pos="v")  # verb
                adjWord = wordnet_lemmatizer.lemmatize(verbWord, pos="a")  # adjective
                lemmatized_words.append(adjWord)

        #Taking word count
        wordCount = Counter(lemmatized_words)
        print(":::::TOP 10 Words ::::\n")
        print(wordCount.most_common(10))
        
        #Converting the list to dataframe
        newdf_freq = pd.DataFrame(wordCount.most_common(10), columns=['word', 'frequency'])
        #plotting bar chart for word and frequency using matplotlib
        newdf_freq.plot(kind="bar",x='word',y='frequency')
        
        clean_string = ','.join(lemmatized_words)

        #generating wordcloud
        wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='white', colormap='Set2',
                            collocations=False).generate(clean_string)
        wordcloud.generate(clean_string)
        self.plot_cloud(wordcloud)
        #Saving wordcloud image to local directory
        wordcloud.to_file(self.getset.getPath()+"\Wordcloud_"+ self.filename + ".png")

        #Dynamically generating Top 10 words related wordclouds
        top10=[]
        for word,i in word_count.most_common(10):
            top10.append(word)
            matching = [s for s in self.allTweetTexts if word in s]
            lemma_word_1=[]

            for wrds in matching:
                wrds = wrds.lower()
                wrds = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", wrds)
                wrds = re.sub(r'[^\w\s]','',wrds)
                wrds= re.sub(r'\b\w{1,3}\b', '', wrds)
                wrds = re.sub(" \d+", " ", wrds)
                word_tokens1 = word_tokenize(wrds)

                words1 = [wrd for wrd in word_tokens1 if wrd.isalnum()]  # Test

                filtered_tweets_1 = []
                for curr_word in words1:
                    if curr_word not in new_stopwords_list:
                        filtered_tweets_1.append(curr_word)

                wordnet_lemmatizer = WordNetLemmatizer()
                for cur_word in filtered_tweets_1:  # MT need to check
                    word_1 = wordnet_lemmatizer.lemmatize(cur_word, pos="n")  # noun
                    word_2 = wordnet_lemmatizer.lemmatize(word_1, pos="v")  # verb
                    word_3 = wordnet_lemmatizer.lemmatize(word_2, pos="a")  # adjective
                    lemma_word_1.append(word_3)

            word_count = Counter(lemma_word_1)
            counts = word_count.most_common(30)
            # Retrieve words and counts from FreqDist tuples
            comm_counts = [x[1] for x in counts]
            comm_words = [x[0] for x in counts]

            # Create dictionary mapping of word count
            top_30_words = dict(zip(comm_words, comm_counts))

            wordclouds = WordCloud(width=600, height=600, colormap = 'Accent', background_color = 'white').generate_from_frequencies(top_30_words)

            # Plot word cloud with matplotlib
            plt.figure( figsize=(10,10))
            plt.imshow(wordclouds, interpolation='bilinear')
            plt.tight_layout()
            plt.axis("off")

            #Saving wordcloud image to local directory
            wordclouds.to_file(self.getset.getPath()+"\Wordcloud_of_"+ word + ".png")
            matching.clear()
            lemma_word_1.clear()
            
        self.getset.setTop10words(top10) 
        #Returing processed list
        return self.allTweetTexts

class sentimentAnalysis(getterSetter):
    def __init__(self,filename, all_tweets,selectedDataList,getset):
        self.filename           =   filename
        self.all_tweetTexts     =   all_tweets
        self.selectedDataList   =   selectedDataList
        self.getset             =   getset
    
    def performAnalysis(self):
        # sentimental Analysis
        sentiment_objects = [TextBlob(tweet) for tweet in self.all_tweetTexts]

        FinalList = []
        final_excel = []
        newlist = []
        positive_count = 0
        Negative_count = 0
        Neutral_count = 0
        

        # Creating List for polarity and subjectivity
        sentiment_values = [[tweet1.sentiment.polarity, tweet1.sentiment.subjectivity, str(tweet1.raw).lower()] for tweet1 in
                            sentiment_objects]
        
        for f, b in zip(sentiment_values, self.selectedDataList):
            length= len(b)
            obj=[]

            #Arranging Values form SelectedDataList
            obj.append(b[0])#Date
            obj.append(b[1])#Time
            obj.append(b[2])#DayoftheWeek
            obj.append(b[5])#Username
            obj.append(b[7])#Count of Likes
            obj.append(b[8])#Count of Retweets
            obj.append(b[9])#Verified
            obj.append(b[10])#Followers Count
            obj.append(b[11])#Friends Count
            obj.append(b[12])#Statuses Count
            obj.append(b[13])#Favorites Count
            obj.append(b[14])#Media Count

            #Arranging Values form sentiment_values for accessing polarity values and tweet
            obj.append(f[2])#Tweet Text
            obj.append(f[0])#Polarity  Value
            obj.append(f[1])#Subjectivity Values

            #Based on Polarity Value which is from -1 to 1,
            #Categorizing tweets into three broad categories
            if (f[0] > 0):
                add = "Positive"
                sent_value = 1
            elif (f[0] < 0):
                add = "Negative"
                sent_value = -1
            else:
                add = "Neutral"
                sent_value = 0

            obj.append(add)
            obj.append(sent_value)
            #Building the list using both sentiment values and tweet information
            FinalList.append(obj)

        #Converting list into Dataframe
        sentiment_df = pd.DataFrame(FinalList, columns=["Date","Time","Day","Username","Like","Retweet","Verifed","Followers Count","Friends Count","Statuses Count","Favorites Count","Media Count","tweet","polarity Value", "Subjectivity", "Sentiment","Sentiment Value"])
        #Exporting Dataframe to CSV File
        sentiment_df.to_excel(self.getset.getPath()+'\SentimentAnalysis_' + self.filename +'.xlsx', index = True)        
        return sentiment_df

class extractMarketData(getterSetter):
    def __init__(self,listingName,startdate,enddate,sentimentDf,getset):
        self.listingName = listingName
        self.startDate = startdate
        self.endDate = enddate
        self.sentimentdf = sentimentDf
        self.getset = getset
    
    def fetchMarketData(self):

        yf.pdr_override() 
         # download dataframe
        data = pdr.get_data_yahoo("MARUTI.BO", start=self.startDate, end=self.endDate,interval = "1d")
        
        #Calculating New Values
        data['% Change'] = (data['Adj Close'] / data['Adj Close'].shift(1))-1
        data['Range'] = data['High'] - data['Low']
        self.marketData = data
        
        #Exporting Market Data to Excel for further analysis
        data.to_excel(self.getset.getPath()+'\MarketData.xlsx', index = True)
        return data

class ldaModelling(getterSetter):
    def __init__(self,data,getset):
        self.dataset = data
        self.getset = getset

    def lemmatize_stemming(self,text):
        stemmer = PorterStemmer()
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(self,text):
        result = []
        
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result
    
    def modelling(self):
        new_clean_data =[]
        for tweet in self.dataset:
            tweet= tweet.lower()	
            tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)
            new_clean_data.append(tweet)

        data_text = pd.DataFrame(new_clean_data,columns=['tweet'])

        documents = data_text
        
        processed_docs = documents['tweet'].map(self.preprocess)

        dictionary = gensim.corpora.Dictionary(processed_docs)

        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        lda_model = gensim.models.LdaMulticore(bow_corpus,id2word=dictionary, passes=10, random_state=5, num_topics=10, workers=4)

        pprint(lda_model.print_topics())
        doc_lda = lda_model[bow_corpus]

        # Visualize the topics
        num_topics = 10
        pyLDAvis.enable_notebook()
        LDAvis_data_filepath = os.path.join(self.getset.getPath()+'/ldavis_prepared_'+str(num_topics))
        # # this is a bit time consuming - make the if statement True
        # # if you want to execute visualization prep yourself
        if 1 == 1:
            LDAvis_prepared = gensimvis.prepare(lda_model, bow_corpus, dictionary)
            with open(LDAvis_data_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)
        # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)
        pyLDAvis.save_html(LDAvis_prepared, self.getset.getPath()+'/ldavis_prepared_'+str(num_topics) +'.html')
        LDAvis_prepared
        print(doc_lda)

class twitterSentimentAnalysis(textProcessingMining,sentimentAnalysis,extractMarketData):
    def __init__(self,filename,searchTerm,listingName,dateFrom,dateTo,maxTweetCount):
        self.dataFilename           =   filename
        self.twitterSearchTerm      =   searchTerm
        self.stockListingName       =   listingName
        self.dateFrom               =   dateFrom
        self.dateTo                 =   dateTo
        self.maxFetchTweetCount     =   maxTweetCount
    
    def excludeHashtags(self,wordlist):
        self.excludeWordList = wordlist
        
    #creating local twitter dataset file for given combinations
    def createCSVFile(self,path):
        newCsvFile = open(path +'\Data_'+ self.dataFilename + '.csv', 'a', newline='', encoding='utf8')
        return newCsvFile
    
    def createDirectory(self,foldername):
        # Directory
        directory = foldername
  
        # Parent Directory path
        parent_dir = "C:/Group4/"

        try:
            os.makedirs(parent_dir, exist_ok = True)
            print("Directory '%s' created successfully" % parent_dir)
            path = os.path.join(parent_dir, directory)
            try:
                os.makedirs(path, exist_ok = True)
                print("Directory '%s' created successfully" % directory)
            except OSError as error:
                print("Directory '%s' can not be created" % directory)
        except OSError as error:
            print("Directory '%s' can not be created" % parent_dir)
            # Path
            path = os.path.join(parent_dir, directory)
            try:
                os.makedirs(path, exist_ok = True)
                print("Directory '%s' created successfully" % directory)
            except OSError as error:
                print("Directory '%s' can not be created" % directory)

        return path
    
    #Core Functionality which includes Fetching data and routing to important methods for further operations
    def main(self,offline=False):
        taskDone = False
        #starting the waiting animation
        t = threading.Thread(target=animate)
        t.start()
        #Variables used across fetch_data method for storing & performing operations on tweets
        alltexttweets = []
        CompleteDataList=[]

        #Creating/Checking Directory
        NewPath = self.createDirectory(self.dataFilename)

        #Initializing Getset Class
        getset = getterSetter()
        #Setting Path in getset
        getset.setPath(NewPath)
        #Fetching Tweets online
        if(offline==False):
            
            # Open/create a file to append data
            csvTweetfile = self.createCSVFile(NewPath)
            # Use csv writer
            csvTweetWriter = csv.writer(csv_file)
            #Writing Column Names to the CSV File
            csvTweetWriter.writerow(['Date','Time','Day', 'Tweet Id', 'Text', 'Username', 'location', 'Like Count', 'RT Count',
            'Verified','Followers Count','Friends Count','Statuses Count','Favorites Count','Media Count'])

            # Running Web Scrapping method for collecting tweet related information using
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper( 
                    self.twitterSearchTerm + ' since:' + self.dateFrom + ' until:' + self.dateTo + ' lang:en ').get_items()):
                #Considering English tweets only

                # Halt fetch process once desired tweet count is reached
                if i > self.maxFetchTweetCount:
                    break
                
                #Writing Relevant data into CSV related to single tweet
                csvTweetWriter.writerow([str(tweet.date.year) + '-' + str(tweet.date.month) + '-' + str(tweet.date.day),
                                    str(tweet.date.hour) +':'+str(tweet.date.minute),
                                    str(calendar.day_name[tweet.date.weekday()]),
                                    tweet.id,tweet.content, tweet.user.username, tweet.user.location,
                                    tweet.likeCount, tweet.retweetCount,
                                    tweet.user.verified,tweet.user.followersCount,tweet.user.friendsCount,tweet.user.statusesCount,tweet.user.favouritesCount,tweet.user.mediaCount])

                #Creating List of required and relevant parameters from twitter object
                selectedDataListitem=[str(tweet.date.year) + '-' + str(tweet.date.month) + '-' + str(tweet.date.day),
                                    str(tweet.date.hour) +':'+str(tweet.date.minute),
                                    str(calendar.day_name[tweet.date.weekday()]),
                                    tweet.id,tweet.content, tweet.user.username, tweet.user.location,
                                    tweet.likeCount, tweet.retweetCount,
                                    tweet.user.verified,tweet.user.followersCount,tweet.user.friendsCount,tweet.user.statusesCount,tweet.user.favouritesCount,tweet.user.mediaCount]
                
                #Creating List of Selected parameters from above step
                CompleteDataList.append(selectedDataListitem)
                #Just collecting tweet texts for text mining
                all_tweets.append(tweet.content)
                # print("Fetching Tweets : " + str(i))
                
            # Closing File after writing file content
            csvTweetfile.close()

            #Exporting DATASET File to Local storage
            data = pd.read_csv( NewPath+'\Data_' + self.dataFilename + '.csv')
            getset.set_LDA_Data(data)
        else:
            #Fetching and Reading Offline CSV File from local system
            data = pd.read_csv( NewPath+'\Data_' + self.dataFilename + '.csv')
            getset.set_LDA_Data(data)
            #Converting Dataframe to List for faster list processing
            CompleteDataList = data.values.tolist()

            #Collecting tweet- texts from twitter objects and finzlizing list
            for listitem in CompleteDataList:
                alltexttweets.append(listitem[4])

        #Once the Data is Fetched and Collected
        if(len(CompleteDataList)== 0):
            print("No Data For Provided Combination")
        else:
            print("Processing Texts..")
            # Data Processing of tweets using Text Mining 
            #Input Parameters: 1. Filename 2.texts in tweets  3.All dataset 4. Exclude wordlist 5. Getter Setter Class 
            TPM = textProcessingMining(self.dataFilename,alltexttweets,CompleteDataList,self.excludeWordList,getset)
            all_tweets = TPM.dataProcessing()

            print("Performing Sentimental Analysis..")
            # Performing Sentimental Analysis on tweets and hashtags 
            #Input Parameters: 1. Filename 2.texts in tweets  3.All dataset 4. Getter Setter Class
            SA=sentimentAnalysis(self.dataFilename,alltexttweets,CompleteDataList,getset)
            sentiment_Df = SA.performAnalysis()
            
            #FINDING TOP INFLUENCERS BASED ON POPULARITY AND REACH
            #calculating the popularity score
            sentiment_Df['Popularity_score'] = sentiment_Df['Statuses Count'] + sentiment_Df['Favorites Count'] + sentiment_Df['Media Count']
            #calculating the reach score
            sentiment_Df['Reach_score'] = sentiment_Df['Followers Count'] - sentiment_Df['Friends Count']

            plt.figure(figsize=(20,10)) #customizing the size of the plot
            #visualizing the data using bar plot
            ax = sns.barplot(x='Username', y='Reach_score', palette="Greens_d",
                            data=sentiment_Df.sort_values(by='Reach_score', ascending=False)[0:25])

            #getting the values of the data
            for p in ax.patches:
                ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.,
                                        p.get_height()), ha = 'center', va = 'center', 
                                        xytext = (0, 10), textcoords = 'offset points')
            #setting the parameters for the title, x and y labels of the plot
            ax.set_title("Reach Score for Top Twitter Account", size=40, weight='bold')
            ax.set_xlabel("Twitter Screen Names", size=20, weight='bold')
            ax.set_ylabel("Reach Score(Followers-Following)", size=20, weight='bold')
            #changing the rotation of the x axis tick labels 
            for item in ax.get_xticklabels():
                item.set_rotation(45)
                
            plt.savefig(getset.getPath()+'/reach.png') #saving the figure
            plt.figure(figsize=(20,10)) #customizing the size of the plot
            sns.set(style="darkgrid") #customizing the style of the plot

            ax = sns.barplot(x='Username', y='Popularity_score', palette="Greens_d",
                            data=sentiment_Df.sort_values(by='Popularity_score', ascending=False)[0:25])

            for p in ax.patches:
                ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.,
                                        p.get_height()), ha = 'center', va = 'center', 
                                        xytext = (0, 10), textcoords = 'offset points')
            #setting the parameters for the title, x and y labels of the plot
            ax.set_title("Popularity Score for Top Twitter Account", size=40, weight='bold')
            ax.set_xlabel("Twitter Screen Names", size=20, weight='bold')
            ax.set_ylabel("Popularity Score(Statuses Count+Favorites Count+ Media Count)", size=20, weight='bold')
            #changing the rotation of the x axis tick labels 
            for item in ax.get_xticklabels():
                item.set_rotation(45)
                
            plt.savefig(getset.getPath()+'/Popularity.png') #saving the figure
            
            # # Fetching Share Market Data
            print("Fetching Market Data for Analysis..")
            #Input Parameters: 1. Stock Market Listing Name 2.Starting Date  3.End Date 4. Getter Setter Class
            emd = extractMarketData(self.stockListingName, self.dateFrom, self.dateTo,sentiment_Df, getset)
            market_Df = emd.fetchMarketData()
            
            getset.setSentimentDF(sentiment_Df)
            getset.setMarketDF(market_Df)
            
            print("Performing Topic Modelling...")
            LDA= ldaModelling(alltexttweets,getset)
            LDA.modelling()

            taskDone = True
            

#Input Parameters: 1. Directory Name 2. Keyword to Search 3.Stock Price Listing Name 4. Starting Date 5. End date 6. Maximum Number of tweets
SentimentDataAnalysis = twitterSentimentAnalysis('MS_Analysis_5','marutisuzuki','MARUTI.BO','2017-11-01','2021-06-19',200000)

#Words explicitly we need to remove which will be removed while removing stopwords
exclude_wordlist = ["maruti","suzuki","car","marutisuzuki","india","nexa","india","u","http","https"]

#Initializing Exclude wordlist
SentimentDataAnalysis.excludeHashtags(exclude_wordlist)
#Online Fetching of Tweets using Scraping
# DataAnalysis.fetch_data()
#Using file from local system
SentimentDataAnalysis.main(offline=True)