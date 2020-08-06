Problem statement
In this competitive world where IT operations are happening round the clock 24 hours, every organization yearns for quickly resolving encountered incidents to have no or minimal business impact.
IT leverages incident management process to achieve this objective which after analyzing the type of the incident assign it to the respective groups to resolve the incident.
In this Capstone project, the goal is to build a classifier that can classify the tickets by analyzing text.

image.png

Approach
image.png

Details about the data and dataset files are given in below link,
https://drive.google.com/file/d/1OZNJm81JXucV3HmZroMq6qCT2m7ez7IJ

Installing Prerequisite Libraries
In [0]:
!pip install langdetect
!pip install googletrans
Requirement already satisfied: langdetect in /usr/local/lib/python3.6/dist-packages (1.0.8)
Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from langdetect) (1.12.0)
Requirement already satisfied: googletrans in /usr/local/lib/python3.6/dist-packages (2.4.0)
Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from googletrans) (2.21.0)
Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->googletrans) (1.24.3)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->googletrans) (2019.11.28)
Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->googletrans) (2.8)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->googletrans) (3.0.4)
Importing Libraries
In [0]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import preprocessing 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics  import accuracy_score, classification_report, confusion_matrix, homogeneity_score, silhouette_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
from sklearn import model_selection, svm

from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, GlobalMaxPool1D, SpatialDropout1D, GRU
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from collections import OrderedDict, Counter

import re

import string
from string import punctuation as punc

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.util import ngrams

from wordcloud import WordCloud, STOPWORDS 

from textblob import TextBlob
from tqdm import tqdm

from spacy.lang.en import English

import googletrans
from googletrans import Translator

from langdetect import detect, DetectorFactory

from google.colab import drive

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
Using TensorFlow backend.
The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.
We recommend you upgrade now or ensure your notebook will continue to use TensorFlow 1.x via the %tensorflow_version 1.x magic: more info.

Mounting Drive and read file
In [0]:
drive.mount('/content/drive/')
Drive already mounted at /content/drive/; to attempt to forcibly remount, call drive.mount("/content/drive/", force_remount=True).
In [0]:
project_path = '/content/drive/My Drive/GL AIML/Capstone'
os.chdir(project_path)
In [0]:
df = pd.read_excel('Input Data Synthetic (created but not used in our project).xlsx')
Data understanding
In [0]:
df.head()
Out[0]:
Short description	Description	Caller	Assignment group
0	login issue	-verified user details.(employee# & manager na...	spxjnwir pjlcoqds	GRP_0
1	outlook	\r\n\r\nreceived from: hmjdrvpb.komuaywn@gmail...	hmjdrvpb komuaywn	GRP_0
2	cant log in to vpn	\r\n\r\nreceived from: eylqgodm.ybqkwiam@gmail...	eylqgodm ybqkwiam	GRP_0
3	unable to access hr_tool page	unable to access hr_tool page	xbkucsvz gcpydteq	GRP_0
4	skype error	skype error	owlgqjme qhcozdfx	GRP_0
In [0]:
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8500 entries, 0 to 8499
Data columns (total 4 columns):
Short description    8492 non-null object
Description          8499 non-null object
Caller               8500 non-null object
Assignment group     8500 non-null object
dtypes: object(4)
memory usage: 265.8+ KB
In [0]:
# Find index of NaN values
SD_nan_index = df.loc[pd.isna(df["Short description"]), :].index
Desc_nan_index = df.loc[pd.isna(df['Description']),:].index
print('Short description null at index ', SD_nan_index)
print('Short description null at index ', Desc_nan_index)
Short description null at index  Int64Index([2604, 3383, 3906, 3910, 3915, 3921, 3924, 4341], dtype='int64')
Short description null at index  Int64Index([4395], dtype='int64')
In [0]:
#Converting all rows to String 
df['Short description']=df['Short description'].apply(str)
df['Description']=df['Description'].apply(str)
df['Caller']=df['Caller'].apply(str)
EDA - Let's Visualize the Data distributrion
Let's look at the distribution of our dataset based on Individual features
Dataset Distribution by "Assignment Group"
In [0]:
plt.figure(figsize=(20,12))
df["Assignment group"].value_counts().plot.pie(autopct='%1.2f%%', fontsize=10, startangle=25)
Out[0]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fd7eb26e198>

In [0]:
fig, (ax1) = plt.subplots(1,1, figsize=(20,18))


x = df['Assignment group'].value_counts()
sns.barplot(x, x.index, ax=ax1)
plt.title('Total Counts of each Assignment Groups ')
plt.gca().set_xlabel('Count of Groups',fontsize=15)
plt.gca().set_ylabel('Assignment Groups',fontsize=15)
Out[0]:
Text(0, 0.5, 'Assignment Groups')

Dataset Distribution by "Caller"
In [0]:
fig, (ax1) = plt.subplots(1,1, figsize=(15,10))

x = df.groupby(['Caller']).size().sort_values(ascending=False).head(20)
sns.barplot(x, x.index, ax=ax1)
plt.title('Total counts of different Callers')
plt.gca().set_xlabel('Count of Callers',fontsize=15)
plt.gca().set_ylabel('Callers',fontsize=15)
Out[0]:
Text(0, 0.5, 'Callers')

In [0]:
df[df['Caller']=='bpctwhsn kzqsbmtp']['Assignment group'].value_counts()
Out[0]:
GRP_8     362
GRP_9     153
GRP_5      96
GRP_6      89
GRP_10     60
GRP_60     16
GRP_12      8
GRP_45      7
GRP_1       6
GRP_13      4
GRP_18      3
GRP_47      2
GRP_29      1
GRP_57      1
GRP_44      1
GRP_14      1
Name: Assignment group, dtype: int64
In [0]:
#Group by ShortDescription and Group
dfSDG = df.groupby(['Short description','Assignment group']).size().sort_values(ascending=False).reset_index()

dfSDG[dfSDG['Assignment group']=='GRP_8']['Short description'].head(8)
Out[0]:
45     job Job_3028 failed in job_scheduler at: 08/24...
115                abended job in job_scheduler: Job_593
138    job Job_549 failed in job_scheduler at: 10/07/...
157    abended job in job_scheduler: bkbackup_tool_re...
160    abended job in job_scheduler: bk_hana_SID_62_e...
161    abended job in job_scheduler: bk_hana_SID_62_e...
166    job Job_749 failed in job_scheduler at: 08/27/...
167    job SID_41arc2 failed in job_scheduler at: 08/...
Name: Short description, dtype: object
In [0]:
#Group by ShortDescription and Group
dfSDG = df.groupby(['Short description','Assignment group']).size().sort_values(ascending=False).reset_index()

dfSDG[dfSDG['Assignment group']=='GRP_9']['Short description'].head(8)
Out[0]:
52     abended job in job_scheduler: Job_1148
55     abended job in job_scheduler: Job_1141
92     abended job in job_scheduler: Job_1320
104    abended job in job_scheduler: Job_1142
113    abended job in job_scheduler: Job_2114
207    abended job in job_scheduler: Job_1305
243               update of ae to bw and hana
519                           issue with bobj
Name: Short description, dtype: object
Dataset Distribution by "Short Description"
In [0]:
fig, (ax1) = plt.subplots(1,1, figsize=(15,10))

x = df['Short description'].value_counts()[:20]
sns.barplot(x, x.index, ax=ax1)
plt.title('Total counts of specific Short Descs')
plt.gca().set_xlabel('Count of Short Desc')
plt.gca().set_ylabel('Short Desc')

plt.show()

Let's do some feature engineering, EDA and preprocessing of data
In [0]:
df_en = df.copy()
In [0]:
df_en.head()
Out[0]:
Short description	Description	Caller	Assignment group
0	login issue	-verified user details.(employee# & manager na...	spxjnwir pjlcoqds	GRP_0
1	outlook	\r\n\r\nreceived from: hmjdrvpb.komuaywn@gmail...	hmjdrvpb komuaywn	GRP_0
2	cant log in to vpn	\r\n\r\nreceived from: eylqgodm.ybqkwiam@gmail...	eylqgodm ybqkwiam	GRP_0
3	unable to access hr_tool page	unable to access hr_tool page	xbkucsvz gcpydteq	GRP_0
4	skype error	skype error	owlgqjme qhcozdfx	GRP_0
In [0]:
#Converting all rows to String 
df_en['Short description']=df_en['Short description'].apply(str)
df_en['Description']=df_en['Description'].apply(str)
df_en['Caller']=df_en['Caller'].apply(str)
Removing Caller Name from Description Feature
In [0]:
caller_list = df_en["Caller"].str.split(" ", n = 1, expand = True)

caller_fname = caller_list[0]
caller_lname = caller_list[1]

caller_fname_list = caller_fname.to_list()
caller_lname_list = caller_lname.to_list()

df_en.Description = df_en.Description.apply(lambda x: ' '.join([word for word in x.split() if word not in caller_fname_list]))
df_en.Description = df_en.Description.apply(lambda x: ' '.join([word for word in x.split() if word not in caller_lname_list]))
In [0]:
#Let's combine all 3 independent attribute to 1
df_en['Complete_Description'] = df_en['Short description'].str.cat(df_en['Description'],sep=" ")
#remove unnecessary spaces
df_en.Complete_Description = df_en.Complete_Description.apply(lambda x: x.strip())
In [0]:
#Remove non consecutive duplicates
df_en['Complete_Description'] = (df_en['Complete_Description'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))
In [0]:
df_en.head(2)
Out[0]:
Short description	Description	Caller	Assignment group	Complete_Description
0	login issue	-verified user details.(employee# & manager na...	spxjnwir pjlcoqds	GRP_0	login issue -verified user details.(employee# ...
1	outlook	received from: hmjdrvpb.komuaywn@gmail.com hel...	hmjdrvpb komuaywn	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail...
As all 3 independent attributes are merged into 1 column, let's remove those columns from dataframe
In [0]:
#As all 3 independent attributes are merged into 1 column, let's remove those columns from dataframe
df_en.drop(['Short description','Description','Caller'], axis=1, inplace=True)
df_en.head()
Out[0]:
Assignment group	Complete_Description
0	GRP_0	login issue -verified user details.(employee# ...
1	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail...
2	GRP_0	cant log in to vpn received from: eylqgodm.ybq...
3	GRP_0	unable to access hr_tool page
4	GRP_0	skype error
We will extract some basic text features such as:
Number of words<br>
Number of characters<br>
Number of stopwords<br>
Number of special characters<br>
Number of numerics<br>
Number of uppercase words<br>
sentiment analysis<br>
Non English Descriptions
and so on...
In [0]:
nltk.download('stopwords')
stop = stopwords.words('english')

stop_words = []
stop_words += ["sr", "psa", "perpsr", "psa", "good", "evening", "will", "night", "afternoon","png", "mailto" "ca","nt","at" "i", "vip", "llv", "xyz", "cid", "image", "gmail","co", "in", "com", "ticket", "company", "received", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "nobody", "non", "none", "nonetheless", "noone", "normally", "nos", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"]
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
WordCloud with Stopwords
In [0]:
#Word Cloud
text = " ".join(review for review in df_en['Complete_Description'])
wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

Basic text Analysis
In [0]:
#df_en.Description=='nan'
def missing_val(text):
  if text=='nan':
    output='True'
  else:
    output='False'
  return output
In [0]:
def get_sentiment(text):
  sentiment = TextBlob(text).sentiment
  return sentiment[0]
In [0]:
translator = Translator()
languages = googletrans.LANGUAGES
In [0]:
DetectorFactory.seed = 0
In [0]:
def count_regexp_occ(regexp='', text=None):
    return len(re.findall(regexp, text))
In [0]:
def txtAnalysis(data, column):
  data['word_count'] = data[column].apply(lambda x : len(x.split(" ")))
  data['char_count'] = data[column].str.len()
  data['word_density'] = data['word_count'] / (data['char_count'] + 1)
  data['punc_count'] = data[column].apply(lambda x : len([a for a in x if a in punc]))
  data['stopwords'] = data[column].apply(lambda x: len([x for x in str(x).split() if x in stop]))
  data['numerics'] = data[column].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
  data['upper'] = data[column].apply(lambda x: len([x for x in x.split() if x.isupper()]))
  data['sentiment'] = data[column].apply(lambda x : TextBlob(x).sentiment[0])
  data['is_null'] = data[column].apply(lambda x : missing_val(x))
  #data['is_english'] = data[column].apply(lambda x : True if isEnglish(x) else False)
  data['Language'] = data[column].apply(lambda x: detect(x))
  data['Language'] = data['Language'].apply(lambda x: languages[x].upper())
  data['num_symbols'] = data[column].apply(lambda x: sum(x.count(w) for w in '*#&$%?!'))
  data['num_unique_words'] = data[column].apply(lambda x: len(set(w for w in x.split())))
  data['num_smilies'] = data[column].apply(lambda x: sum(x.count(w) for w in (':-)', ':)', ';-)', ';)')))
  # Count number of \n
  data['num_slash_n'] = data[column].apply(lambda x: count_regexp_occ(r"\n", x))
  # Check for time stamp
  data['has_timestamp'] = data[column].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
  # Check for http links
  data['has_http'] = data[column].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
  return data.head(5)
In [0]:
df_eda = df_en.copy()
txtAnalysis(df_eda, 'Complete_Description')
Out[0]:
Assignment group	Complete_Description	word_count	char_count	word_density	punc_count	stopwords	numerics	upper	sentiment	is_null	Language	num_symbols	num_unique_words	num_smilies	num_slash_n	has_timestamp	has_http
0	GRP_0	login issue -verified user details.(employee# ...	28	183	0.152174	14	7	0	0	0.45	False	ENGLISH	2	28	0	0	0	0
1	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail...	24	177	0.134831	8	7	0	0	0.60	False	ENGLISH	1	24	0	0	0	0
2	GRP_0	cant log in to vpn received from: eylqgodm.ybq...	13	81	0.158537	4	4	0	0	1.00	False	ENGLISH	0	13	0	0	0	0
3	GRP_0	unable to access hr_tool page	5	29	0.166667	1	1	0	0	-0.50	False	ENGLISH	0	5	0	0	0	0
4	GRP_0	skype error	2	11	0.166667	0	0	0	0	0.00	False	NORWEGIAN	0	2	0	0	0	0
English Vs Non English Description text Distribution
In [0]:
english_tickets = len(df_eda[df_eda['Language'] == 'ENGLISH'])
non_english_tickets = len(df_eda[df_eda['Language'] != 'ENGLISH'])
languages = pd.DataFrame(columns=['Language','Percentage'])
languages = languages.append({'Language':'English', 'Percentage':(english_tickets/8500)*100}, ignore_index=True)
languages = languages.append({'Language':'Non-English', 'Percentage':(non_english_tickets/8500)*100}, ignore_index=True)
languages
Out[0]:
Language	Percentage
0	English	82.764706
1	Non-English	17.235294
In [0]:
# Pie chart for English Vs Non-English languages
plt.figure(figsize=(4,4))
plt.pie(data=languages, x = 'Percentage', labels='Language', radius=1.5, shadow=True, autopct='%1.1f%%', explode=(0,0.2))
plt.savefig("LanguagesDist.png")

In [0]:
df_eda.groupby('Language')['Language'].count().sort_values(ascending=False)
Out[0]:
Language
ENGLISH       7035
GERMAN         381
AFRIKAANS      255
FRENCH         146
ITALIAN        125
NORWEGIAN       91
CATALAN         80
DANISH          77
SWEDISH         76
DUTCH           58
SPANISH         38
POLISH          31
PORTUGUESE      25
FILIPINO        12
WELSH           12
ROMANIAN        11
ALBANIAN        10
SLOVENIAN        7
FINNISH          5
ESTONIAN         5
CROATIAN         5
INDONESIAN       5
TURKISH          4
LITHUANIAN       2
CZECH            2
LATVIAN          1
SLOVAK           1
Name: Language, dtype: int64
In [0]:
plt.figure(figsize=(10,6))
df_eda.groupby('Language')['Language'].count().plot.bar()
Out[0]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fd7e3d1b240>

In [0]:
df_eda.describe()
Out[0]:
word_count	char_count	word_density	punc_count	stopwords	numerics	upper	sentiment	num_symbols	num_unique_words	num_smilies	num_slash_n	has_timestamp	has_http
count	8500.000000	8500.000000	8500.000000	8500.000000	8500.000000	8500.000000	8500.000000	8500.000000	8500.000000	8500.000000	8500.000000	8500.0	8500.000000	8500.0
mean	23.176353	173.733176	0.139796	14.133294	5.179882	0.559882	0.099529	-0.043817	0.420118	23.176353	0.001529	0.0	5.031176	0.0
std	38.950384	335.434937	0.032013	50.896100	6.583806	6.751471	0.451924	0.278579	2.692823	38.950384	0.039080	0.0	31.995243	0.0
min	1.000000	1.000000	0.013514	0.000000	0.000000	0.000000	0.000000	-1.000000	0.000000	1.000000	0.000000	0.0	0.000000	0.0
25%	6.000000	44.000000	0.117021	1.000000	1.000000	0.000000	0.000000	-0.133333	0.000000	6.000000	0.000000	0.0	0.000000	0.0
50%	12.000000	103.000000	0.141026	5.000000	2.000000	0.000000	0.000000	0.000000	0.000000	12.000000	0.000000	0.0	1.000000	0.0
75%	28.000000	198.000000	0.160654	11.000000	7.000000	0.000000	0.000000	0.000000	0.000000	28.000000	0.000000	0.0	6.000000	0.0
max	981.000000	8753.000000	0.500000	2296.000000	59.000000	271.000000	7.000000	1.000000	156.000000	981.000000	1.000000	0.0	1016.000000	0.0
In [0]:
df_eda.sort_values(by='word_count', ascending=False).head(5)
Out[0]:
Assignment group	Complete_Description	word_count	char_count	word_density	punc_count	stopwords	numerics	upper	sentiment	is_null	Language	num_symbols	num_unique_words	num_smilies	num_slash_n	has_timestamp	has_http
4089	GRP_2	security incidents - ( sw #in33895560 ) : mage...	981	8753	0.112063	572	46	271	1	-0.004900	False	ENGLISH	28	981	0	0	941	0
4087	GRP_39	security incidents - ( sw #in33895560 ) : mage...	980	8749	0.112000	571	46	271	1	-0.004900	False	ENGLISH	28	980	0	0	941	0
5433	GRP_2	security incidents - ( #in33765965 ) : possibl...	960	8009	0.119850	2296	5	218	2	-0.285511	False	CATALAN	80	960	0	0	1016	0
7997	GRP_2	security incidents - ( sw #in33544563 ) : poss...	838	7016	0.119424	720	54	119	2	-0.054466	False	ENGLISH	27	838	0	0	596	0
8002	GRP_62	security incidents - ( sw #in33544563 ) : poss...	838	7016	0.119424	720	54	119	2	-0.054466	False	ENGLISH	27	838	0	0	596	0
In [0]:
print("Maximum number of words used in the description: ",df_eda.word_count.max())
Maximum number of words used in the description:  981
In [0]:
print("Maximun number of characters used in the description: ",df_eda.char_count.max())
Maximun number of characters used in the description:  8753
In [0]:
print("Maximun number of punctuations used in the description: ",df_eda.punc_count.max())
Maximun number of punctuations used in the description:  2296
In [0]:
print("Maximun number of stopwords used in the description: ",df_eda.stopwords.max())
Maximun number of stopwords used in the description:  59
In [0]:
print("Maximun number of numerics/digits used in the description: ",df_eda.numerics.max())
Maximun number of numerics/digits used in the description:  271
Let's visualize the features that we extracted above
Word Density Distribution
In [0]:
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df_eda.word_density, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Word Density')
plt.ylabel('Count')
plt.title('Histogram of Word Density')
Out[0]:
Text(0.5, 1.0, 'Histogram of Word Density')

Word Count Distribution
In [0]:
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df_eda.word_count, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Word Count')
plt.ylabel('Count')
plt.title('Histogram of Word Count')
Out[0]:
Text(0.5, 1.0, 'Histogram of Word Count')

Punctuation count Distribution
In [0]:
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df_eda.punc_count, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Punctuation Count')
plt.ylabel('Count')
plt.title('Histogram of Punctuation Count')
plt.show();

Stopwords count Distribution
In [0]:
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df_eda.stopwords, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Stopwords Count')
plt.ylabel('Count')
plt.title('Histogram of Stopwords Count')
plt.show();

Word count and stopwords Distribution
In [0]:
df_eda[['word_count','stopwords']].plot(figsize=(10,5), grid=True)
Out[0]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fd7e39f4b70>

Word count and punctuation count Distribution
In [0]:
df_eda[['word_count','punc_count']].plot(figsize=(10,5), grid=True)
Out[0]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fd7e3b217f0>

Numerics count distribution
In [0]:
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df_eda.numerics, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Numeric Count')
plt.ylabel('Count')
plt.title('Histogram of Numeric Count in Description')
Out[0]:
Text(0.5, 1.0, 'Histogram of Numeric Count in Description')

Null values Distribution after merging short description and description columns
In [0]:
x=df_eda['is_null'].value_counts()
x=x.sort_index()
plt.figure(figsize=(8,6))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Issue Description Text Distribution")
plt.ylabel('count')
plt.xlabel('Issue Description Text')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

Sentiment Analysis Distribution
In [0]:
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df_eda.sentiment, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Histogram of Sentiment')
plt.show();

Let's see the text based on the sentiment
Positive selntiment texts
In [0]:
print('3 random Description with the highest positive sentiment: \n')
print('*********************************************************')
cl = df_eda.loc[df_eda.sentiment == 1, ['Complete_Description']].sample(3).values
for c in cl:
    print(c[0])
    print('---------------------')
3 random Description with the highest positive sentiment: 

*********************************************************
password cannot changed received from: tbukjcyl.lxncwqbj@gmail.com dear all could you please help me to fix it, [cid:image001.jpg@01d1f978.729627d0] best
---------------------
email-anzeige received from: trgqbeax.hfyzudql@gmail.com [cid:image001.png@01d1f7b0.223a83e0] leider ist das feld â€žvon" abhanden gekommen â˜¹ danke + viele grÃ¼ÃŸe mit freundlichen grÃ¼ÃŸen | best
---------------------
unlock erp logon received from: jofvunqs.uwigjmzv@gmail.com hello: please help me logon. i can not the system ,input my password. best
---------------------
Neutral sentiment texts
In [0]:
print('3 random Description with the Neutral sentiment: \n')
print('*********************************************************')
cl = df_eda.loc[df_eda.sentiment == 0, ['Complete_Description']].sample(3).values
for c in cl:
    print(c[0])
    print('---------------------')
3 random Description with the Neutral sentiment: 

*********************************************************
password reset and access to reporting_engineering_tools
---------------------
erp SID_34 account locked.
---------------------
phishing emails uacyltoe hxgaycze query
---------------------
Negative sentiment texts
In [0]:
df_eda.sentiment.min()
Out[0]:
-1.0
In [0]:
print('Description with the most negative sentiment: \n')
print('*********************************************************')
cl = df_eda.loc[df_eda.sentiment == -1.0, ['Complete_Description']].values
for c in cl:
    print(c[0])
Description with the most negative sentiment: 

*********************************************************
pls. help to run out dn under sto#5019320060,thx! received from: wktesmbp.lorjymef@gmail.com dear team, we got a stock recall notic#plant_101-101016-01 for mm#3098450 & should return 373pc materials plant_101,then i created sto#5019320060 base on this recall. dn#916955708/105pc was just against the sto, of rest 268pc ,thx lot! [cid:image001.png@01d223c2.0ac78080] b.rgds judthtihty.zhuyhts company hardpoint apac-wgq dc
Most Common and Rare words
Most Common
In [0]:
Counter(" ".join(df_eda['Complete_Description']).split() ).most_common(20)
Out[0]:
[('to', 3316),
 ('in', 2642),
 ('the', 2555),
 ('from:', 2445),
 ('received', 2329),
 ('is', 1937),
 ('not', 1688),
 ('on', 1592),
 ('and', 1545),
 ('please', 1524),
 ('for', 1482),
 ('i', 1261),
 ('a', 1164),
 ('of', 1052),
 ('it', 1017),
 ('job', 997),
 ('erp', 969),
 ('monitoring_tool@company.com', 961),
 ('password', 937),
 ('unable', 870)]
Rare words
In [0]:
#Least Word Freq
pd.set_option('display.max_colwidth', -1)

S = pd.Series(" ".join(df_eda['Complete_Description']).split() ).value_counts().sort_values(ascending=True)
df_least_word_freq = pd.DataFrame(S).reset_index()
df_least_word_freq.columns = ['word', 'count']
df_least_word_freq[df_least_word_freq['count'] < 5]
Out[0]:
word	count
0	imaginal	1
1	12:43:00	1
2	xgrhplvk.coejktzn@gmail.com	1
3	outside:192.168.0.109/135	1
4	krcscfpr061y	1
...	...	...
27874	456e	4
27875	crashed	4
27876	easy	4
27877	ÑƒÐ²Ð°Ð¶ÐµÐ½Ð¸ÐµÐ¼,	4
27878	remember	4
27879 rows × 2 columns

Preprocessing
In [0]:
dfTicketAssign =  df_eda.copy()
In [0]:
dfTicketAssign.head()
Out[0]:
Assignment group	Complete_Description	word_count	char_count	word_density	punc_count	stopwords	numerics	upper	sentiment	is_null	Language	num_symbols	num_unique_words	num_smilies	num_slash_n	has_timestamp	has_http
0	GRP_0	login issue -verified user details.(employee# & manager name) -checked the name in ad and reset password. -advised to check. -caller confirmed that he was able login. -issue resolved.	28	183	0.152174	14	7	0	0	0.45	False	ENGLISH	2	28	0	0	0	0
1	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail.com hello team, my meetings/skype meetings etc are not appearing in calendar, can somebody please advise how to correct this? kind	24	177	0.134831	8	7	0	0	0.60	False	ENGLISH	1	24	0	0	0	0
2	GRP_0	cant log in to vpn received from: eylqgodm.ybqkwiam@gmail.com hi i cannot on best	13	81	0.158537	4	4	0	0	1.00	False	ENGLISH	0	13	0	0	0	0
3	GRP_0	unable to access hr_tool page	5	29	0.166667	1	1	0	0	-0.50	False	ENGLISH	0	5	0	0	0	0
4	GRP_0	skype error	2	11	0.166667	0	0	0	0	0.00	False	NORWEGIAN	0	2	0	0	0	0
Basic Preprocessing
In [0]:
regexList = ['From:(.*)\r\n',
 'IiNnCc[0-9]*',
 'ticket[_]*[\\s]*[0-9]*',
 'Sent:(.*)\r\n',
 'Received:(.*)\r\n',
 'To:(.*)\r\n',
 'CC:(.*)\r\n',
 '\\[cid:(.*)]',
 'https?:[^\\]\n\r]+',
 'Subject:',
 '[0-9][\\-0–90-9 ]+',# phones
 '[0-9]',# numbers
 '[^a-zA-z 0-9]+',# anything that is not a letter
 '[\r\n]',# single letters
 ' [a-zA-Z] ',  # two-letter words
 '  ', # double spaces
 '^[_a-z0-9-]+(\\.[_a-z0-9-]+)*@[a-z0-9-]+(\\.[a-z0-9-]+)*(\\.[a-z]{2,4})$',
 '[\\w\\d\\-\\_\\.]+ @ [\\w\\d\\-\\_\\.]+',
 'Subject:',
 '[^a-zA-Z]',
 '\\S+@\\S+',# emails 
 "\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b" #IP Address
 ]
In [0]:
#Remove punctuations
def remove_punctuations(text):
    for punctuation in string.punctuation:
          text = text.replace(punctuation, ' ')
    return text
In [0]:
def removeString(data, regex):
    return data.str.lower().str.replace(regex.lower(), ' ')
In [0]:
#Translate tickets to English
def fn_translate(desc, lang):
  try:
    if lang == 'ENGLISH':
        return desc
    else:
        return translator.translate(desc).text
  except:
    return desc
In [0]:
def textPreprocessing(data,column):
  print("updating all cases to lower cases:")
  data[column] = data[column].apply(lambda x: " ".join(x.lower() for x in x.split()))
  print("Translating Non English to English:")
  data['EnglishDescription'] = data.apply(lambda x: fn_translate(x[column], x['Language']), axis=1)
  print("removing data using regular expression List:")
  for regex in regexList:
            data['EnglishDescription'] = removeString(data['EnglishDescription'], regex)
  print("removing stopwords:")            
  data['EnglishDescription'] = data['EnglishDescription'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
  freq = pd.Series(' '.join(data['EnglishDescription']).split()).value_counts()[:20]
  print("removing top 20 Most common words:")
  data['EnglishDescription'] = data['EnglishDescription'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
  rare = pd.Series(' '.join(data['EnglishDescription']).split()).value_counts()[-20:]
  print("removing top 20 rare words ata using regular expression List:")
  data['EnglishDescription'] = data['EnglishDescription'].apply(lambda x: " ".join(x for x in x.split() if x not in rare))
  print("removing punctuations:")
  data['EnglishDescription'] = data['EnglishDescription'].apply(remove_punctuations)
  return data.head(5)
In [0]:
dfTicketAssign.head()
Out[0]:
Assignment group	Complete_Description	word_count	char_count	word_density	punc_count	stopwords	numerics	upper	sentiment	is_null	Language	num_symbols	num_unique_words	num_smilies	num_slash_n	has_timestamp	has_http
0	GRP_0	login issue -verified user details.(employee# & manager name) -checked the name in ad and reset password. -advised to check. -caller confirmed that he was able login. -issue resolved.	28	183	0.152174	14	7	0	0	0.45	False	ENGLISH	2	28	0	0	0	0
1	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail.com hello team, my meetings/skype meetings etc are not appearing in calendar, can somebody please advise how to correct this? kind	24	177	0.134831	8	7	0	0	0.60	False	ENGLISH	1	24	0	0	0	0
2	GRP_0	cant log in to vpn received from: eylqgodm.ybqkwiam@gmail.com hi i cannot on best	13	81	0.158537	4	4	0	0	1.00	False	ENGLISH	0	13	0	0	0	0
3	GRP_0	unable to access hr_tool page	5	29	0.166667	1	1	0	0	-0.50	False	ENGLISH	0	5	0	0	0	0
4	GRP_0	skype error	2	11	0.166667	0	0	0	0	0.00	False	NORWEGIAN	0	2	0	0	0	0
In [0]:
textPreprocessing(dfTicketAssign,'Complete_Description')
updating all cases to lower cases:
Translating Non English to English:
removing data using regular expression List:
removing stopwords:
removing top 20 Most common words:
removing top 20 rare words ata using regular expression List:
removing punctuations:
Out[0]:
Assignment group	Complete_Description	word_count	char_count	word_density	punc_count	stopwords	numerics	upper	sentiment	is_null	Language	num_symbols	num_unique_words	num_smilies	num_slash_n	has_timestamp	has_http	EnglishDescription
0	GRP_0	login issue -verified user details.(employee# & manager name) -checked the name in ad and reset password. -advised to check. -caller confirmed that he was able login. -issue resolved.	28	183	0.152174	14	7	0	0	0.45	False	ENGLISH	2	28	0	0	0	0	login verified details employee manager checked advised check caller confirmed login resolved
1	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail.com hello team, my meetings/skype meetings etc are not appearing in calendar, can somebody please advise how to correct this? kind	24	177	0.134831	8	7	0	0	0.60	False	ENGLISH	1	24	0	0	0	0	outlook hmjdrvpb komuaywn team meetings skype meetings appearing calendar somebody advise correct kind
2	GRP_0	cant log in to vpn received from: eylqgodm.ybqkwiam@gmail.com hi i cannot on best	13	81	0.158537	4	4	0	0	1.00	False	ENGLISH	0	13	0	0	0	0	cant log vpn eylqgodm ybqkwiam cannot
3	GRP_0	unable to access hr_tool page	5	29	0.166667	1	1	0	0	-0.50	False	ENGLISH	0	5	0	0	0	0	hr
4	GRP_0	skype error	2	11	0.166667	0	0	0	0	0.00	False	NORWEGIAN	0	2	0	0	0	0	skype
Most Common words after preprocessing
In [0]:
counter = Counter(" ".join(dfTicketAssign['EnglishDescription']).split() ).most_common(20)
counter
Out[0]:
[('system', 606),
 ('network', 591),
 ('outlook', 549),
 ('need', 546),
 ('vendor', 525),
 ('login', 523),
 ('power', 494),
 ('telecom', 481),
 ('message', 439),
 ('see', 412),
 ('phone', 404),
 ('outage', 401),
 ('team', 383),
 ('locked', 370),
 ('usa', 355),
 ('vpn', 341),
 ('update', 339),
 ('server', 337),
 ('number', 336),
 ('contact', 334)]
In [0]:
labels, values = zip(*Counter(" ".join(dfTicketAssign['EnglishDescription']).split() ).most_common(20))
plt.figure(figsize=(20,6))

indexes = np.arange(len(labels))
width = 0.25

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()

In [0]:
pd.set_option('display.max_colwidth', -1)

S = pd.Series(" ".join(dfTicketAssign['EnglishDescription']).split() ).value_counts().sort_values(ascending=True)
df_least_word_freq = pd.DataFrame(S).reset_index()
df_least_word_freq.columns = ['word', 'count']
df_least_word_freq[df_least_word_freq['count'] < 5]
Out[0]:
word	count
0	uncaught	1
1	vbmzgsdk	1
2	gerberghty	1
3	stamped	1
4	awnftgev	1
...	...	...
11353	outputs	4
11354	soll	4
11355	komuaywn	4
11356	cyxieuwk	4
11357	ufriscym	4
11358 rows × 2 columns

Tokenization and Lemmatization
In [0]:
# Word tokenization
nlp = English()
In [0]:
def tokenizeText(text):
  #  "nlp" Object is used to create documents with linguistic annotations.
  my_doc = nlp(text)
  # Create list of word tokens
  token_list = []
  for token in my_doc:
    token_list.append(token.text)
  return token_list
In [0]:
nltk.download('punkt')
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
Out[0]:
True
In [0]:
dfTicketAssign['Tokens'] = dfTicketAssign['EnglishDescription'].apply(lambda x: tokenizeText(x))
In [0]:
#Cleaning stop words after tokenization
cleanTokens = []
for col_desc in range(len(dfTicketAssign.Tokens)):
  str_token = dfTicketAssign.Tokens[col_desc]
  cleanTokens.append([w for w in str_token if w not in stop_words] )

dfTicketAssign['Tokens'] = cleanTokens
In [0]:
# Python code to remove duplicate elements after tokenization
def RemovDupWordTokens(dupToken): 
    finalTokenlist = [] 
    for num in dupToken: 
        if num not in finalTokenlist: 
            finalTokenlist.append(num) 
    return finalTokenlist
In [0]:
def lemmatizeText(text):
  text = ' '.join(map(str, text)) 
  # Implementing lemmatization
  lem = nlp(text)
  lemma_list = []
  # finding lemma for each word
  for word in lem:
    lemma_list.append(word.lemma_)
  return lemma_list
  #return list(set(lemma_list))
In [0]:
dfTicketAssign['LemmaWords'] = dfTicketAssign['Tokens'].apply(lambda x: lemmatizeText(x))
Word Cloud after Lemmatization
In [0]:
#Word Cloud
text = ' '.join(map(str, dfTicketAssign.LemmaWords))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

In [0]:
print('Average count of chars per Description cell is {0:.0f}.'.format(dfTicketAssign.groupby('Assignment group')['EnglishDescription'].count().mean()))
Average count of chars per Description cell is 115.
In [0]:
dfTicketAssign['LemmaWords'] = dfTicketAssign['LemmaWords'].apply(lambda x: RemovDupWordTokens(x))
Word Cloud after removing duplicates from Lemmatized words
In [0]:
#Word Cloud
text = ' '.join(map(str, dfTicketAssign.LemmaWords))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

N-Grams
In [0]:
dfTicketngrams = dfTicketAssign.copy()
In [0]:
dfTicketngrams.head(5)
Out[0]:
Assignment group	Complete_Description	word_count	char_count	word_density	punc_count	stopwords	numerics	upper	sentiment	is_null	Language	num_symbols	num_unique_words	num_smilies	num_slash_n	has_timestamp	has_http	EnglishDescription	Tokens	LemmaWords
0	GRP_0	login issue -verified user details.(employee# & manager name) -checked the name in ad and reset password. -advised to check. -caller confirmed that he was able login. -issue resolved.	28	183	0.152174	14	7	0	0	0.45	False	ENGLISH	2	28	0	0	0	0	login verified details employee manager checked advised check caller confirmed login resolved	[login, verified, details, employee, manager, checked, advised, check, caller, confirmed, login, resolved]	[login, verify, detail, employee, manager, check, advise, caller, confirm, resolve]
1	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail.com hello team, my meetings/skype meetings etc are not appearing in calendar, can somebody please advise how to correct this? kind	24	177	0.134831	8	7	0	0	0.60	False	ENGLISH	1	24	0	0	0	0	outlook hmjdrvpb komuaywn team meetings skype meetings appearing calendar somebody advise correct kind	[outlook, hmjdrvpb, komuaywn, team, meetings, skype, meetings, appearing, calendar, somebody, advise, correct, kind]	[outlook, hmjdrvpb, komuaywn, team, meeting, skype, appear, calendar, somebody, advise, correct, kind]
2	GRP_0	cant log in to vpn received from: eylqgodm.ybqkwiam@gmail.com hi i cannot on best	13	81	0.158537	4	4	0	0	1.00	False	ENGLISH	0	13	0	0	0	0	cant log vpn eylqgodm ybqkwiam cannot	[log, vpn, eylqgodm, ybqkwiam, not]	[log, vpn, eylqgodm, ybqkwiam, not]
3	GRP_0	unable to access hr_tool page	5	29	0.166667	1	1	0	0	-0.50	False	ENGLISH	0	5	0	0	0	0	hr	[hr]	[hr]
4	GRP_0	skype error	2	11	0.166667	0	0	0	0	0.00	False	NORWEGIAN	0	2	0	0	0	0	skype	[skype]	[skype]
In [0]:
def get_top_ticketdesc_unigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(1, 1)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    print(words_freq)
    return words_freq[:n]
In [0]:
plt.figure(figsize=(10,5))
top_unigrams=get_top_ticketdesc_unigrams(dfTicketngrams['EnglishDescription'])[:20]
x,y=map(list,zip(*top_unigrams))
sns.barplot(x=y,y=x)
[('system', 606), ('network', 591), ('outlook', 549), ('need', 546), ('vendor', 525), ('login', 523), ('power', 494), ('telecom', 481), ('message', 439), ('see', 412), ('phone', 404), ('outage', 401), ('team', 383), ('locked', 370), ('usa', 355), ('vpn', 341), ('update', 339), ('server', 337), ('number', 336), ('contact', 334), ('customer', 332), ('check', 327), ('log', 325), ('plant', 317), ('engineering', 310), ('time', 310), ('circuit', 307), ('manager', 305), ('cannot', 301), ('windows', 301), ('request', 300), ('summary', 297), ('site', 296), ('crm', 295), ('open', 289), ('mailto', 289), ('skype', 286), ('maint', 285), ('type', 282), ('microsoft', 282), ('problem', 271), ('management', 260), ('attached', 260), ('internet', 248), ('change', 245), ('data', 243), ('connect', 233), ('order', 231), ('work', 230), ('sales', 230), ('start', 228), ('language', 225), ('platform', 224), ('global', 224), ('same', 224), ('collaboration', 223), ('printer', 221), ('software', 215), ('unlock', 207), ('dear', 206), ('screen', 204), ('remote', 202), ('computer', 201), ('explorer', 200), ('active', 199), ('production', 199), ('laptop', 198), ('device', 197), ('browser', 195), ('inside', 195), ('other', 192), ('telephone', 189), ('verified', 187), ('backup', 184), ('hr', 183), ('office', 182), ('connection', 182), ('details', 176), ('mit', 175), ('germany', 173), ('file', 173), ('started', 170), ('business', 170), ('portal', 169), ('create', 168), ('additional', 168), ('passwords', 167), ('use', 167), ('address', 165), ('users', 164), ('issues', 164), ('gsc', 164), ('service', 164), ('maintenance', 163), ('application', 163), ('scheduled', 160), ('print', 159), ('mail', 159), ('abended', 153), ('know', 152), ('needs', 150), ('report', 150), ('provider', 149), ('nwfodmhc', 149), ('exurcwkm', 149), ('cert', 148), ('equipment', 148), ('add', 147), ('dial', 147), ('link', 146), ('notified', 146), ('diagnostics', 144), ('location', 144), ('provide', 144), ('verizon', 143), ('employee', 141), ('client', 140), ('uacyltoe', 137), ('view', 135), ('delivery', 134), ('blocked', 134), ('should', 133), ('connected', 132), ('status', 130), ('agent', 130), ('resolved', 129), ('mm', 129), ('urgent', 128), ('inwarehouse', 127), ('install', 126), ('code', 126), ('set', 125), ('problems', 125), ('today', 124), ('reporting', 124), ('note', 123), ('kindly', 123), ('changed', 122), ('hxgaycze', 121), ('support', 120), ('possible', 119), ('nicht', 118), ('confirmed', 116), ('folder', 113), ('event', 112), ('resolve', 112), ('emails', 111), ('enter', 110), ('morning', 110), ('his', 109), ('required', 108), ('high', 108), ('drive', 108), ('mobile', 107), ('security', 107), ('sw', 106), ('list', 106), ('inplant', 105), ('center', 104), ('august', 103), ('process', 101), ('september', 101), ('files', 100), ('showing', 98), ('priority', 98), ('advise', 95), ('action', 95), ('slow', 94), ('group', 93), ('host', 92), ('telephony', 91), ('ing', 91), ('created', 91), ('bitte', 91), ('notification', 90), ('app', 90), ('interface', 89), ('caller', 88), ('apac', 88), ('october', 88), ('being', 87), ('setup', 87), ('correct', 86), ('pls', 86), ('times', 86), ('tcp', 86), ('material', 85), ('ethics', 84), ('checked', 83), ('india', 83), ('services', 83), ('attachment', 83), ('running', 82), ('missing', 82), ('content', 82), ('freundlichen', 82), ('excel', 81), ('orders', 81), ('internal', 79), ('space', 78), ('printing', 78), ('send', 77), ('und', 77), ('der', 77), ('complete', 76), ('receive', 76), ('kind', 75), ('teamviewer', 75), ('screenshot', 75), ('events', 75), ('sir', 74), ('label', 74), ('port', 74), ('click', 73), ('monitor', 73), ('model', 73), ('dell', 72), ('questions', 71), ('document', 71), ('blank', 71), ('multiple', 71), ('because', 71), ('die', 71), ('wrong', 71), ('ess', 70), ('logon', 70), ('net', 70), ('want', 70), ('completed', 70), ('aerp', 69), ('automatically', 69), ('ist', 69), ('exe', 69), ('mii', 69), ('day', 68), ('local', 67), ('installed', 67), ('assist', 67), ('warehouse', 67), ('updated', 67), ('tools', 66), ('wifi', 66), ('machine', 66), ('fine', 66), ('probleme', 66), ('volume', 66), ('version', 66), ('iphone', 65), ('field', 65), ('hallo', 65), ('asa', 65), ('admin', 64), ('reports', 63), ('sign', 63), ('unlocked', 63), ('review', 63), ('item', 63), ('form', 63), ('receiving', 62), ('hana', 62), ('sep', 62), ('transaction', 62), ('von', 62), ('default', 62), ('web', 62), ('explicit', 62), ('select', 61), ('purchasing', 61), ('source', 61), ('distributor', 61), ('opening', 61), ('disk', 61), ('product', 60), ('fw', 60), ('longer', 60), ('submit', 59), ('hard', 59), ('friday', 59), ('needed', 59), ('driver', 59), ('accounts', 59), ('attach', 59), ('shot', 59), ('monday', 59), ('correctly', 58), ('home', 58), ('days', 58), ('meeting', 58), ('consumed', 58), ('hub', 58), ('external', 58), ('supply', 58), ('hrp', 58), ('approved', 58), ('installation', 58), ('programdnty', 58), ('mails', 57), ('documents', 57), ('search', 56), ('assign', 56), ('incident', 56), ('total', 56), ('query', 55), ('dsw', 55), ('yesterday', 55), ('switch', 55), ('response', 55), ('warning', 55), ('website', 54), ('impact', 54), ('dn', 54), ('destination', 54), ('plm', 54), ('exchange', 54), ('online', 54), ('refer', 54), ('south', 54), ('desk', 54), ('activation', 53), ('she', 53), ('prod', 53), ('currently', 53), ('week', 53), ('chain', 53), ('calls', 53), ('someone', 53), ('download', 53), ('scan', 53), ('load', 53), ('incidents', 53), ('auf', 53), ('audio', 53), ('display', 52), ('expense', 52), ('training', 52), ('advised', 51), ('called', 51), ('finance', 51), ('interaction', 51), ('delete', 51), ('domain', 51), ('processing', 51), ('settings', 50), ('changes', 50), ('options', 50), ('sync', 49), ('copy', 49), ('requested', 49), ('sto', 48), ('payroll', 48), ('works', 48), ('gso', 48), ('person', 48), ('alert', 48), ('desktop', 48), ('rtr', 48), ('first', 48), ('close', 48), ('systems', 48), ('traffic', 48), ('her', 47), ('applications', 47), ('primary', 47), ('calling', 47), ('alerts', 47), ('stock', 46), ('description', 46), ('save', 46), ('sincerely', 46), ('defekt', 46), ('soc', 46), ('wireless', 45), ('netweaver', 45), ('engineer', 45), ('tuesday', 45), ('future', 45), ('responding', 45), ('count', 45), ('authorized', 45), ('gb', 45), ('pdf', 44), ('java', 44), ('deleted', 44), ('mb', 44), ('hours', 43), ('incorrect', 43), ('personal', 43), ('enable', 43), ('word', 43), ('approval', 43), ('daily', 42), ('investigate', 42), ('kann', 42), ('amerirtca', 42), ('seeing', 42), ('ship', 42), ('entered', 42), ('handle', 42), ('ewew', 42), ('assigned', 41), ('bex', 41), ('items', 41), ('remove', 41), ('certificate', 41), ('reason', 41), ('condition', 41), ('importance', 41), ('customers', 41), ('corresponding', 41), ('expired', 40), ('result', 40), ('username', 40), ('partner', 40), ('bobj', 40), ('administrator', 40), ('database', 40), ('floor', 40), ('ws', 40), ('win', 40), ('loading', 39), ('logging', 39), ('cold', 39), ('needful', 39), ('confirm', 39), ('different', 39), ('lhqksbdx', 39), ('ntgdsehl', 39), ('wednesday', 39), ('launch', 39), ('technical', 39), ('free', 39), ('occurrence', 39), ('month', 38), ('pcs', 38), ('post', 38), ('drawings', 38), ('employees', 38), ('current', 38), ('area', 38), ('ping', 38), ('updating', 38), ('failure', 38), ('shared', 38), ('read', 38), ('purposes', 38), ('nothing', 37), ('restart', 37), ('directory', 37), ('analysis', 37), ('teams', 37), ('output', 37), ('cost', 37), ('workflow', 37), ('lean', 37), ('solve', 37), ('facing', 37), ('rxoynvgi', 37), ('supervisor', 37), ('distribution', 37), ('upgrade', 37), ('activity', 37), ('das', 37), ('handling', 37), ('stopped', 36), ('tab', 36), ('hp', 36), ('errors', 36), ('located', 36), ('departments', 36), ('coming', 36), ('den', 36), ('gmbh', 36), ('concerns', 36), ('delegating', 36), ('medium', 36), ('mentioned', 35), ('org', 35), ('appears', 35), ('minutes', 35), ('shop', 35), ('project', 35), ('quote', 35), ('servers', 35), ('reference', 35), ('drucker', 35), ('bei', 35), ('protocol', 35), ('escalation', 35), ('calendar', 34), ('threshold', 34), ('dynamics', 34), ('fixed', 34), ('bkwin', 34), ('text', 34), ('price', 34), ('tax', 34), ('something', 34), ('directly', 34), ('accept', 34), ('differently', 34), ('escalating', 34), ('printed', 33), ('option', 33), ('attendance', 33), ('method', 33), ('thursday', 33), ('happened', 33), ('function', 33), ('dmvpn', 33), ('lock', 33), ('communication', 33), ('connecting', 33), ('organization', 33), ('valid', 33), ('sich', 33), ('aug', 33), ('approve', 32), ('transfer', 32), ('alternate', 32), ('reply', 32), ('box', 32), ('dev', 32), ('case', 32), ('hq', 32), ('temporarily', 32), ('sms', 32), ('mpls', 32), ('control', 32), ('write', 32), ('window', 32), ('drivers', 32), ('ich', 32), ('cvss', 32), ('rule', 32), ('average', 32), ('critical', 31), ('value', 31), ('session', 31), ('etime', 31), ('disconnected', 31), ('him', 31), ('netch', 31), ('conversation', 31), ('effective', 31), ('moved', 31), ('credentials', 31), ('configuration', 31), ('attachments', 31), ('forward', 31), ('rakthyesh', 31), ('malware', 31), ('devices', 30), ('lost', 30), ('lan', 30), ('disclaimer', 30), ('shipping', 30), ('shortly', 30), ('alerting', 30), ('joined', 30), ('provided', 30), ('guest', 30), ('size', 30), ('info', 30), ('people', 30), ('zugriff', 30), ('procedures', 30), ('vid', 30), ('isensor', 30), ('button', 29), ('basis', 29), ('qa', 29), ('point', 29), ('restarted', 29), ('long', 29), ('opened', 29), ('rth', 29), ('bw', 29), ('german', 29), ('sie', 29), ('dst', 29), ('rpc', 29), ('directionality', 29), ('scwx', 29), ('sherlock', 29), ('sle', 29), ('inspector', 29), ('generating', 29), ('share', 28), ('bkbackup', 28), ('assistance', 28), ('manually', 28), ('blue', 28), ('master', 28), ('visitor', 28), ('hxgayczeing', 28), ('room', 28), ('tag', 28), ('return', 28), ('properly', 28), ('bit', 28), ('oetlgbfw', 28), ('bsctrnwp', 28), ('transactions', 28), ('funktioniert', 28), ('batch', 28), ('danke', 28), ('sender', 28), ('src', 28), ('sound', 27), ('drawing', 27), ('wy', 27), ('fail', 27), ('denied', 27), ('sending', 27), ('returned', 27), ('command', 27), ('balancing', 27), ('outbound', 27), ('pricing', 27), ('resources', 27), ('drives', 27), ('url', 27), ('messages', 27), ('configure', 27), ('rechner', 27), ('performance', 27), ('ger', 27), ('operation', 27), ('mailbox', 27), ('creating', 27), ('mehr', 27), ('hotf', 27), ('financial', 27), ('occurred', 27), ('materials', 27), ('turn', 27), ('eb', 27), ('vlan', 27), ('utc', 27), ('proto', 27), ('release', 26), ('colleagues', 26), ('profile', 26), ('worked', 26), ('replacement', 26), ('tracker', 26), ('original', 26), ('reboot', 26), ('earliest', 26), ('recreate', 26), ('nsu', 26), ('latitude', 26), ('gesch', 26), ('amar', 26), ('mitteilung', 26), ('meetings', 25), ('tomorrow', 25), ('jobs', 25), ('ensure', 25), ('starting', 25), ('lockout', 25), ('license', 25), ('msd', 25), ('generated', 25), ('mfg', 25), ('assignment', 25), ('detailed', 25), ('cell', 25), ('saturday', 25), ('performed', 25), ('warm', 25), ('repair', 25), ('anything', 25), ('changing', 25), ('printers', 25), ('sourcing', 25), ('spam', 25), ('ftsf', 25), ('hrer', 25), ('samples', 25), ('score', 25), ('infection', 25), ('php', 25), ('overview', 25), ('disabled', 24), ('shipment', 24), ('detected', 24), ('reported', 24), ('billing', 24), ('mikhghytr', 24), ('discount', 24), ('card', 24), ('updates', 24), ('entering', 24), ('input', 24), ('asked', 24), ('include', 24), ('dat', 24), ('posts', 24), ('empty', 24), ('tiyhum', 24), ('kuyiomar', 24), ('expected', 24), ('nach', 24), ('lhqsm', 24), ('diese', 24), ('packet', 24), ('werden', 24), ('firewall', 24), ('cant', 23), ('urgently', 23), ('tablet', 23), ('setting', 23), ('null', 23), ('attempts', 23), ('node', 23), ('environment', 23), ('respond', 23), ('scanner', 23), ('path', 23), ('general', 23), ('behalf', 23), ('sartlgeo', 23), ('folders', 23), ('cookie', 23), ('saved', 23), ('defective', 23), ('necessary', 23), ('left', 23), ('numbers', 23), ('station', 23), ('determined', 23), ('daypay', 23), ('logged', 23), ('userid', 23), ('exists', 23), ('azbtkqwx', 23), ('hear', 23), ('incoming', 23), ('sind', 23), ('classification', 23), ('ontology', 23), ('wants', 22), ('single', 22), ('experiencing', 22), ('activesync', 22), ('loaded', 22), ('cause', 22), ('records', 22), ('screenshots', 22), ('owned', 22), ('friendly', 22), ('quality', 22), ('weeks', 22), ('key', 22), ('termination', 22), ('ben', 22), ('boot', 22), ('earlier', 22), ('pop', 22), ('procedure', 22), ('snp', 22), ('heu', 22), ('regen', 22), ('bad', 22), ('examples', 22), ('wird', 22), ('ramdntythanjesh', 22), ('antivirus', 22), ('keine', 22), ('israel', 22), ('asset', 22), ('true', 22), ('wenn', 22), ('repeat', 22), ('companysecure', 21), ('entry', 21), ('known', 21), ('gigabitethernet', 21), ('lot', 21), ('accessing', 21), ('comments', 21), ('confirmation', 21), ('passwort', 21), ('mode', 21), ('greet', 21), ('permission', 21), ('removed', 21), ('citrix', 21), ('router', 21), ('including', 21), ('verify', 21), ('secondary', 21), ('symantec', 21), ('restore', 21), ('fields', 21), ('dashbankrd', 21), ('sensor', 21), ('gmt', 21), ('html', 21), ('noris', 21), ('appearing', 20), ('upload', 20), ('markhtyeting', 20), ('recall', 20), ('apo', 20), ('matheywter', 20), ('trouble', 20), ('controller', 20), ('director', 20), ('step', 20), ('appear', 20), ('members', 20), ('azure', 20), ('oct', 20), ('zur', 20), ('extend', 20), ('wu', 20), ('owner', 20), ('contacting', 20), ('validate', 20), ('moment', 20), ('forgot', 20), ('operations', 20), ('russia', 20), ('deu', 20), ('rad', 20), ('quick', 20), ('closed', 20), ('fd', 20), ('knowledge', 20), ('based', 20), ('gru', 20), ('schutzw', 20), ('rdig', 20), ('tos', 20), ('geolocation', 20), ('ago', 19), ('reach', 19), ('object', 19), ('travel', 19), ('connectivity', 19), ('requester', 19), ('zu', 19), ('dns', 19), ('reinstall', 19), ('filesys', 19), ('db', 19), ('block', 19), ('wanted', 19), ('inbox', 19), ('department', 19), ('mp', 19), ('infopath', 19), ('exist', 19), ('battery', 19), ('european', 19), ('deployment', 19), ('android', 19), ('pad', 19), ('duration', 19), ('arc', 19), ('durch', 19), ('malicious', 19), ('goods', 18), ('format', 18), ('manufacturing', 18), ('displayed', 18), ('freezing', 18), ('permissions', 18), ('happening', 18), ('arrange', 18), ('till', 18), ('place', 18), ('senior', 18), ('contacts', 18), ('appropriate', 18), ('causing', 18), ('wdugqatr', 18), ('eng', 18), ('hcm', 18), ('continuing', 18), ('gesendet', 18), ('listed', 18), ('english', 18), ('bis', 18), ('md', 18), ('offline', 18), ('previous', 18), ('notes', 18), ('correlation', 18), ('task', 18), ('emea', 18), ('intended', 18), ('queries', 18), ('determine', 18), ('blocking', 18), ('flag', 18), ('sport', 18), ('itype', 18), ('ttl', 18), ('dgmlen', 18), ('ascii', 18), ('hex', 18), ('att', 18), ('nginx', 18), ('oder', 18), ('dann', 18), ('worm', 18), ('replaced', 17), ('chn', 17), ('stack', 17), ('keybankrd', 17), ('standard', 17), ('viewer', 17), ('tel', 17), ('noise', 17), ('processed', 17), ('cycle', 17), ('rerouted', 17), ('unavailable', 17), ('wnkpzcmv', 17), ('attempting', 17), ('ind', 17), ('mouse', 17), ('plan', 17), ('authentic', 17), ('intune', 17), ('clicks', 17), ('betreff', 17), ('datacenter', 17), ('existing', 17), ('solution', 17), ('lead', 17), ('pgi', 17), ('impacted', 17), ('requesting', 17), ('engine', 17), ('notice', 17), ('mac', 17), ('bank', 17), ('operator', 17), ('payment', 17), ('doc', 17), ('video', 17), ('icon', 17), ('virus', 17), ('replace', 17), ('qty', 17), ('dport', 17), ('icode', 17), ('iplen', 17), ('seq', 17), ('ack', 17), ('tcplen', 17), ('pcap', 17), ('dm', 17), ('references', 17), ('dem', 17), ('destined', 17), ('oracle', 16), ('pick', 16), ('sys', 16), ('usb', 16), ('fastethernet', 16), ('utilization', 16), ('require', 16), ('requests', 16), ('steps', 16), ('pa', 16), ('jionmpsf', 16), ('runs', 16), ('carrier', 16), ('continue', 16), ('pull', 16), ('important', 16), ('edit', 16), ('inform', 16), ('unmonitored', 16), ('planned', 16), ('industrial', 16), ('statistics', 16), ('unauthorized', 16), ('stuck', 16), ('quantity', 16), ('thx', 16), ('agents', 16), ('connections', 16), ('attempt', 16), ('extra', 16), ('des', 16), ('fehlermeldung', 16), ('rqfhiong', 16), ('moving', 16), ('submitted', 16), ('analyst', 16), ('downloading', 16), ('public', 16), ('stefyty', 16), ('framdntyework', 16), ('conference', 16), ('refresh', 16), ('lauacyltoe', 16), ('months', 16), ('hardware', 16), ('cache', 16), ('anbtr', 16), ('nger', 16), ('kein', 16), ('compromised', 16), ('ctoc', 16), ('dieser', 16), ('icmp', 16), ('flash', 15), ('player', 15), ('match', 15), ('gui', 15), ('pay', 15), ('told', 15), ('bwfhtumx', 15), ('japznrvb', 15), ('regional', 15), ('sql', 15), ('vacation', 15), ('flowchart', 15), ('agreement', 15), ('inbound', 15), ('asia', 15), ('entries', 15), ('receipt', 15), ('year', 15), ('fails', 15), ('fyi', 15), ('ecc', 15), ('extended', 15), ('microsoftonline', 15), ('never', 15), ('automatic', 15), ('values', 15), ('contain', 15), ('contacted', 15), ('chg', 15), ('vor', 15), ('psswords', 15), ('names', 15), ('core', 15), ('vzqomdgt', 15), ('locations', 15), ('reaching', 15), ('media', 15), ('glich', 15), ('pieces', 15), ('owa', 15), ('hoti', 15), ('light', 15), ('others', 15), ('notify', 15), ('informed', 15), ('understand', 15), ('frequent', 15), ('adobe', 15), ('frequently', 15), ('dac', 15), ('wg', 15), ('ffnen', 15), ('noch', 15), ('facility', 15), ('tooltion', 15), ('trojan', 15), ('infected', 15), ('length', 15), ('ein', 15), ('corrected', 14), ('pwd', 14), ('requires', 14), ('sites', 14), ('vitalyst', 14), ('channel', 14), ('pos', 14), ('hangs', 14), ('generate', 14), ('requirement', 14), ('table', 14), ('activate', 14), ('restored', 14), ('msg', 14), ('workstation', 14), ('takes', 14), ('koahsriq', 14), ('assistant', 14), ('downloaded', 14), ('record', 14), ('credit', 14), ('inquiry', 14), ('relevant', 14), ('period', 14), ('purchase', 14), ('msonlineservicesteam', 14), ('book', 14), ('direct', 14), ('prompt', 14), ('csr', 14), ('pw', 14), ('dv', 14), ('authorization', 14), ('cases', 14), ('zkwfqagb', 14), ('indicating', 14), ('policy', 14), ('confidential', 14), ('route', 14), ('vogelfontein', 14), ('kis', 14), ('visible', 14), ('alicona', 14), ('question', 14), ('zebra', 14), ('renew', 14), ('patching', 14), ('versions', 14), ('habe', 14), ('allowed', 14), ('legitimate', 14), ('informationen', 14), ('netbios', 14), ('eventid', 14), ('afkstcev', 14), ('utbnkyop', 14), ('schen', 14), ('beim', 14), ('occurrences', 14), ('eines', 14), ('efdl', 14), ('fehler', 14), ('instances', 14), ('secureworks', 14), ('helpdesk', 13), ('couple', 13), ('live', 13), ('duplication', 13), ('self', 13), ('authentication', 13), ('cancel', 13), ('cpu', 13), ('managing', 13), ('loud', 13), ('accounting', 13), ('adding', 13), ('identify', 13), ('broken', 13), ('programdntys', 13), ('technology', 13), ('clicking', 13), ('catalog', 13), ('resetting', 13), ('signed', 13), ('europe', 13), ('dwfiykeo', 13), ('argtxmvcumar', 13), ('opportunities', 13), ('unit', 13), ('sunday', 13), ('qlhmawgi', 13), ('myself', 13), ('north', 13), ('believe', 13), ('jwoqbuml', 13), ('conduct', 13), ('functioning', 13), ('geht', 13), ('red', 13), ('niptbwdq', 13), ('csenjruz', 13), ('eemw', 13), ('picture', 13), ('clear', 13), ('vkzwafuh', 13), ('tcjnuswg', 13), ('activated', 13), ('checking', 13), ('solved', 13), ('pollaurid', 13), ('schedule', 13), ('vom', 13), ('recipient', 13), ('main', 13), ('kommt', 13), ('zwirhcol', 13), ('narzlmfw', 13), ('tuqrvowp', 13), ('fxmzkvqo', 13), ('sold', 13), ('deny', 13), ('signature', 13), ('mapping', 13), ('mozilla', 13), ('einzig', 13), ('allein', 13), ('nutzung', 13), ('adressaten', 13), ('bestimmt', 13), ('enthalten', 13), ('vertraulich', 13), ('geltendem', 13), ('recht', 13), ('offenlegung', 13), ('ausgenommen', 13), ('verbreitung', 13), ('verteilung', 13), ('vervielf', 13), ('ltigung', 13), ('personen', 13), ('denen', 13), ('beabsichtigten', 13), ('empf', 13), ('handelt', 13), ('streng', 13), ('verboten', 13), ('aufgrund', 13), ('versehens', 13), ('ihnen', 13), ('eingegangen', 13), ('benachrichtigen', 13), ('absender', 13), ('misconfiguration', 13), ('chat', 12), ('jpecxuty', 12), ('sv', 12), ('revert', 12), ('variable', 12), ('freezes', 12), ('configured', 12), ('wvngzrca', 12), ('creation', 12), ('stating', 12), ('enabled', 12), ('australia', 12), ('forwarded', 12), ('compliance', 12), ('limited', 12), ('selected', 12), ('forecast', 12), ('follow', 12), ('join', 12), ('rma', 12), ('thing', 12), ('enterprise', 12), ('edt', 12), ('corporate', 12), ('turned', 12), ('jpg', 12), ('sgwipoxn', 12), ('weekly', 12), ('madam', 12), ('final', 12), ('awyl', 12), ('responsible', 12), ('pte', 12), ('queuing', 12), ('aborted', 12), ('mbytes', 12), ('discussed', 12), ('barcode', 12), ('making', 12), ('quarantine', 12), ('vendors', 12), ('normal', 12), ('prompted', 12), ('cb', 12), ('pfzxecbo', 12), ('registered', 12), ('conversion', 12), ('disclosure', 12), ('spread', 12), ('strictly', 12), ('prohibited', 12), ('state', 12), ('cmp', 12), ('phishing', 12), ('awyw', 12), ('hrt', 12), ('ordner', 12), ('fb', 12), ('matheywtyuews', 12), ('auto', 12), ('auch', 12), ('wie', 12), ('wurde', 12), ('kirty', 12), ('opportstorage', 12), ('level', 12), ('mfeyouli', 12), ('uri', 12), ('relay', 12), ('foreseeconndirection', 12), ('outgoing', 12), ('foreseeinternalip', 12), ('inspectoreventid', 12), ('ileatdatacenter', 12), ('foreseemaliciouscomment', 12), ('sherlockruleid', 12), ('outbreak', 12), ('operators', 12), ('functionality', 12), ('thu', 12), ('weaver', 12), ('pbl', 12), ('installing', 11), ('ecwtrjnq', 11), ('zscr', 11), ('dly', 11), ('nor', 11), ('configair', 11), ('root', 11), ('sfmrzdth', 11), ('target', 11), ('paper', 11), ('observing', 11), ('export', 11), ('resolution', 11), ('gjtyswkb', 11), ('dpvaymxr', 11), ('clock', 11), ('private', 11), ('davidthd', 11), ('computers', 11), ('repeated', 11), ('tasks', 11), ('speaker', 11), ('groups', 11), ('bergehend', 11), ('extract', 11), ('plug', 11), ('labels', 11), ('availability', 11), ('synchronizing', 11), ('terminate', 11), ('bios', 11), ('logistics', 11), ('perform', 11), ('rate', 11), ('ooo', 11), ('background', 11), ('monthly', 11), ('viele', 11), ('forms', 11), ('applicable', 11), ('rebooted', 11), ('template', 11), ('products', 11), ('blktuiae', 11), ('jzakfmhw', 11), ('law', 11), ('infrastructure', 11), ('nx', 11), ('bar', 11), ('emp', 11), ('usd', 11), ('fe', 11), ('launching', 11), ('alwaysupservice', 11), ('secure', 11), ('header', 11), ('amerirtcas', 11), ('quotes', 11), ('aus', 11), ('personnel', 11), ('dthyan', 11), ('worklist', 11), ('clients', 11), ('kls', 11), ('passive', 11), ('dhcpd', 11), ('dhcpack', 11), ('eth', 11), ('lease', 11), ('irreceivedtime', 11), ('fax', 11), ('acl', 11), ('sinkhole', 11), ('execute', 11), ('eine', 11), ('processes', 11), ('wieder', 11), ('dropping', 11), ('vulnerability', 11), ('autoresolve', 11), ('cms', 11), ('ebhsm', 11), ('singapore', 11), ('static', 10), ('speakers', 10), ('intermittent', 10), ('prtgghj', 10), ('acces', 10), ('rights', 10), ('apps', 10), ('map', 10), ('white', 10), ('logs', 10), ('aware', 10), ('pbx', 10), ('module', 10), ('wrcktgbd', 10), ('wzrgyunp', 10), ('misplaced', 10), ('talk', 10), ('sso', 10), ('safe', 10), ('subject', 10), ('toolmail', 10), ('detecting', 10), ('posting', 10), ('hour', 10), ('dba', 10), ('synchronisierung', 10), ('ihrem', 10), ('blockiert', 10), ('charging', 10), ('erro', 10), ('routing', 10), ('rectify', 10), ('toolkuznetsk', 10), ('hope', 10), ('vl', 10), ('rgds', 10), ('hardpoint', 10), ('major', 10), ('inventory', 10), ('states', 10), ('noticed', 10), ('poland', 10), ('special', 10), ('helped', 10), ('scanning', 10), ('defined', 10), ('big', 10), ('ports', 10), ('konto', 10), ('prints', 10), ('zsd', 10), ('title', 10), ('looked', 10), ('booting', 10), ('aa', 10), ('reinstalled', 10), ('kxmidsga', 10), ('zokivdfa', 10), ('shots', 10), ('capture', 10), ('attempted', 10), ('eh', 10), ('component', 10), ('cad', 10), ('weekend', 10), ('consultant', 10), ('unknown', 10), ('wgq', 10), ('dank', 10), ('shatryung', 10), ('shut', 10), ('reached', 10), ('archiving', 10), ('role', 10), ('extension', 10), ('epmsystem', 10), ('mir', 10), ('choose', 10), ('min', 10), ('slo', 10), ('revenue', 10), ('colleague', 10), ('cfibdamq', 10), ('ber', 10), ('lines', 10), ('july', 10), ('itself', 10), ('ssen', 10), ('alive', 10), ('eventtypeid', 10), ('ontologyid', 10), ('srchostname', 10), ('inspectorruleid', 10), ('agentid', 10), ('ctainstanceid', 10), ('logtimestamp', 10), ('unblock', 10), ('suspicious', 10), ('assigning', 10), ('gte', 10), ('bin', 10), ('encoding', 10), ('bf', 10), ('nkthumgf', 10), ('mwgdenbs', 10), ('herr', 10), ('lhqsid', 10), ('ccgslb', 10), ('addresses', 10), ('directive', 10), ('category', 10), ('gvderpbx', 10), ('udrzjxkm', 10), ('propagating', 10), ('inactive', 10), ('disable', 9), ('closing', 9), ('bwhrattr', 9), ('views', 9), ('benefits', 9), ('invoke', 9), ('util', 9), ('iterator', 9), ('base', 9), ('finally', 9), ('great', 9), ('robhyertyj', 9), ('observed', 9), ('filter', 9), ('present', 9), ('azxhejvq', 9), ('fyemlavd', 9), ('dept', 9), ('administration', 9), ('rjanhbde', 9), ('owfkyjcp', 9), ('broadband', 9), ('centers', 9), ('delegation', 9), ('requisition', 9), ('objects', 9), ('erpgui', 9), ('docking', 9), ('nvyjtmca', 9), ('xjhpznds', 9), ('zuxcfonv', 9), ('nyhpkrbe', 9), ('award', 9), ('wly', 9), ('displaying', 9), ('region', 9), ('jokidavy', 9), ('posted', 9), ('register', 9), ('raise', 9), ('become', 9), ('impossible', 9), ('upgraded', 9), ('mitarbeiter', 9), ('cable', 9), ('released', 9), ('rmb', 9), ('locallist', 9), ('johthryugftyson', 9), ('advice', 9), ('bb', 9), ('alle', 9), ('prompting', 9), ('hash', 9), ('shipped', 9), ('addin', 9), ('ctzykflo', 9), ('evzbhgru', 9), ('judthtihty', 9), ('zhuyhts', 9), ('africa', 9), ('enclosed', 9), ('immediate', 9), ('fixing', 9), ('vielen', 9), ('shift', 9), ('ext', 9), ('finished', 9), ('human', 9), ('tech', 9), ('docked', 9), ('headset', 9), ('instructions', 9), ('runtime', 9), ('ability', 9), ('mins', 9), ('immer', 9), ('replicated', 9), ('exact', 9), ('figure', 9), ('ughzilfm', 9), ('fen', 9), ('currency', 9), ('addressed', 9), ('discounts', 9), ('ribbon', 9), ('duplicate', 9), ('screens', 9), ('nan', 9), ('complaint', 9), ('uezonywf', 9), ('rldbvipu', 9), ('difference', 9), ('successful', 9), ('aksthyuhath', 9), ('shettythruy', 9), ('ipc', 9), ('caused', 9), ('memory', 9), ('uyrpdvoq', 9), ('mbzevtcx', 9), ('wir', 9), ('dstport', 9), ('srcport', 9), ('evaluationmodels', 9), ('eventtypepriority', 9), ('china', 9), ('connc', 9), ('hosts', 9), ('verbindung', 9), ('msie', 9), ('specific', 9), ('folgende', 9), ('apul', 9), ('sinkholed', 9), ('tigt', 9), ('isensplant', 9), ('bom', 9), ('ryculmsd', 9), ('wofgvkrb', 9), ('iso', 9), ('deliveries', 9), ('kirtyled', 9), ('misconfigured', 9), ('wp', 9), ('teamordner', 9), ('queue', 8), ('rfc', 8), ('wkly', 8), ('accepting', 8), ('kiosk', 8), ('occurs', 8), ('forgotten', 8), ('mistake', 8), ('notebook', 8), ('restarting', 8), ('grinding', 8), ('advance', 8), ('rdfjsawg', 8), ('zpmxgdcw', 8), ('crashes', 8), ('behavior', 8), ('sponsor', 8), ('uploading', 8), ('administrative', 8), ('hand', 8), ('beshryu', 8), ('prtqx', 8), ('transferred', 8), ('reactivate', 8), ('professional', 8), ('submitting', 8), ('retrieve', 8), ('deletion', 8), ('samsung', 8), ('kbclinop', 8), ('vsczklfp', 8), ('mobility', 8), ('loss', 8), ('personally', 8), ('pulling', 8), ('gdnwlkit', 8), ('balance', 8), ('tige', 8), ('similar', 8), ('raised', 8), ('pfjwinbg', 8), ('ljtzbdqg', 8), ('hot', 8), ('lpoebzsc', 8), ('grknswyo', 8), ('library', 8), ('serial', 8), ('remotely', 8), ('completely', 8), ('xwirzvda', 8), ('okhyipgr', 8), ('laserjet', 8), ('country', 8), ('building', 8), ('pwr', 8), ('hmc', 8), ('invalid', 8), ('higher', 8), ('happen', 8), ('persists', 8), ('picking', 8), ('meine', 8), ('gesperrt', 8), ('companyguest', 8), ('temp', 8), ('empw', 8), ('dpuifqeo', 8), ('eagcldaten', 8), ('csd', 8), ('xlsx', 8), ('import', 8), ('insert', 8), ('tracking', 8), ('audit', 8), ('bmudkpie', 8), ('qolrvbip', 8), ('suddenly', 8), ('itnakpmc', 8), ('kpm', 8), ('sao', 8), ('ptygkvzl', 8), ('difficult', 8), ('supplied', 8), ('waiting', 8), ('directors', 8), ('prompts', 8), ('recipients', 8), ('bankrd', 8), ('planning', 8), ('wktesmbp', 8), ('lorjymef', 8), ('feedback', 8), ('hydstheud', 8), ('mddwwyleh', 8), ('ramdnty', 8), ('shipments', 8), ('sample', 8), ('large', 8), ('possibility', 8), ('dir', 8), ('lists', 8), ('gt', 8), ('distributors', 8), ('pradtheyp', 8), ('gew', 8), ('johthryu', 8), ('workers', 8), ('keheu', 8), ('prior', 8), ('laser', 8), ('archive', 8), ('actual', 8), ('separate', 8), ('delay', 8), ('failing', 8), ('zum', 8), ('chrome', 8), ('sort', 8), ('real', 8), ('hold', 8), ('hat', 8), ('ios', 8), ('meldung', 8), ('resource', 8), ('asks', 8), ('logic', 8), ('taiwan', 8), ('insufficient', 8), ('donnerstag', 8), ('indicator', 8), ('reading', 8), ('expenses', 8), ('acct', 8), ('paramdntyeter', 8), ('kurtyar', 8), ('useid', 8), ('robot', 8), ('udp', 8), ('switzerland', 8), ('mscrm', 8), ('einrichten', 8), ('prt', 8), ('anubis', 8), ('probable', 8), ('dce', 8), ('compatible', 8), ('inq', 8), ('telefon', 8), ('skv', 8), ('freigabe', 8), ('targeted', 8), ('wysiwyg', 8), ('wdwxhcml', 8), ('szv', 8), ('mal', 8), ('pers', 8), ('michghytuael', 8), ('nnen', 8), ('freischalten', 8), ('deutschland', 8), ('oneteam', 8), ('worms', 8), ('blaster', 8), ('msblast', 8), ('lovsan', 8), ('welchia', 8), ('nachi', 8), ('reatle', 8), ('wanrtyg', 8), ('ngm', 8), ('aghw', 8), ('lpal', 8), ('comercial', 8), ('employment', 7), ('payslips', 7), ('reasons', 7), ('stored', 7), ('bring', 7), ('supported', 7), ('jcoerpmanager', 7), ('omforiginalsexport', 7), ('rollfgyuej', 7), ('amssm', 7), ('expires', 7), ('continuously', 7), ('clicked', 7), ('virtual', 7), ('refreshing', 7), ('functions', 7), ('sept', 7), ('kd', 7), ('quarantined', 7), ('crashing', 7), ('cksetzen', 7), ('fmzdkyqv', 7), ('dbrslnhe', 7), ('ndigung', 7), ('trust', 7), ('indirect', 7), ('ftp', 7), ('sbgvrncj', 7), ('idfhtoqv', 7), ('dierppeared', 7), ('osjqfbvw', 7), ('hlmgrfpx', 7), ('wk', 7), ('zip', 7), ('mbps', 7), ('instruction', 7), ('terminated', 7), ('tmqfjard', 7), ('qzhgdoua', 7), ('hpqc', 7), ('links', 7), ('costs', 7), ('saving', 7), ('ready', 7), ('analytics', 7), ('msc', 7), ('doubts', 7), ('doesnt', 7), ('touch', 7), ('startup', 7), ('suggested', 7), ('press', 7), ('uwe', 7), ('lib', 7), ('powerpoint', 7), ('awards', 7), ('position', 7), ('bytes', 7), ('powder', 7), ('mexico', 7), ('thought', 7), ('emailed', 7), ('planner', 7), ('programdntyme', 7), ('speed', 7), ('plugin', 7), ('cif', 7), ('slowly', 7), ('leaving', 7), ('filled', 7), ('analyzer', 7), ('adapter', 7), ('approvals', 7), ('trial', 7), ('validity', 7), ('eglwsfkn', 7), ('invoicing', 7), ('uninstalled', 7), ('pro', 7), ('initial', 7), ('staff', 7), ('drop', 7), ('handlers', 7), ('scriptmanager', 7), ('glovia', 7), ('capacity', 7), ('cleared', 7), ('ytqhfmwi', 7), ('disconnecting', 7), ('zywoxerf', 7), ('paqxtrfk', 7), ('cancelled', 7), ('improvement', 7), ('sk', 7), ('registergerirtcht', 7), ('rfwlsoej', 7), ('yvtjzkaw', 7), ('solely', 7), ('addressee', 7), ('mistaking', 7), ('united', 7), ('stay', 7), ('phones', 7), ('freundlichem', 7), ('fan', 7), ('huge', 7), ('possibly', 7), ('rest', 7), ('hall', 7), ('uplink', 7), ('xfdkwusj', 7), ('gyklresa', 7), ('sev', 7), ('proper', 7), ('contract', 7), ('lunch', 7), ('browse', 7), ('excluded', 7), ('ora', 7), ('seit', 7), ('intercompany', 7), ('loop', 7), ('gtehdnyu', 7), ('vh', 7), ('marfhtyio', 7), ('dates', 7), ('chance', 7), ('checks', 7), ('miro', 7), ('validation', 7), ('appreciated', 7), ('coating', 7), ('karghyuen', 7), ('counter', 7), ('build', 7), ('pcl', 7), ('advanced', 7), ('engg', 7), ('abend', 7), ('modify', 7), ('hrs', 7), ('belongs', 7), ('features', 7), ('busy', 7), ('marftgytin', 7), ('spreadsheet', 7), ('proceed', 7), ('permanent', 7), ('things', 7), ('potential', 7), ('specification', 7), ('selecting', 7), ('signal', 7), ('facilities', 7), ('history', 7), ('prtqv', 7), ('kingdom', 7), ('ssl', 7), ('operating', 7), ('roles', 7), ('drops', 7), ('avoid', 7), ('sabrthy', 7), ('custom', 7), ('sharing', 7), ('addition', 7), ('vfx', 7), ('ndobtzpw', 7), ('demand', 7), ('unique', 7), ('rejected', 7), ('snapshot', 7), ('double', 7), ('ntsowaem', 7), ('jfgslyde', 7), ('erirtc', 7), ('endpoint', 7), ('protection', 7), ('rgtarthi', 7), ('erjgypa', 7), ('spf', 7), ('google', 7), ('smtp', 7), ('returns', 7), ('kathght', 7), ('shfhyw', 7), ('nealxjbc', 7), ('owjduxai', 7), ('riqmdnzs', 7), ('mtlghwex', 7), ('maus', 7), ('muss', 7), ('kb', 7), ('laufwerk', 7), ('damage', 7), ('execution', 7), ('cve', 7), ('bzw', 7), ('mich', 7), ('crkdjbot', 7), ('qiztrxne', 7), ('kannst', 7), ('ansi', 7), ('zollerfgh', 7), ('qngschtz', 7), ('member', 7), ('manual', 7), ('usage', 7), ('freundliche', 7), ('lassen', 7), ('siehe', 7), ('dsccache', 7), ('divestiture', 7), ('rerun', 7), ('pollaurido', 7), ('projekte', 7), ('package', 7), ('encryption', 7), ('hw', 7), ('linnes', 7), ('linnemann', 7), ('seconds', 6), ('picked', 6), ('locking', 6), ('switches', 6), ('originals', 6), ('ha', 6), ('jvpkulxw', 6), ('assume', 6), ('aes', 6), ('apply', 6), ('copied', 6), ('hxgayczeed', 6), ('consignment', 6), ('confirmations', 6), ('years', 6), ('vice', 6), ('troubleshooting', 6), ('means', 6), ('accepted', 6), ('obanjrhg', 6), ('otc', 6), ('synchronized', 6), ('identified', 6), ('fwd', 6), ('dialog', 6), ('development', 6), ('wafglhdrhjop', 6), ('abap', 6), ('inhekdol', 6), ('lmsl', 6), ('optiplex', 6), ('conf', 6), ('intranet', 6), ('playing', 6), ('success', 6), ('bujiesrg', 6), ('zopcrshl', 6), ('faced', 6), ('leave', 6), ('president', 6), ('permanently', 6), ('rqxw', 6), ('ignore', 6), ('managers', 6), ('gigaset', 6), ('mass', 6), ('steel', 6), ('selection', 6), ('lwizucan', 6), ('zvnxlobq', 6), ('half', 6), ('vmsliazh', 6), ('ltksxmyv', 6), ('reminder', 6), ('fault', 6), ('named', 6), ('crash', 6), ('gen', 6), ('himghtmelreich', 6), ('routed', 6), ('salary', 6), ('ohdrnswl', 6), ('rezuibdt', 6), ('shopfloor', 6), ('approver', 6), ('usas', 6), ('pinging', 6), ('suspect', 6), ('menu', 6), ('bihfazru', 6), ('bhjqvtzm', 6), ('buttons', 6), ('zhwmifvx', 6), ('missed', 6), ('conditions', 6), ('parts', 6), ('toold', 6), ('activities', 6), ('corp', 6), ('byclpwmv', 6), ('esafrtbh', 6), ('pending', 6), ('res', 6), ('sequence', 6), ('bma', 6), ('quoting', 6), ('tooling', 6), ('profit', 6), ('registration', 6), ('city', 6), ('short', 6), ('neue', 6), ('specialist', 6), ('verzeichnis', 6), ('pops', 6), ('shutting', 6), ('xerox', 6), ('ktghvuwr', 6), ('uwtakcmj', 6), ('push', 6), ('npc', 6), ('pass', 6), ('qidgvtwa', 6), ('qvbutayx', 6), ('grade', 6), ('canceled', 6), ('established', 6), ('evhw', 6), ('storage', 6), ('assembly', 6), ('concerned', 6), ('webpage', 6), ('qasdhyzm', 6), ('yuglsrwx', 6), ('maximum', 6), ('fyzceglp', 6), ('nuhfwplj', 6), ('ojcwxser', 6), ('associated', 6), ('launcher', 6), ('exited', 6), ('difficulty', 6), ('bob', 6), ('def', 6), ('chk', 6), ('threat', 6), ('ilypdtno', 6), ('machining', 6), ('hybiaxlk', 6), ('lawptzir', 6), ('cisco', 6), ('earthworks', 6), ('reader', 6), ('thanking', 6), ('ground', 6), ('charge', 6), ('applied', 6), ('synchronization', 6), ('ecs', 6), ('centre', 6), ('expire', 6), ('attaching', 6), ('attributes', 6), ('recognized', 6), ('dan', 6), ('siemens', 6), ('ewewx', 6), ('neuen', 6), ('unter', 6), ('einen', 6), ('tier', 6), ('jeffrghryey', 6), ('issued', 6), ('automated', 6), ('dqplrwoy', 6), ('cutpwjie', 6), ('nxd', 6), ('alte', 6), ('machines', 6), ('eagw', 6), ('absolutely', 6), ('designer', 6), ('rep', 6), ('trade', 6), ('party', 6), ('transport', 6), ('hit', 6), ('feel', 6), ('billings', 6), ('newly', 6), ('reimbursement', 6), ('partial', 6), ('neu', 6), ('mgr', 6), ('play', 6), ('cust', 6), ('incomplete', 6), ('bxeagsmt', 6), ('zrwdgsco', 6), ('finding', 6), ('zdcheloy', 6), ('ppt', 6), ('world', 6), ('jb', 6), ('disconnection', 6), ('utilized', 6), ('canada', 6), ('tess', 6), ('allowing', 6), ('committed', 6), ('eastern', 6), ('funktionieren', 6), ('markhty', 6), ('lacw', 6), ('ck', 6), ('boxes', 6), ('nicrhty', 6), ('confirming', 6), ('adds', 6), ('snipping', 6), ('usaed', 6), ('zone', 6), ('deploy', 6), ('reverse', 6), ('universal', 6), ('idea', 6), ('txt', 6), ('visit', 6), ('namprd', 6), ('cvd', 6), ('movement', 6), ('lisbon', 6), ('nop', 6), ('gzip', 6), ('situation', 6), ('certified', 6), ('kugwsrjz', 6), ('xnygwtle', 6), ('allinvest', 6), ('anmeldung', 6), ('thomklmas', 6), ('logins', 6), ('returning', 6), ('cec', 6), ('richtig', 6), ('pvd', 6), ('tologin', 6), ('ehs', 6), ('newflv', 6), ('sohu', 6), ('mon', 6), ('ita', 6), ('milano', 6), ('domains', 6), ('sst', 6), ('srinfhyath', 6), ('vulnerable', 6), ('attackers', 6), ('popularity', 6), ('zw', 6), ('uyw', 6), ('fq', 6), ('shutdown', 6), ('daten', 6), ('sitz', 6), ('sheet', 6), ('imts', 6), ('aber', 6), ('tips', 6), ('guten', 6), ('paramdntyeters', 6), ('dfiyvmec', 6), ('wxioadpt', 6), ('dane', 6), ('williuthyr', 6), ('rgtw', 6), ('gecko', 6), ('news', 6), ('common', 6), ('jul', 6), ('historically', 6), ('propagate', 6), ('epmap', 6), ('mapper', 6), ('locator', 6), ('datagramdnty', 6), ('smb', 6), ('supposed', 6), ('failures', 6), ('zdsxmcwu', 6), ('wh', 6), ('assylias', 6), ('tor', 6), ('palo', 6), ('ptuchwad', 6), ('yzvrlcqa', 6), ('jctnelqs', 6), ('lansuiwe', 6), ('karaffa', 6), ('calculation', 6), ('tdmgolwn', 6), ('qwijaspo', 6), ('ukynmfig', 6), ('srv', 6), ('kxvwsatr', 6), ('nmywsqrg', 6), ('kds', 6), ('evhl', 6), ('zqbmxdgy', 6), ('stuwbacm', 6), ('reputation', 6), ('sogou', 6), ('cadastra', 6), ('miiserver', 6), ('deliver', 5), ('mx', 5), ('accessible', 5), ('cadagent', 5), ('bda', 5), ('expiring', 5), ('contents', 5), ('stays', 5), ('su', 5), ('scanned', 5), ('shi', 5), ('acc', 5), ('recover', 5), ('areas', 5), ('writing', 5), ('scrap', 5), ('parent', 5), ('acknowledgement', 5), ('leads', 5), ('delivered', 5), ('azurewebsites', 5), ('sfb', 5), ('everytime', 5), ('stops', 5), ('begin', 5), ('anpocezt', 5), ('qturbxsg', 5), ('maintain', 5), ('sehr', 5), ('anvqzdif', 5), ('mailboxes', 5), ('font', 5), ('ipbl', 5), ('updation', 5), ('entire', 5), ('mention', 5), ('catalogue', 5), ('qdxyifhj', 5), ('zbwtunpy', 5), ('memo', 5), ('mam', 5), ('guard', 5), ('temporary', 5), ('removing', 5), ('ewel', 5), ('respect', 5), ('flow', 5), ('spaces', 5), ('yzugpdco', 5), ('nsyapewg', 5), ('involved', 5), ('cyber', 5), ('ransomware', 5), ('purchasingupstreamsso', 5), ('samaccountname', 5), ('november', 5), ('dienstag', 5), ('prevent', 5), ('met', 5), ('odbc', 5), ('recommended', 5), ('logo', 5), ('aidl', 5), ('hd', 5), ('rung', 5), ('rrc', 5), ('range', 5), ('skmdgnuh', 5), ('utgclesd', 5), ('timed', 5), ('packets', 5), ('operational', 5), ('yellow', 5), ('mi', 5), ('tsicojkp', 5), ('kghaozew', 5), ('pre', 5), ('inxsupmy', 5), ('included', 5), ('csrs', 5), ('indexing', 5), ('frozen', 5), ('hardcopy', 5), ('expedite', 5), ('wait', 5), ('constantly', 5), ('zlettel', 5), ('encounter', 5), ('older', 5), ('rmt', 5), ('technician', 5), ('cookies', 5), ('zeit', 5), ('design', 5), ('del', 5), ('assignments', 5), ('grind', 5), ('nummer', 5), ('air', 5), ('ovhtgsxd', 5), ('dcqhnrmy', 5), ('weight', 5), ('physical', 5), ('benutzer', 5), ('art', 5), ('greatly', 5), ('color', 5), ('walkme', 5), ('precision', 5), ('turkey', 5), ('prove', 5), ('imei', 5), ('exception', 5), ('sha', 5), ('scriptresourcehandler', 5), ('severity', 5), ('terms', 5), ('eqxyvfpi', 5), ('gbaljypo', 5), ('campo', 5), ('jirecvta', 5), ('executive', 5), ('bay', 5), ('yegzbvru', 5), ('sale', 5), ('stra', 5), ('track', 5), ('temperature', 5), ('remember', 5), ('nda', 5), ('laptops', 5), ('low', 5), ('forwarding', 5), ('zaf', 5), ('umzcxfah', 5), ('aoshpjiu', 5), ('sthyuraj', 5), ('sektyhar', 5), ('unsuccessful', 5), ('archived', 5), ('figures', 5), ('itclukpe', 5), ('aimcfeko', 5), ('tauschen', 5), ('served', 5), ('purpose', 5), ('resolving', 5), ('authorisation', 5), ('gmhkdsnw', 5), ('late', 5), ('nov', 5), ('sridthshar', 5), ('herytur', 5), ('factory', 5), ('nathyresh', 5), ('pradyhtueep', 5), ('yyufs', 5), ('anti', 5), ('fmxcnwpu', 5), ('tcwrdqboinition', 5), ('remains', 5), ('answer', 5), ('referenced', 5), ('copying', 5), ('bhayhtrathramdnty', 5), ('mamilujli', 5), ('recognize', 5), ('sim', 5), ('sch', 5), ('ploxzuts', 5), ('utvimnwo', 5), ('gestern', 5), ('crashed', 5), ('trurthyuft', 5), ('visio', 5), ('notifications', 5), ('incorrectly', 5), ('totally', 5), ('voucher', 5), ('mismatch', 5), ('partners', 5), ('blinking', 5), ('models', 5), ('ebusiness', 5), ('prtor', 5), ('ikerxqwz', 5), ('prkyuitl', 5), ('starts', 5), ('vf', 5), ('pulverleitstand', 5), ('kaguhxwo', 5), ('uoyipxqg', 5), ('haben', 5), ('maschine', 5), ('fbyusmxz', 5), ('kxvmcbly', 5), ('june', 5), ('dealer', 5), ('column', 5), ('regular', 5), ('dierppear', 5), ('sein', 5), ('receipts', 5), ('compatibility', 5), ('versucht', 5), ('santthyumar', 5), ('shtyhant', 5), ('werk', 5), ('bereits', 5), ('shrugott', 5), ('tyhuellis', 5), ('dropped', 5), ('synching', 5), ('participants', 5), ('zfburidj', 5), ('jmilguev', 5), ('qgrbnjiu', 5), ('hidzlfma', 5), ('identifying', 5), ('credential', 5), ('life', 5), ('force', 5), ('inconvenience', 5), ('aevzsogn', 5), ('peojqgvm', 5), ('qayeptuo', 5), ('ndern', 5), ('oss', 5), ('druckt', 5), ('datenbank', 5), ('script', 5), ('delta', 5), ('extraction', 5), ('unauthorised', 5), ('built', 5), ('filters', 5), ('swap', 5), ('nmpworvu', 5), ('trusted', 5), ('clhqsm', 5), ('cloud', 5), ('hsh', 5), ('xsrkthvf', 5), ('nderung', 5), ('bls', 5), ('aborting', 5), ('tells', 5), ('jxgobwrm', 5), ('qkugdipo', 5), ('rekwlqmu', 5), ('panel', 5), ('solutions', 5), ('hrb', 5), ('downloader', 5), ('yahoo', 5), ('lhql', 5), ('foreseeexternalip', 5), ('nld', 5), ('vendorpriority', 5), ('isp', 5), ('familiar', 5), ('dock', 5), ('dg', 5), ('iu', 5), ('reconfigured', 5), ('launched', 5), ('compensation', 5), ('gergryth', 5), ('cpic', 5), ('anivdcor', 5), ('rbmfhiox', 5), ('awareness', 5), ('privileged', 5), ('exempt', 5), ('peer', 5), ('concern', 5), ('gdhyrts', 5), ('muggftyali', 5), ('authority', 5), ('notwkdgr', 5), ('zvmesjpt', 5), ('undeliverable', 5), ('selector', 5), ('antispam', 5), ('permitted', 5), ('pds', 5), ('smxoklny', 5), ('hbecskgl', 5), ('xwertljy', 5), ('zrmlhkyq', 5), ('cde', 5), ('experience', 5), ('bluetooth', 5), ('videos', 5), ('plants', 5), ('oe', 5), ('scjxobhd', 5), ('ldypjkmf', 5), ('sproc', 5), ('wlan', 5), ('durchwahl', 5), ('papers', 5), ('datei', 5), ('indonesia', 5), ('jashyht', 5), ('dallas', 5), ('distributes', 5), ('germanytially', 5), ('parties', 5), ('effectively', 5), ('choice', 5), ('exploit', 5), ('attacks', 5), ('exploitation', 5), ('lid', 5), ('alr', 5), ('grugermany', 5), ('wrote', 5), ('deeghyupak', 5), ('leider', 5), ('frau', 5), ('sab', 5), ('morgen', 5), ('jgnxyahz', 5), ('cixzwuyf', 5), ('ryafbthn', 5), ('lhbsm', 5), ('rules', 5), ('relationship', 5), ('contained', 5), ('bosch', 5), ('anhang', 5), ('screensaver', 5), ('practices', 5), ('flags', 5), ('syn', 5), ('pkdavqwt', 5), ('tafrmxsh', 5), ('hier', 5), ('callie', 5), ('hilfe', 5), ('dot', 5), ('thdjzolwronization', 5), ('linked', 5), ('lbdw', 5), ('edml', 5), ('mein', 5), ('mercedes', 5), ('man', 5), ('herren', 5), ('gnasmtvx', 5), ('cwxtsvkm', 5), ('integrity', 5), ('ndig', 5), ('vfrdxtqw', 5), ('jfbmsenz', 5), ('zusammen', 5), ('survey', 5), ('wqfzjycu', 5), ('omleknjd', 5), ('franhtyu', 5), ('typing', 5), ('bmhxwvys', 5), ('victim', 5), ('vulnerabilities', 5), ('uiu', 5), ('srcassetofinterest', 5), ('xed', 5), ('plugins', 5), ('inboundio', 5), ('csv', 5), ('uploader', 5), ('nicolmghyu', 5), ('couskjgd', 5), ('hmjdrvpb', 4), ('komuaywn', 4), ('dhcp', 4), ('reroute', 4), ('scm', 4), ('enquiry', 4), ('ekpo', 4), ('trace', 4), ('conn', 4), ('jco', 4), ('ihkolepb', 4), ('ozhnjyef', 4), ('gain', 4), ('ovuweygj', 4), ('scratch', 4), ('outbox', 4), ('acrobat', 4), ('pozna', 4), ('aez', 4), ('educated', 4), ('guide', 4), ('insurance', 4), ('ends', 4), ('inspection', 4), ('websites', 4), ('environments', 4), ('prospect', 4), ('jan', 4), ('ygkzwsud', 4), ('cvjgkxws', 4), ('zartupsw', 4), ('signing', 4), ('babiluntr', 4), ('eagl', 4), ('throws', 4), ('zlqfptjx', 4), ('xnklbfua', 4), ('small', 4), ('cnc', 4), ('soft', 4), ('ynsqjehx', 4), ('kqgrsawl', 4), ('communications', 4), ('opens', 4), ('statements', 4), ('vsp', 4), ('reassign', 4), ('head', 4), ('reminders', 4), ('ivnhumzjalakrisyuhnyrtn', 4), ('puxsvfwr', 4), ('cwkjruni', 4), ('njhaqket', 4), ('writes', 4), ('btvmxdfc', 4), ('yfahetsc', 4), ('maintained', 4), ('ya', 4), ('globalengservices', 4), ('poncacity', 4), ('mitgckqf', 4), ('ufpwmybi', 4), ('lowercase', 4), ('gdkiehbr', 4), ('kdithjsr', 4), ('filtering', 4), ('eluvxqhw', 4), ('gpbfkqeu', 4), ('calculating', 4), ('investigating', 4), ('feature', 4), ('dtlmbcrx', 4), ('mwuateyx', 4), ('transportation', 4), ('quattro', 4), ('agreements', 4), ('occured', 4), ('codes', 4), ('slip', 4), ('points', 4), ('csqe', 4), ('infonet', 4), ('flashing', 4), ('vb', 4), ('central', 4), ('cut', 4), ('slrgconp', 4), ('calculator', 4), ('interrupted', 4), ('izwtdnfq', 4), ('xptuoaid', 4), ('kijhcwur', 4), ('wzs', 4), ('complaining', 4), ('pesonal', 4), ('portion', 4), ('synch', 4), ('vat', 4), ('documentation', 4), ('auftr', 4), ('suite', 4), ('itens', 4), ('implementation', 4), ('controlling', 4), ('class', 4), ('auditor', 4), ('roboworker', 4), ('faulty', 4), ('trgqbeax', 4), ('hfyzudql', 4), ('azubis', 4), ('unsere', 4), ('ibm', 4), ('pmr', 4), ('console', 4), ('cantabria', 4), ('stat', 4), ('iak', 4), ('quotation', 4), ('lagp', 4), ('circle', 4), ('outputs', 4), ('soldfnbq', 4), ('uhnbsvqd', 4), ('passw', 4), ('hang', 4), ('consultants', 4), ('coast', 4), ('cpp', 4), ('programdntym', 4), ('holemaking', 4), ('executing', 4), ('constructed', 4), ('legal', 4), ('wild', 4), ('projector', 4), ('commstorage', 4), ('favor', 4), ('referring', 4), ('ideas', 4), ('jet', 4), ('keeping', 4), ('backorder', 4), ('keys', 4), ('tim', 4), ('qs', 4), ('wvdxnkhf', 4), ('manipulate', 4), ('measuring', 4), ('rtpcnyhq', 4), ('ceqmwkhi', 4), ('gkzedilm', 4), ('tkpfumeb', 4), ('repeatedly', 4), ('max', 4), ('apple', 4), ('ntuhoafg', 4), ('bzwefjvk', 4), ('joiner', 4), ('formatheywted', 4), ('alarm', 4), ('aqrhwjgo', 4), ('cyelqkvs', 4), ('dabhruji', 4), ('mnlvhtug', 4), ('imvetgoa', 4), ('publications', 4), ('projects', 4), ('ilbkhgxd', 4), ('hirsqytd', 4), ('mxjcnqfs', 4), ('finish', 4), ('richoscan', 4), ('mtb', 4), ('adaptor', 4), ('elengineering', 4), ('effect', 4), ('bld', 4), ('lrrsm', 4), ('sitting', 4), ('generirtc', 4), ('popup', 4), ('tabs', 4), ('kicked', 4), ('refreshed', 4), ('ecoljnvt', 4), ('lbdqmvfs', 4), ('adjust', 4), ('awb', 4), ('complained', 4), ('verification', 4), ('intermittently', 4), ('vnjdghui', 4), ('synced', 4), ('produktion', 4), ('jvxtfhkg', 4), ('petrghada', 4), ('alt', 4), ('postings', 4), ('triggering', 4), ('christgry', 4), ('knocked', 4), ('charged', 4), ('timeout', 4), ('regularly', 4), ('modified', 4), ('mgmt', 4), ('asst', 4), ('markhtyed', 4), ('encountered', 4), ('li', 4), ('limit', 4), ('numerirtcal', 4), ('solid', 4), ('impacts', 4), ('pur', 4), ('bhqgdoiu', 4), ('suhtnhdyio', 4), ('syxewkji', 4), ('extent', 4), ('repeats', 4), ('raifstow', 4), ('gfeymtql', 4), ('supplier', 4), ('eva', 4), ('parkeyhrt', 4), ('delegate', 4), ('receives', 4), ('lxfnwyuv', 4), ('azm', 4), ('messger', 4), ('erkheim', 4), ('construction', 4), ('posed', 4), ('glog', 4), ('keyhtyvin', 4), ('toriaytun', 4), ('populate', 4), ('vvamrtryot', 4), ('lights', 4), ('bulk', 4), ('pictures', 4), ('reg', 4), ('proxies', 4), ('populating', 4), ('npr', 4), ('speak', 4), ('ron', 4), ('nsoikcyf', 4), ('jhybqael', 4), ('agreed', 4), ('rzucjgvp', 4), ('ioqjgmah', 4), ('jdamieul', 4), ('fandyhgg', 4), ('shesyhur', 4), ('snap', 4), ('chargenverwaltung', 4), ('processor', 4), ('exit', 4), ('config', 4), ('gel', 4), ('scht', 4), ('inter', 4), ('wondering', 4), ('weszfyok', 4), ('fbadnjhu', 4), ('types', 4), ('rmezbnqt', 4), ('ntbmkpuh', 4), ('zxobmreq', 4), ('udikorhv', 4), ('toolcal', 4), ('prtgt', 4), ('beginning', 4), ('vkezwolt', 4), ('fgnqzeai', 4), ('certificates', 4), ('duoyrpvi', 4), ('wgjpviul', 4), ('dated', 4), ('iqcylpok', 4), ('ascpqvni', 4), ('buyer', 4), ('correction', 4), ('monitors', 4), ('heute', 4), ('nyifqpmv', 4), ('kfirxjag', 4), ('transaktion', 4), ('appointment', 4), ('sollte', 4), ('reps', 4), ('failagain', 4), ('highlighted', 4), ('minute', 4), ('territory', 4), ('sufficient', 4), ('ethical', 4), ('sipppr', 4), ('techsupport', 4), ('catalogs', 4), ('ksgytjqr', 4), ('tone', 4), ('detection', 4), ('nord', 4), ('poor', 4), ('ticqvhal', 4), ('vgokzesi', 4), ('serious', 4), ('ytzpxhql', 4), ('ntfxgpms', 4), ('sends', 4), ('neokfwiy', 4), ('ufriscym', 4), ('brhlcpqv', 4), ('sfozwkyx', 4), ('tape', 4), ('stahyru', 4), ('vv', 4), ('bars', 4), ('wrench', 4), ('ahead', 4), ('store', 4), ('risk', 4), ('manjgtiry', 4), ('vnetbgio', 4), ('lqxztben', 4), ('completion', 4), ('reboots', 4), ('wxstfouy', 4), ('isjzcotm', 4), ('images', 4), ('contracts', 4), ('aqihfoly', 4), ('bctypmjw', 4), ('cbhnxafz', 4), ('continues', 4), ('mittwoch', 4), ('embedded', 4), ('providing', 4), ('productivity', 4), ('grund', 4), ('privileges', 4), ('mae', 4), ('hcuixqgj', 4), ('mavxgqbs', 4), ('zkaowfrx', 4), ('rounding', 4), ('tom', 4), ('displays', 4), ('inconsistent', 4), ('zdfymgjp', 4), ('executable', 4), ('xe', 4), ('csrsv', 4), ('refererproxycorrelationurl', 4), ('netacuity', 4), ('vendoreventid', 4), ('tcpflags', 4), ('inlineaction', 4), ('globalproxycorrelationurl', 4), ('vendorversion', 4), ('bridgex', 4), ('element', 4), ('massive', 4), ('proceeding', 4), ('limits', 4), ('miecoszw', 4), ('mhvbnodw', 4), ('edi', 4), ('liuytre', 4), ('xosycftu', 4), ('olhpmsdw', 4), ('cyxieuwk', 4), ('segment', 4), ('direction', 4), ('confirms', 4), ('typed', 4), ('releasing', 4), ('accout', 4), ('nightly', 4), ('phqwmniy', 4), ('kjucgqom', 4), ('environmental', 4), ('confidentiality', 4), ('caution', 4), ('accompanying', 4), ('sole', 4), ('dissemination', 4), ('destrtgoy', 4), ('effort', 4), ('dhermosi', 4), ('defines', 4), ('lu', 4), ('srvr', 4), ('multipart', 4), ('boundary', 4), ('val', 4), ('crosscomp', 4), ('posrt', 4), ('gclnfkis', 4), ('favorites', 4), ('clappdico', 4), ('shathyra', 4), ('uprmwlgb', 4), ('iauqlrjk', 4), ('csscdrill', 4), ('cfac', 4), ('myslidz', 4), ('cont', 4), ('ent', 4), ('dom', 4), ('foreign', 4), ('jxphgfmb', 4), ('rubiargty', 4), ('hnynhsth', 4), ('jsuyhwssad', 4), ('docad', 4), ('stats', 4), ('xsso', 4), ('notwendig', 4), ('wnorzsyv', 4), ('onsite', 4), ('lcosm', 4), ('round', 4), ('kzbuhixt', 4), ('zjdmoahr', 4), ('dscsag', 4), ('zigioachstyac', 4), ('unten', 4), ('eec', 4), ('caac', 4), ('emporarily', 4), ('serv', 4), ('paid', 4), ('research', 4), ('policies', 4), ('naming', 4), ('responsibility', 4), ('easily', 4), ('additionally', 4), ('infotype', 4), ('magento', 4), ('dmz', 4), ('mage', 4), ('adminhtml', 4), ('widget', 4), ('serve', 4), ('attacker', 4), ('pyx', 4), ('kn', 4), ('apost', 4), ('urlencoded', 4), ('lehbyxt', 4), ('fmvcasicd', 4), ('qgqevyvfjbido', 4), ('vyifdirvjfigv', 4), ('dhjhieltie', 4), ('lou', 4), ('vyycaoygzpcnn', 4), ('bmftzwasigbsyxn', 4), ('rva', 4), ('rzw', 4), ('zpcnn', 4), ('xhc', 4), ('ruyw', 4), ('ljywnc', 4), ('nvbw', 4), ('lcmnllmnvbscsj', 4), ('fcm', 4), ('pzcx', 4), ('jkzxisupply', 4), ('hbwunkts', 4), ('tibg', 4), ('yzxbvcnrfc', 4), ('vhcmnox', 4), ('dyawqgb', 4), ('chv', 4), ('evt', 4), ('ja', 4), ('constraints', 4), ('wer', 4), ('managed', 4), ('rohjghit', 4), ('kumghtwar', 4), ('dyrgfwbm', 4), ('netzwerkverbindung', 4), ('vkzwibco', 4), ('pueyvhoi', 4), ('statement', 4), ('road', 4), ('king', 4), ('zload', 4), ('krutnylz', 4), ('convert', 4), ('digital', 4), ('wpkcbtjl', 4), ('produktions', 4), ('wegen', 4), ('oben', 4), ('mti', 4), ('berechtigung', 4), ('nlich', 4), ('haftende', 4), ('gesellschafterin', 4), ('estorage', 4), ('productly', 4), ('carry', 4), ('reopen', 4), ('dringend', 4), ('voraus', 4), ('verf', 4), ('austauschen', 4), ('respective', 4), ('ehfvwltg', 4), ('eakjbtoi', 4), ('erpstartsrv', 4), ('rabhtui', 4), ('zcor', 4), ('starten', 4), ('einloggen', 4), ('cflrqoew', 4), ('qbgjwaye', 4), ('cwdzunxs', 4), ('carts', 4), ('upwonzvd', 4), ('adresse', 4), ('malaysia', 4), ('productive', 4), ('matching', 4), ('unlocking', 4), ('dwujlnhs', 4), ('ecxvrpyl', 4), ('bokrgadu', 4), ('euobrlcn', 4), ('ctu', 4), ('intern', 4), ('paycheck', 4), ('arbeitsplatz', 4), ('lxkecjgr', 4), ('fwknxupq', 4), ('ida', 4), ('startet', 4), ('presentations', 4), ('pptx', 4), ('koenigsee', 4), ('hebrew', 4), ('macro', 4), ('iid', 4), ('efdw', 4), ('sxthcobm', 4), ('taxcizwv', 4), ('quantities', 4), ('tcl', 4), ('steht', 4), ('lsuepvyx', 4), ('lhqwx', 4), ('rqxsm', 4), ('adoption', 4), ('kt', 4), ('anonymizing', 4), ('ahbgjrqz', 4), ('jc', 4), ('jywvemun', 4), ('hgermany', 4), ('nur', 4), ('nrbgctwm', 4), ('kfwdhrmt', 4), ('damen', 4), ('assurance', 4), ('scannen', 4), ('qubywmgf', 4), ('jouickqx', 4), ('umbau', 4), ('erledigung', 4), ('folks', 4), ('msfc', 4), ('soll', 4), ('cutview', 4), ('xmlbfjpg', 4), ('rantlypb', 4), ('edfl', 4), ('edfw', 4), ('gurpthy', 4), ('aoyrspjv', 4), ('hctgfeal', 4), ('anxmvsor', 4), ('fdjoawxq', 4), ('mmaster', 4), ('mnakehrf', 4), ('mvunqihf', 4), ('echo', 4), ('afwzehqs', 4), ('jfbxegac', 4), ('tooldcvcgenratn', 4), ('firefox', 4), ('companyipg', 4), ('partials', 4), ('filename', 4), ('uidgt', 4), ('flat', 4), ('pz', 4), ('olibercsu', 4), ('intrusion', 4), ('prevention', 4), ('eraser', 4), ('definitions', 4), ('revocation', 4), ('sonar', 4), ('signatures', 4), ('lpawx', 4), ('dsthostname', 4), ('snkz', 4), ('locky', 4), ('sbinuxja', 4), ('vtbegcho', 4), ('geffen', 4), ('payload', 4), ('dbff', 4), ('idioma', 4), ('idcx', 4), ('uzojtkmh', 4), ('ltm', 4), ('eylqgodm', 3), ('ybqkwiam', 3), ('erpdata', 3), ('cards', 3), ('kxsceyzo', 3), ('naokumlb', 3), ('guess', 3), ('care', 3), ('acd', 3), ('cdbaoqts', 3), ('wqbsodni', 3), ('abc', 3), ('lkfzibrx', 3), ('ljnabpgx', 3), ('yhmwxsqj', 3), ('ugnthxky', 3), ('campus', 3), ('plmfile', 3), ('omf', 3), ('obrfunctioneventqueue', 3), ('bapi', 3), ('expiry', 3), ('slips', 3), ('roll', 3), ('ring', 3), ('extracted', 3), ('yfqoaepn', 3), ('xnezhsit', 3), ('mms', 3), ('rhwvpmlq', 3), ('zuwhpqrc', 3), ('fbvpcytz', 3), ('nokypgvx', 3), ('pulled', 3), ('benefit', 3), ('tbhkenlo', 3), ('udetjzmn', 3), ('ayueswcm', 3), ('participate', 3), ('lxfwopyq', 3), ('wtqdyoin', 3), ('seek', 3), ('ngt', 3), ('bills', 3), ('awaiting', 3), ('house', 3), ('spare', 3), ('rnafleys', 3), ('fygrwuna', 3), ('gomcekzi', 3), ('laeusvjo', 3), ('fvaihgpx', 3), ('prs', 3), ('lhejbwkc', 3), ('xbmyvnqf', 3), ('kw', 3), ('dqwhpjxy', 3), ('pozjxbey', 3), ('copies', 3), ('reflect', 3), ('viewing', 3), ('appeared', 3), ('beshryulisted', 3), ('shortcut', 3), ('delays', 3), ('mw', 3), ('ordered', 3), ('nmgesubf', 3), ('wugbalmy', 3), ('warden', 3), ('beamer', 3), ('lvxakohq', 3), ('tsfnhowj', 3), ('ware', 3), ('experts', 3), ('sandplant', 3), ('rozsyfai', 3), ('zncajubh', 3), ('ownership', 3), ('prgewfly', 3), ('ndtfvple', 3), ('ucewizyd', 3), ('accidentally', 3), ('fully', 3), ('acceptance', 3), ('vksfrhdx', 3), ('maaryuyten', 3), ('cockpit', 3), ('ksem', 3), ('heading', 3), ('qpixeudn', 3), ('rjlziysd', 3), ('tbvpkjoh', 3), ('wnxzhqoa', 3), ('ringing', 3), ('rebooting', 3), ('rjsulvat', 3), ('uanigkqc', 3), ('hdswinlo', 3), ('ordering', 3), ('wdkyiqfx', 3), ('published', 3), ('daylight', 3), ('jumps', 3), ('qvncizuf', 3), ('ueiybanz', 3), ('gm', 3), ('deactivated', 3), ('hz', 3), ('troubleshoot', 3), ('driving', 3), ('bug', 3), ('male', 3), ('spot', 3), ('mount', 3), ('qbewrpfu', 3), ('lwibmxzo', 3), ('jhyazros', 3), ('azdxonjg', 3), ('ost', 3), ('automation', 3), ('dwsyaqpr', 3), ('bzasnmvw', 3), ('developed', 3), ('escalate', 3), ('vxuikqaf', 3), ('music', 3), ('bwdpmbkp', 3), ('hpqjaory', 3), ('gfrwmije', 3), ('interruption', 3), ('ctc', 3), ('complaints', 3), ('tskvmwag', 3), ('awkrdqzb', 3), ('lose', 3), ('ivohcdpw', 3), ('ixcanwbm', 3), ('searching', 3), ('troubles', 3), ('yiramdntyjqc', 3), ('qtrcepsa', 3), ('onukdesq', 3), ('configuring', 3), ('emsw', 3), ('fgljepar', 3), ('xpsarwiz', 3), ('lndypaqg', 3), ('dhqwtcsr', 3), ('nge', 3), ('faster', 3), ('xosdfhbu', 3), ('gtbfkisl', 3), ('booked', 3), ('merdivan', 3), ('hupnceij', 3), ('hyozjakb', 3), ('sinc', 3), ('salutations', 3), ('eps', 3), ('wgpimkle', 3), ('kentip', 3), ('zheqafyo', 3), ('bqirpxag', 3), ('cowsvzel', 3), ('ryhkefwv', 3), ('fast', 3), ('fence', 3), ('april', 3), ('etnroabk', 3), ('jkvshlfm', 3), ('modem', 3), ('lenxvcbq', 3), ('vwnhjtoi', 3), ('dxwuovgs', 3), ('lmuxizht', 3), ('drafting', 3), ('xvwchsdg', 3), ('pasword', 3), ('deactivate', 3), ('sits', 3), ('qc', 3), ('rckf', 3), ('letters', 3), ('rerouting', 3), ('requirements', 3), ('cubicle', 3), ('corpbusinessdev', 3), ('stand', 3), ('yhmzxcia', 3), ('heszapvl', 3), ('capability', 3), ('leiter', 3), ('money', 3), ('responds', 3), ('uses', 3), ('responses', 3), ('configurator', 3), ('kr', 3), ('arsbtkvd', 3), ('qieagkos', 3), ('mecftgobusa', 3), ('kenci', 3), ('smkpfjzv', 3), ('angefragt', 3), ('titel', 3), ('zugriffs', 3), ('thrown', 3), ('scheduling', 3), ('networking', 3), ('rter', 3), ('stage', 3), ('west', 3), ('dashbankrds', 3), ('polycom', 3), ('fert', 3), ('ids', 3), ('leitung', 3), ('tcb', 3), ('graphics', 3), ('mex', 3), ('juarez', 3), ('fter', 3), ('ear', 3), ('rlhuwmve', 3), ('krcfhoxj', 3), ('stocks', 3), ('fp', 3), ('utiliuytretion', 3), ('limitation', 3), ('kennwort', 3), ('pin', 3), ('ublisodp', 3), ('qydfvpgw', 3), ('corrupted', 3), ('gkwcxzum', 3), ('answkqpe', 3), ('customs', 3), ('pweaver', 3), ('representative', 3), ('runtimeassembly', 3), ('runtimescriptresourcehandler', 3), ('scriptreference', 3), ('datafile', 3), ('instance', 3), ('qfcxbpht', 3), ('oiykfzlr', 3), ('workflows', 3), ('chosen', 3), ('submittal', 3), ('saves', 3), ('row', 3), ('bernardo', 3), ('vpro', 3), ('edksm', 3), ('cursor', 3), ('xaertwdh', 3), ('kcsagvpy', 3), ('distributed', 3), ('enviada', 3), ('para', 3), ('aplications', 3), ('court', 3), ('rtnyumbg', 3), ('yzemkhbq', 3), ('harald', 3), ('nnlein', 3), ('rar', 3), ('hops', 3), ('intelligence', 3), ('vfnraqxc', 3), ('msoffice', 3), ('aspx', 3), ('gkad', 3), ('irqpwgtn', 3), ('dpautgeh', 3), ('synched', 3), ('assisting', 3), ('giuliasana', 3), ('byhdderni', 3), ('aytjedki', 3), ('rucfxpla', 3), ('mat', 3), ('collegues', 3), ('infostand', 3), ('quota', 3), ('loaner', 3), ('ubiqcrvy', 3), ('lync', 3), ('htnvbwxs', 3), ('gwfrzuex', 3), ('toolical', 3), ('xernsfqa', 3), ('mtbu', 3), ('ele', 3), ('ver', 3), ('vxzahrlc', 3), ('frtkpehy', 3), ('carb', 3), ('junk', 3), ('seeking', 3), ('psf', 3), ('mkdfetuq', 3), ('calendars', 3), ('kryuisti', 3), ('blocker', 3), ('tomashtgd', 3), ('mchectg', 3), ('switched', 3), ('attention', 3), ('dshferby', 3), ('houtnzdi', 3), ('exec', 3), ('yxliakph', 3), ('soucfnqe', 3), ('apprentice', 3), ('vc', 3), ('break', 3), ('backed', 3), ('booking', 3), ('sdbcpvtx', 3), ('hzpctsla', 3), ('mulhylen', 3), ('duties', 3), ('fjaqbgnld', 3), ('yukdzwxs', 3), ('docs', 3), ('ramdntyassthywamy', 3), ('heptuizn', 3), ('pethrywr', 3), ('maryhtutina', 3), ('bauuyternfeyt', 3), ('bcxpeuko', 3), ('utorqehx', 3), ('cer', 3), ('gayhtjula', 3), ('znr', 3), ('bakyhrer', 3), ('huhuyghes', 3), ('munxvfhw', 3), ('texsbopi', 3), ('notices', 3), ('danyhuie', 3), ('deyhtwet', 3), ('nbdljruw', 3), ('axcrspyh', 3), ('randomly', 3), ('issuing', 3), ('jwhmqnye', 3), ('xlpvdwre', 3), ('counsel', 3), ('dw', 3), ('xszoedmc', 3), ('migo', 3), ('zkwfqagbs', 3), ('mnxbeuso', 3), ('rfmdlwuo', 3), ('lalthy', 3), ('mobaidfx', 3), ('gviwlsrm', 3), ('ctrl', 3), ('pxbqkfgm', 3), ('qexvrmcn', 3), ('vivthyek', 3), ('byuihand', 3), ('hjcpyxtq', 3), ('okycpbsz', 3), ('luxdnsvk', 3), ('qmnyzcfs', 3), ('hotline', 3), ('heard', 3), ('tuzkadxv', 3), ('smktofel', 3), ('etsoirbw', 3), ('std', 3), ('ucphibmr', 3), ('dfvkbtsj', 3), ('vga', 3), ('bwhrertran', 3), ('alkuozfr', 3), ('psfshytd', 3), ('seocompanyxv', 3), ('numerous', 3), ('doug', 3), ('douglas', 3), ('afe', 3), ('hwrxutgm', 3), ('fjlwmson', 3), ('increasingly', 3), ('prtqi', 3), ('rings', 3), ('gacfhedw', 3), ('iqustfzh', 3), ('gokcerthy', 3), ('dyhtuiel', 3), ('graceuyt', 3), ('bqmjyprz', 3), ('anlegen', 3), ('infos', 3), ('replicate', 3), ('agjzikpf', 3), ('nhfrbxek', 3), ('edksw', 3), ('javascript', 3), ('ase', 3), ('accountant', 3), ('safety', 3), ('dxf', 3), ('unchecked', 3), ('bgflmyar', 3), ('levels', 3), ('iom', 3), ('chassis', 3), ('reads', 3), ('ndige', 3), ('dob', 3), ('abholen', 3), ('thrgxqsuojr', 3), ('xwbesorfs', 3), ('tracert', 3), ('icons', 3), ('vsphere', 3), ('eagcutview', 3), ('luagmhds', 3), ('iymwcelx', 3), ('networks', 3), ('virakv', 3), ('reachable', 3), ('elixsfvu', 3), ('pxwbjofl', 3), ('dsc', 3), ('unreachable', 3), ('outdated', 3), ('kzishqfu', 3), ('bmdawzoi', 3), ('hierarchy', 3), ('pathuick', 3), ('march', 3), ('geoyhurg', 3), ('chriuimjiann', 3), ('wf', 3), ('fuer', 3), ('wrongly', 3), ('vtykrubi', 3), ('whsipqno', 3), ('chukhyt', 3), ('wise', 3), ('terminal', 3), ('rfumsv', 3), ('sls', 3), ('pereira', 3), ('slowing', 3), ('ugrmkdhx', 3), ('jeknosml', 3), ('gkcoltsy', 3), ('geplant', 3), ('hrmann', 3), ('eaodcgsw', 3), ('trmzwbyc', 3), ('xmeytziq', 3), ('wothyehre', 3), ('tables', 3), ('understanding', 3), ('symbols', 3), ('clarification', 3), ('stream', 3), ('schlumhdyhter', 3), ('reconciliation', 3), ('hotel', 3), ('ebus', 3), ('bunch', 3), ('richtige', 3), ('messed', 3), ('utoegyqx', 3), ('lhosidqg', 3), ('cash', 3), ('profiles', 3), ('telling', 3), ('xhaomnjl', 3), ('ctusaqpr', 3), ('printouts', 3), ('spool', 3), ('fueiklyv', 3), ('jargqpkm', 3), ('dierppears', 3), ('uwofavej', 3), ('hxyatnjc', 3), ('columns', 3), ('erscheint', 3), ('ohne', 3), ('dass', 3), ('ffnet', 3), ('cobsfvjz', 3), ('bekommen', 3), ('scorecard', 3), ('iscl', 3), ('erathyur', 3), ('smartphone', 3), ('timecards', 3), ('hbmwlprq', 3), ('ilfvyodx', 3), ('vmware', 3), ('druckauftr', 3), ('druck', 3), ('saerpwno', 3), ('qsdfmakc', 3), ('reflected', 3), ('originally', 3), ('enquiries', 3), ('fjohugzb', 3), ('fhagjskd', 3), ('stated', 3), ('estimated', 3), ('soujqrxw', 3), ('mvwduljx', 3), ('workbook', 3), ('minitab', 3), ('qwsjptlo', 3), ('hnlasbed', 3), ('eqtofwbm', 3), ('mojfbwds', 3), ('krisyuhnyrt', 3), ('strange', 3), ('communicate', 3), ('massage', 3), ('informations', 3), ('oprbatch', 3), ('kvrmnuix', 3), ('consigned', 3), ('causes', 3), ('tommyth', 3), ('duyhurmont', 3), ('kehtxprg', 3), ('uekapfzt', 3), ('bigger', 3), ('vybmcrxo', 3), ('kirxdspz', 3), ('excellence', 3), ('invite', 3), ('browsing', 3), ('mwst', 3), ('dardabthyr', 3), ('copier', 3), ('ihfkwzjd', 3), ('erbxoyqk', 3), ('vhlepcta', 3), ('lqbgcxpt', 3), ('liu', 3), ('settle', 3), ('companycenter', 3), ('scans', 3), ('databases', 3), ('upgtrvnj', 3), ('koximpja', 3), ('alrthyu', 3), ('toolonic', 3), ('feed', 3), ('kvp', 3), ('panghyiraj', 3), ('shthuihog', 3), ('acqpinyd', 3), ('ecygimqd', 3), ('beschichtungsleitstand', 3), ('abort', 3), ('jkpwisnf', 3), ('lgpdyncm', 3), ('handheld', 3), ('occurring', 3), ('punch', 3), ('aiqjxhuv', 3), ('smhdyhti', 3), ('haunm', 3), ('stdezpqw', 3), ('converted', 3), ('ausgef', 3), ('schtrtgoyht', 3), ('basic', 3), ('yzbjhmpw', 3), ('vzrulkog', 3), ('headquarters', 3), ('applying', 3), ('bare', 3), ('xref', 3), ('foreseedstipgeo', 3), ('srcmacaddress', 3), ('reserved', 3), ('reassigned', 3), ('setzen', 3), ('ceo', 3), ('indexed', 3), ('atp', 3), ('safrgyynjit', 3), ('cutting', 3), ('delivering', 3), ('claim', 3), ('settlement', 3), ('reject', 3), ('traveling', 3), ('credits', 3), ('sgnubadl', 3), ('gpkovbah', 3), ('joiners', 3), ('zno', 3), ('christgrytoph', 3), ('sbvlxuwm', 3), ('yanbikrx', 3), ('prir', 3), ('pause', 3), ('partially', 3), ('miiadmin', 3), ('weird', 3), ('sense', 3), ('converting', 3), ('tfesaxip', 3), ('cvorpnth', 3), ('suggestions', 3), ('pair', 3), ('wtubpdsz', 3), ('traversecity', 3), ('proygkjt', 3), ('mwetuhqf', 3), ('amrthruta', 3), ('kadgdyam', 3), ('shopping', 3), ('saver', 3), ('imaged', 3), ('defender', 3), ('village', 3), ('prices', 3), ('kbyivdfz', 3), ('zwutmehy', 3), ('admins', 3), ('rsa', 3), ('jean', 3), ('topic', 3), ('yb', 3), ('vw', 3), ('bcl', 3), ('ruleid', 3), ('lang', 3), ('crosstenant', 3), ('joftgost', 3), ('furnace', 3), ('whoyizfp', 3), ('arkulcoi', 3), ('plugged', 3), ('olympus', 3), ('sebfghkasthian', 3), ('anniversary', 3), ('tempuser', 3), ('trainee', 3), ('dropbox', 3), ('nijdaukz', 3), ('ghkkytu', 3), ('completing', 3), ('bbf', 3), ('tempo', 3), ('rarily', 3), ('tml', 3), ('cook', 3), ('doma', 3), ('ekxl', 3), ('ser', 3), ('banking', 3), ('detect', 3), ('wkgpcxqd', 3), ('vobarhzk', 3), ('promotion', 3), ('erzeugen', 3), ('rebuild', 3), ('bcd', 3), ('slowness', 3), ('calculate', 3), ('vanghtydec', 3), ('accesses', 3), ('closet', 3), ('bxspwfyo', 3), ('vzystwor', 3), ('usx', 3), ('vcyktjxp', 3), ('uxdojvrq', 3), ('becoming', 3), ('tengigabitethernet', 3), ('xp', 3), ('kcompany', 3), ('relations', 3), ('ytd', 3), ('stopping', 3), ('programdntya', 3), ('konferenzraum', 3), ('xb', 3), ('bals', 3), ('organisation', 3), ('combination', 3), ('uisewznr', 3), ('ewtmkphs', 3), ('lalanne', 3), ('akku', 3), ('downloads', 3), ('estate', 3), ('sethdyr', 3), ('hdtyr', 3), ('rxqtvanc', 3), ('kthqwxvb', 3), ('restoration', 3), ('easy', 3), ('excise', 3), ('wiederherstellen', 3), ('raus', 3), ('pthsqroz', 3), ('moedyanvess', 3), ('mehrere', 3), ('brauche', 3), ('vaigycet', 3), ('charger', 3), ('koithc', 3), ('noon', 3), ('packages', 3), ('reconnect', 3), ('ngvwoukp', 3), ('practice', 3), ('mdbegvct', 3), ('dbvichlg', 3), ('permit', 3), ('branch', 3), ('duty', 3), ('sinkholes', 3), ('false', 3), ('enforcement', 3), ('mitigation', 3), ('examined', 3), ('characteristics', 3), ('curious', 3), ('indication', 3), ('hierarchical', 3), ('associating', 3), ('participating', 3), ('entities', 3), ('translating', 3), ('locating', 3), ('designating', 3), ('authoritative', 3), ('subdomains', 3), ('specifies', 3), ('structures', 3), ('exchanges', 3), ('hole', 3), ('routable', 3), ('denies', 3), ('benign', 3), ('ramdntyifications', 3), ('leakage', 3), ('identifiable', 3), ('reputational', 3), ('feeding', 3), ('victims', 3), ('beshryulists', 3), ('impede', 3), ('trojans', 3), ('hostnames', 3), ('leading', 3), ('fea', 3), ('units', 3), ('simple', 3), ('netzwerk', 3), ('howfanzi', 3), ('smart', 3), ('individual', 3), ('repaired', 3), ('mapped', 3), ('toner', 3), ('roshyario', 3), ('mcfaullfhry', 3), ('ssofgrtymerset', 3), ('josefghph', 3), ('hughdthes', 3), ('jofgyst', 3), ('langytge', 3), ('mic', 3), ('gthydanp', 3), ('kflqpite', 3), ('gbeoqsnc', 3), ('brand', 3), ('injection', 3), ('discover', 3), ('grid', 3), ('auditing', 3), ('administrators', 3), ('commands', 3), ('answering', 3), ('freischaltung', 3), ('wichtig', 3), ('unterst', 3), ('wimozrdc', 3), ('iodvtuaz', 3), ('raghyvhdra', 3), ('najuty', 3), ('fchijage', 3), ('editor', 3), ('como', 3), ('ahlqgjwx', 3), ('wbsfavhg', 3), ('clocks', 3), ('kpobysnc', 3), ('wrelsfqa', 3), ('qfwosjkh', 3), ('nieghjyukea', 3), ('italy', 3), ('reisekosten', 3), ('cbupnjzo', 3), ('daflthkw', 3), ('accessed', 3), ('nhsogrwy', 3), ('qkxhbnvp', 3), ('aktuell', 3), ('lixwgnto', 3), ('gf', 3), ('vgsqetbx', 3), ('falsch', 3), ('jofghyuach', 3), ('wurden', 3), ('regard', 3), ('bqapjkcl', 3), ('ljeakcqf', 3), ('charm', 3), ('vbda', 3), ('stanfghyley', 3), ('guhtykes', 3), ('fmp', 3), ('hkrecpfv', 3), ('kgwpbexv', 3), ('reimage', 3), ('palghjmal', 3), ('dec', 3), ('eingegeben', 3), ('tonight', 3), ('mandatory', 3), ('pack', 3), ('ghjkzalez', 3), ('travelling', 3), ('gbar', 3), ('bearbeiten', 3), ('schreib', 3), ('abstechprogramdntym', 3), ('gqwdslpc', 3), ('clhgpqnb', 3), ('february', 3), ('vfjsubao', 3), ('helfen', 3), ('ujtmipzv', 3), ('zedlet', 3), ('tfazwrdv', 3), ('gasbfqvp', 3), ('fmvqgjih', 3), ('keinen', 3), ('zugang', 3), ('homburg', 3), ('reconditioning', 3), ('junior', 3), ('occur', 3), ('calculated', 3), ('contractor', 3), ('batches', 3), ('vm', 3), ('findet', 3), ('mcphgvnb', 3), ('bdegqtyj', 3), ('investigated', 3), ('reparo', 3), ('drills', 3), ('initiated', 3), ('xzupryaf', 3), ('vlbikhsm', 3), ('depending', 3), ('fact', 3), ('eligibility', 3), ('nter', 3), ('avmeocnk', 3), ('mvycfwka', 3), ('wqzarvhx', 3), ('hfsojckw', 3), ('utthku', 3), ('tehrsytu', 3), ('tryhutehdtui', 3), ('forbidden', 3), ('reversed', 3), ('modific', 3), ('verl', 3), ('drum', 3), ('bertragen', 3), ('ausgetauscht', 3), ('bur', 3), ('orde', 3), ('kl', 3), ('ganz', 3), ('reinhard', 3), ('warnings', 3), ('alex', 3), ('krcscfpr', 3), ('lmxl', 3), ('simfghon', 3), ('imported', 3), ('keypad', 3), ('espinosa', 3), ('unser', 3), ('drucken', 3), ('scheint', 3), ('auftrag', 3), ('netz', 3), ('nichts', 3), ('apologize', 3), ('aghynil', 3), ('eonhuwlg', 3), ('wydorpzi', 3), ('fjzywdpg', 3), ('vkdobexr', 3), ('qcjevayr', 3), ('xirwgjks', 3), ('uyocgasl', 3), ('ogabwxzv', 3), ('nfdtriwx', 3), ('lriupqct', 3), ('fall', 3), ('telekom', 3), ('tastatur', 3), ('formatheywting', 3), ('variant', 3), ('trailing', 3), ('ulroqsyf', 3), ('wctpnarb', 3), ('khrtyujuine', 3), ('orange', 3), ('rqxl', 3), ('ik', 3), ('qdtywmkv', 3), ('aolijwnx', 3), ('instandsetzung', 3), ('ghaltiek', 3), ('triggered', 3), ('cube', 3), ('rebate', 3), ('presence', 3), ('vpns', 3), ('registry', 3), ('dnc', 3), ('zjawqgcs', 3), ('tohqcxla', 3), ('specifications', 3), ('erstellen', 3), ('cugjzqlf', 3), ('djwbyact', 3), ('danghtnuell', 3), ('rdp', 3), ('efficiency', 3), ('places', 3), ('homepage', 3), ('matghyuthdw', 3), ('vk', 3), ('santrtos', 3), ('persons', 3), ('adjusted', 3), ('xaqzisrk', 3), ('juli', 3), ('nabjpdhy', 3), ('bjuqwidt', 3), ('payments', 3), ('strahlraum', 3), ('ancile', 3), ('ndert', 3), ('owhuxbnf', 3), ('sxbgyrou', 3), ('phr', 3), ('uhr', 3), ('gesellschaft', 3), ('tivbxojn', 3), ('gorlajmp', 3), ('uperform', 3), ('flickering', 3), ('purple', 3), ('treat', 3), ('benz', 3), ('caas', 3), ('freigegeben', 3), ('abgeschlossen', 3), ('zhrgtangs', 3), ('attend', 3), ('breqgycv', 3), ('pogredrty', 3), ('reconnaissance', 3), ('discovery', 3), ('patch', 3), ('fertigung', 3), ('hartbearbeitung', 3), ('ewkw', 3), ('selbstst', 3), ('hortl', 3), ('globalview', 3), ('tru', 3), ('efjzbtcm', 3), ('mdpviqbf', 3), ('jusfrttin', 3), ('gtehdnyuerrf', 3), ('debgrtybie', 3), ('savgrtyuille', 3), ('hsbc', 3), ('svc', 3), ('sylvthryia', 3), ('ihre', 3), ('exel', 3), ('domestic', 3), ('international', 3), ('uwncfovt', 3), ('vxjbunfi', 3), ('otd', 3), ('wznkpjis', 3), ('suhrhtyju', 3), ('migration', 3), ('pjxclyhs', 3), ('fcniljtu', 3), ('funcionando', 3), ('eqwaiphc', 3), ('qxwfeuth', 3), ('yfmaqovp', 3), ('wdonhbez', 3), ('bildschirm', 3), ('vmax', 3), ('threats', 3), ('filesystem', 3), ('mswineventlog', 3), ('allocated', 3), ('article', 3), ('dynamic', 3), ('recsynqt', 3), ('byoezmla', 3), ('halle', 3), ('meinem', 3), ('geb', 3), ('infoblox', 3), ('mss', 3), ('andrdgrtew', 3), ('damaged', 3), ('vipqmdse', 3), ('omrsiqdv', 3), ('iwhcpgoa', 3), ('swmiynoz', 3), ('deflate', 3), ('disposition', 3), ('plain', 3), ('isset', 3), ('preg', 3), ('typ', 3), ('vetkdblx', 3), ('nsuwkraj', 3), ('daghyunny', 3), ('visiting', 3), ('dead', 3), ('pichayapuk', 3), ('blagtnco', 3), ('prishry', 3), ('kpnzvsuw', 3), ('lwmqyjbv', 3), ('initiate', 3), ('ftgvlneh', 3), ('aitsgqwo', 3), ('santiago', 3), ('zcnc', 3), ('lpaw', 3), ('random', 3), ('clean', 3), ('reimaged', 3), ('fca', 3), ('jenfrgryui', 3), ('street', 3), ('docx', 3), ('division', 3), ('utislgov', 3), ('fetaqndw', 3), ('sadiertpta', 3), ('palffs', 3), ('rohitdrf', 3), ('rak', 3), ('gncpezhx', 3), ('hopqcvza', 3), ('wilsfgtjl', 3), ('mohnrysu', 3), ('mijhmiles', 3), ('somebody', 2), ('ins', 2), ('ugephfta', 2), ('hrbqkvij', 2), ('dceoufyz', 2), ('saufqkmd', 2), ('coded', 2), ('dierppearing', 2), ('yisohglr', 2), ('uvteflgb', 2), ('didnt', 2), ('merktc', 2), ('initiatives', 2), ('dlougnqw', 2), ('jiuybxew', 2), ('rewards', 2), ('przndfbo', 2), ('pldqbhtn', 2), ('fbmugzrl', 2), ('ahyiuqev', 2), ('bas', 2), ('checkoutview', 2), ('bop', 2), ('authorize', 2), ('noscwdpm', 2), ('akiowsmp', 2), ('krlszbqo', 2), ('spimolgz', 2), ('callback', 2), ('administrador', 2), ('hss', 2), ('finishing', 2), ('dcvphjru', 2), ('ybomrjst', 2), ('gv', 2), ('nxhwyepl', 2), ('mudstbxo', 2), ('continuous', 2), ('locally', 2), ('uijxpazn', 2), ('gvtzlphs', 2), ('incredibly', 2), ('mob', 2), ('nip', 2), ('windy', 2), ('scy', 2), ('wallpaper', 2), ('qiwhfkdv', 2), ('ltaballotcsalesemp', 2), ('repyzajo', 2), ('avurmegj', 2), ('pxgmjynu', 2), ('clearing', 2), ('umpteenth', 2), ('presents', 2), ('rectified', 2), ('dhcopwxa', 2), ('dll', 2), ('paying', 2), ('basically', 2), ('nearing', 2), ('expiration', 2), ('supervisors', 2), ('gstdy', 2), ('xyculgav', 2), ('cuqptoah', 2), ('itry', 2), ('taxes', 2), ('publish', 2), ('hgygrtui', 2), ('presently', 2), ('zanivrec', 2), ('capbfhur', 2), ('hprdlbxf', 2), ('nozjtgwi', 2), ('flickers', 2), ('delayed', 2), ('pushed', 2), ('zsdr', 2), ('sgblhypi', 2), ('htqmidsn', 2), ('tnlshpwb', 2), ('abl', 2), ('eto', 2), ('iksqbuxf', 2), ('muzxgwvk', 2), ('commercial', 2), ('severe', 2), ('everyday', 2), ('nogo', 2), ('trunk', 2), ('govt', 2), ('aiuw', 2), ('rtnzvplq', 2), ('erhmuncq', 2), ('edits', 2), ('invitations', 2), ('amet', 2), ('winows', 2), ('schulung', 2), ('reopened', 2), ('landing', 2), ('wdpzfqgi', 2), ('zndgqcux', 2), ('pofgtzdravem', 2), ('faq', 2), ('cnn', 2), ('sandir', 2), ('gvcfhwjy', 2), ('lyxcorqb', 2), ('vewdsifl', 2), ('zjdmftkv', 2), ('sthry', 2), ('platformonline', 2), ('mqjdyizg', 2), ('amhywoqg', 2), ('retention', 2), ('oncidblt', 2), ('knlrgsiv', 2), ('cqvuexjz', 2), ('xoukpfvr', 2), ('oxvakgcl', 2), ('jabra', 2), ('uvbmysgcbenezer', 2), ('thrydad', 2), ('tcflirwg', 2), ('ojflyruq', 2), ('thryd', 2), ('skpe', 2), ('delivers', 2), ('uat', 2), ('kit', 2), ('hints', 2), ('lnbdm', 2), ('philadelph', 2), ('xtsuifdz', 2), ('wktgzcyl', 2), ('dbwkxalj', 2), ('ewourgcx', 2), ('bngell', 2), ('cgdaytshqsd', 2), ('onmicrosoft', 2), ('rao', 2), ('pacific', 2), ('badge', 2), ('vvsardkajdjtf', 2), ('oinqckds', 2), ('qieswrfu', 2), ('oktober', 2), ('jchlkard', 2), ('tcaiyjfg', 2), ('zqt', 2), ('xabkyoug', 2), ('disruption', 2), ('savings', 2), ('cracked', 2), ('dctvfjrn', 2), ('aolhgbps', 2), ('pbxqtcek', 2), ('vriendelijke', 2), ('groet', 2), ('directeur', 2), ('mean', 2), ('infosthryda', 2), ('involve', 2), ('gqhfieys', 2), ('tqnbkjgu', 2), ('aese', 2), ('fdqjsygx', 2), ('aivdjqtr', 2), ('targets', 2), ('kmscan', 2), ('awywx', 2), ('salesforce', 2), ('reportncqulao', 2), ('qauighdpnager', 2), ('nzuofeam', 2), ('exszgtwd', 2), ('workplace', 2), ('wei', 2), ('edm', 2), ('hanging', 2), ('ici', 2), ('queues', 2), ('zlz', 2), ('trail', 2), ('hjsastadad', 2), ('kjddwdd', 2), ('routine', 2), ('oee', 2), ('sirs', 2), ('hdd', 2), ('karashsnnsb', 2), ('vjzfocgt', 2), ('beeping', 2), ('stall', 2), ('reviewing', 2), ('alejayhsdtffndro', 2), ('chose', 2), ('commodity', 2), ('obsolete', 2), ('staying', 2), ('tjwdhwdw', 2), ('tdlwdkunis', 2), ('dialing', 2), ('schuette', 2), ('experienced', 2), ('vtwxaefm', 2), ('ljisafue', 2), ('hxgayczeraum', 2), ('rfcserver', 2), ('contatc', 2), ('tnks', 2), ('vorg', 2), ('trigger', 2), ('loads', 2), ('recovery', 2), ('biaprod', 2), ('ngenmessmaschine', 2), ('iboltufk', 2), ('ezfnvcqp', 2), ('gogtyekhan', 2), ('disconnections', 2), ('lasplant', 2), ('fulfill', 2), ('depend', 2), ('worse', 2), ('chek', 2), ('aqrzskpg', 2), ('naruedlk', 2), ('mpvhakdq', 2), ('castings', 2), ('corrupt', 2), ('idg', 2), ('hfyujqti', 2), ('lpawty', 2), ('yks', 2), ('calibration', 2), ('srvlavstorage', 2), ('itar', 2), ('wdwddw', 2), ('pdv', 2), ('arbeiten', 2), ('abarbeiten', 2), ('pladjmxt', 2), ('schung', 2), ('delegated', 2), ('verifying', 2), ('diagnosis', 2), ('variants', 2), ('eventually', 2), ('replacing', 2), ('ecxwnmqi', 2), ('pztiqjuh', 2), ('usalikfj', 2), ('lfmpxbcn', 2), ('term', 2), ('sg', 2), ('bankverbindung', 2), ('csewdwdwdndmill', 2), ('configig', 2), ('ucdwyxko', 2), ('apktrsyq', 2), ('vciknubg', 2), ('wdlkabms', 2), ('shbgwxeparamdnty', 2), ('din', 2), ('kmvwxdti', 2), ('uaoyhcep', 2), ('nice', 2), ('phjencfg', 2), ('kwtcyazx', 2), ('vnhaycfo', 2), ('rgpvdhcm', 2), ('mgvpabsj', 2), ('ks', 2), ('wired', 2), ('affects', 2), ('ckzusetzen', 2), ('hswddwk', 2), ('temps', 2), ('interns', 2), ('costly', 2), ('optimization', 2), ('unread', 2), ('einstellen', 2), ('herrn', 2), ('identification', 2), ('tdkfuobm', 2), ('qrtmaxos', 2), ('buy', 2), ('lrrw', 2), ('syncing', 2), ('coordinator', 2), ('retrieving', 2), ('globalace', 2), ('waste', 2), ('helper', 2), ('remind', 2), ('agnwfwieszka', 2), ('camera', 2), ('hdmi', 2), ('demonstrations', 2), ('dongle', 2), ('disconnects', 2), ('banned', 2), ('hyperlink', 2), ('believed', 2), ('plugs', 2), ('french', 2), ('concall', 2), ('aswl', 2), ('kvetadzo', 2), ('preparing', 2), ('pvjdtrya', 2), ('oevyhtdx', 2), ('lsgthhuart', 2), ('suspected', 2), ('outofmemoryexception', 2), ('reflection', 2), ('getrawbytes', 2), ('boolean', 2), ('iscriptresourcehandler', 2), ('getscriptresourceurl', 2), ('string', 2), ('eventargs', 2), ('alabama', 2), ('ltabthrysallotcsalesman', 2), ('bthrob', 2), ('xziwkgeo', 2), ('gdiraveu', 2), ('appreciatehub', 2), ('recognition', 2), ('rev', 2), ('uninstalling', 2), ('technologies', 2), ('venue', 2), ('problematic', 2), ('letter', 2), ('kunden', 2), ('wlsazrce', 2), ('uwehsqbk', 2), ('oziflwma', 2), ('nhgvmqdl', 2), ('xpugntjv', 2), ('zcaermdt', 2), ('brought', 2), ('miowvyrs', 2), ('ter', 2), ('gustathsvo', 2), ('assunto', 2), ('jesjnlyenmrest', 2), ('vogtfyne', 2), ('isugmpcn', 2), ('lnpgjhus', 2), ('beteiligungs', 2), ('regster', 2), ('yopvwrjq', 2), ('techn', 2), ('planck', 2), ('detectors', 2), ('disconnect', 2), ('lpapr', 2), ('cthaasnoc', 2), ('gic', 2), ('trghwyng', 2), ('herself', 2), ('ray', 2), ('hqap', 2), ('rjodlbcf', 2), ('uorcpftk', 2), ('rqeuest', 2), ('luck', 2), ('improperly', 2), ('schyhty', 2), ('cothyshy', 2), ('icloud', 2), ('iam', 2), ('backups', 2), ('exceeded', 2), ('mfizgpoy', 2), ('akbvznci', 2), ('gruss', 2), ('gpc', 2), ('cross', 2), ('spdczoth', 2), ('cooling', 2), ('stick', 2), ('webi', 2), ('stdiondwd', 2), ('rawdwu', 2), ('autobank', 2), ('hanx', 2), ('offer', 2), ('venktyamk', 2), ('transmission', 2), ('analyze', 2), ('ckmeldungen', 2), ('express', 2), ('cubdsrml', 2), ('znewqgop', 2), ('mdw', 2), ('sxnzacoj', 2), ('lwvqgfby', 2), ('aswyuysm', 2), ('mtbelengineering', 2), ('reconfigure', 2), ('matlxjgi', 2), ('elrndiuy', 2), ('meet', 2), ('opeyctrhbkm', 2), ('nearby', 2), ('bwsdslspln', 2), ('fugwxdqh', 2), ('sekarf', 2), ('accidently', 2), ('turleythy', 2), ('hire', 2), ('combinations', 2), ('explore', 2), ('uicjxvng', 2), ('yjscozva', 2), ('lyjoeacv', 2), ('nk', 2), ('defect', 2), ('instant', 2), ('hell', 2), ('carriers', 2), ('rnueobcz', 2), ('lwhcbati', 2), ('vlc', 2), ('collect', 2), ('rayhtukumujar', 2), ('await', 2), ('vomtbcej', 2), ('lyiwqrct', 2), ('written', 2), ('freight', 2), ('shyheehew', 2), ('dtheb', 2), ('lokiwfhg', 2), ('udkoqrcg', 2), ('assuming', 2), ('uypsqcbm', 2), ('fqpybgri', 2), ('aerospace', 2), ('wqxzleky', 2), ('pubreports', 2), ('benelthyux', 2), ('crackling', 2), ('essa', 2), ('workgroups', 2), ('gogtyekthyto', 2), ('quarter', 2), ('learning', 2), ('sh', 2), ('env', 2), ('rps', 2), ('youfzmgp', 2), ('xvysrnmb', 2), ('trainer', 2), ('ended', 2), ('aorthyme', 2), ('rnsuipbk', 2), ('ta', 2), ('carried', 2), ('distance', 2), ('nxjvzcta', 2), ('apt', 2), ('challan', 2), ('becomes', 2), ('defaulting', 2), ('defaults', 2), ('gmnhjfbw', 2), ('farnwhji', 2), ('schoegdythu', 2), ('clocking', 2), ('ultramdntyet', 2), ('wrongful', 2), ('vflagort', 2), ('xyotrhlf', 2), ('shouldn', 2), ('highlight', 2), ('property', 2), ('unzip', 2), ('chief', 2), ('fpbmtxei', 2), ('jtqbcnfs', 2), ('solver', 2), ('ockthiyj', 2), ('ypladjeu', 2), ('wzfryxav', 2), ('guruythupyhtyad', 2), ('servicing', 2), ('jayatramdntydba', 2), ('cvyg', 2), ('amihtar', 2), ('vvkuthyrppg', 2), ('kdeqjncw', 2), ('zdgxtfqs', 2), ('tibmhxcs', 2), ('gabryltke', 2), ('plc', 2), ('yhhm', 2), ('direkt', 2), ('school', 2), ('answered', 2), ('govind', 2), ('tinyurl', 2), ('rxloutpn', 2), ('purrqs', 2), ('fmhlugqk', 2), ('dpraethi', 2), ('amy', 2), ('crysyhtal', 2), ('xithya', 2), ('xamtgvnw', 2), ('usdekfzq', 2), ('manufactured', 2), ('revision', 2), ('handset', 2), ('eglavnhx', 2), ('uprodleq', 2), ('rangini', 2), ('deal', 2), ('hiyhtull', 2), ('malfunction', 2), ('phish', 2), ('zrvbahym', 2), ('mailserver', 2), ('renewed', 2), ('aisl', 2), ('rob', 2), ('offinance', 2), ('othybin', 2), ('przcxbml', 2), ('hip', 2), ('ylfwnbkr', 2), ('pfad', 2), ('traiyctrhbkm', 2), ('plvnuxmrterial', 2), ('breakdown', 2), ('paternoster', 2), ('loging', 2), ('fbl', 2), ('spoke', 2), ('loosing', 2), ('reduced', 2), ('tvecikxn', 2), ('reinstalling', 2), ('esqcuwbg', 2), ('xgufkidq', 2), ('lzapwbnc', 2), ('effecting', 2), ('fylrosuk', 2), ('kedgmiul', 2), ('photos', 2), ('progress', 2), ('qjtbrvfy', 2), ('avwqmhsp', 2), ('thoyhts', 2), ('brthyrtiv', 2), ('cksetzung', 2), ('rack', 2), ('datas', 2), ('ink', 2), ('rugdyxqh', 2), ('aqvocmuy', 2), ('invalidated', 2), ('rhaycqjg', 2), ('fdyietau', 2), ('dvsyxwbu', 2), ('zhhtyangq', 2), ('yjwivxsh', 2), ('fcetobrj', 2), ('iewnguxv', 2), ('bufwxeiy', 2), ('stgyott', 2), ('gdhdyrham', 2), ('flap', 2), ('boithdfa', 2), ('tojwnydh', 2), ('designated', 2), ('wc', 2), ('ongoing', 2), ('conducted', 2), ('nabjwvtd', 2), ('sprhouiv', 2), ('segvwfyn', 2), ('mogtrevn', 2), ('depute', 2), ('yuxloigj', 2), ('tzfwjxhe', 2), ('thsyrley', 2), ('helpful', 2), ('sanchrtyn', 2), ('mileage', 2), ('presentation', 2), ('presenter', 2), ('compared', 2), ('genuine', 2), ('anleitung', 2), ('webpages', 2), ('presenting', 2), ('presented', 2), ('selections', 2), ('hnyeajrw', 2), ('purch', 2), ('characters', 2), ('decided', 2), ('negative', 2), ('productions', 2), ('mailing', 2), ('uagqromi', 2), ('sqgtkmci', 2), ('umdyvbxo', 2), ('qwzstijr', 2), ('achthyardk', 2), ('troxyekl', 2), ('lzdvgwut', 2), ('endg', 2), ('ltige', 2), ('anderen', 2), ('wechseln', 2), ('jadqhguy', 2), ('euro', 2), ('dcgwuvfk', 2), ('pressure', 2), ('punched', 2), ('better', 2), ('proof', 2), ('gzqijaoc', 2), ('rfywvloa', 2), ('investor', 2), ('relation', 2), ('wchidyuk', 2), ('shqbfpuy', 2), ('listener', 2), ('specially', 2), ('jashtyckie', 2), ('jacyjddwline', 2), ('yotywdsef', 2), ('justification', 2), ('periods', 2), ('upper', 2), ('lookup', 2), ('inco', 2), ('town', 2), ('bqjvxsaf', 2), ('aupdonjy', 2), ('rus', 2), ('mcoswhjuanthila', 2), ('correspondence', 2), ('refused', 2), ('khadfhty', 2), ('facilitator', 2), ('juhu', 2), ('jojfufn', 2), ('astmvqhc', 2), ('ghyaniel', 2), ('dealgce', 2), ('wyxqkzmf', 2), ('urigtqnp', 2), ('coffee', 2), ('popping', 2), ('hrss', 2), ('hrssc', 2), ('accident', 2), ('dunnings', 2), ('acquiring', 2), ('bluescreen', 2), ('pallutyr', 2), ('xceliron', 2), ('designed', 2), ('timely', 2), ('frydqbgs', 2), ('ugmnzfik', 2), ('sinic', 2), ('ebhl', 2), ('vorgesetzten', 2), ('extensions', 2), ('thinks', 2), ('kollegen', 2), ('dazu', 2), ('kommen', 2), ('ckt', 2), ('lanigpkq', 2), ('qzhakunx', 2), ('paths', 2), ('cvihupnk', 2), ('dbrugslc', 2), ('ydwtsunh', 2), ('njdxwpvg', 2), ('onlehdsi', 2), ('wkzrenmj', 2), ('vsyctbzk', 2), ('zvjxuahe', 2), ('annoying', 2), ('tyhufrey', 2), ('thyel', 2), ('eilt', 2), ('gestartet', 2), ('erfolg', 2), ('vfoyenlw', 2), ('ntpbdeyf', 2), ('courage', 2), ('qycgdfhz', 2), ('iqshzdru', 2), ('mreocsnk', 2), ('swoyxzma', 2), ('pivot', 2), ('amended', 2), ('likes', 2), ('moments', 2), ('suppliers', 2), ('wish', 2), ('ojdukgzc', 2), ('ftmill', 2), ('uvorgwts', 2), ('mlqzaicb', 2), ('ldgl', 2), ('characteristic', 2), ('stamp', 2), ('timecard', 2), ('identifies', 2), ('knemilvx', 2), ('dvqtziya', 2), ('zj', 2), ('shivakuhdty', 2), ('vsiemxgh', 2), ('lgeciroy', 2), ('intransit', 2), ('duplicated', 2), ('japan', 2), ('lbxugpjw', 2), ('cnmfbdui', 2), ('belong', 2), ('lehsm', 2), ('vxfkwaqh', 2), ('observation', 2), ('licenses', 2), ('horeduca', 2), ('ogrhivnm', 2), ('breach', 2), ('cpinsety', 2), ('muapxkns', 2), ('wpdxlbhz', 2), ('dymanics', 2), ('supporting', 2), ('ejecting', 2), ('tapes', 2), ('sheets', 2), ('figured', 2), ('buffer', 2), ('putting', 2), ('mhtyike', 2), ('szumyhtulas', 2), ('greyed', 2), ('tnxiuramdnty', 2), ('ops', 2), ('continually', 2), ('stp', 2), ('inboxes', 2), ('cutter', 2), ('kawtidthry', 2), ('latency', 2), ('qualifying', 2), ('prarthyr', 2), ('mvfnbces', 2), ('urbckxna', 2), ('ftsqkvre', 2), ('bqzrupic', 2), ('nthryitin', 2), ('gedruckt', 2), ('freitag', 2), ('graurkart', 2), ('eeml', 2), ('train', 2), ('russ', 2), ('eakhgxbw', 2), ('pfyadjmb', 2), ('drill', 2), ('stefytyn', 2), ('truview', 2), ('oydlehun', 2), ('svnfrxdk', 2), ('cann', 2), ('kkc', 2), ('bench', 2), ('inserting', 2), ('utilize', 2), ('cesgrtar', 2), ('abgrtyreu', 2), ('proxy', 2), ('lower', 2), ('errror', 2), ('chrthryui', 2), ('stavenheim', 2), ('explain', 2), ('fox', 2), ('chatgrylouy', 2), ('fdmobjul', 2), ('oicarvqt', 2), ('hatryupsfshytd', 2), ('transfered', 2), ('packing', 2), ('fhurakgsl', 2), ('mldufqov', 2), ('vuxdrbng', 2), ('owqplduj', 2), ('asheshopsw', 2), ('prdord', 2), ('salesman', 2), ('uninstall', 2), ('tsbnfixp', 2), ('numwqahj', 2), ('paragraph', 2), ('mar', 2), ('advantage', 2), ('temodell', 2), ('tetyp', 2), ('tebetriebssystem', 2), ('tebenutzer', 2), ('tezugriffsstatus', 2), ('connects', 2), ('loggin', 2), ('zuehlke', 2), ('stoped', 2), ('recovering', 2), ('dceghpwn', 2), ('combined', 2), ('visibility', 2), ('routinely', 2), ('invitation', 2), ('panjkytr', 2), ('mehrotra', 2), ('keith', 2), ('bkmeuhfz', 2), ('beneath', 2), ('cas', 2), ('raw', 2), ('qt', 2), ('sees', 2), ('zlnfpuam', 2), ('aktplhre', 2), ('dependent', 2), ('requestor', 2), ('helmu', 2), ('mycompany', 2), ('fqhlvcxn', 2), ('drilling', 2), ('xz', 2), ('ahost', 2), ('aconnection', 2), ('lowercaseurlcorrelation', 2), ('srcip', 2), ('urlcorrelation', 2), ('vendorreference', 2), ('urlpath', 2), ('httpmethod', 2), ('devip', 2), ('dstip', 2), ('urlfullpath', 2), ('urlhost', 2), ('httpversion', 2), ('kingston', 2), ('falsche', 2), ('elements', 2), ('zwip', 2), ('metal', 2), ('vjuxfokc', 2), ('cwhxnoug', 2), ('usual', 2), ('lagqcompanyo', 2), ('fybwjzhx', 2), ('ways', 2), ('lryturhy', 2), ('workbooks', 2), ('oqvwgnkc', 2), ('gkjylpzx', 2), ('gajthyana', 2), ('hegdergyt', 2), ('grhryueg', 2), ('dewicrth', 2), ('kadyuiluza', 2), ('zkb', 2), ('transmitted', 2), ('mgjxwept', 2), ('synchronising', 2), ('langmar', 2), ('jertyur', 2), ('procenter', 2), ('motor', 2), ('exls', 2), ('pumjbcna', 2), ('scluvtyj', 2), ('romertanj', 2), ('zcnp', 2), ('sides', 2), ('godjevmy', 2), ('gfaevrdq', 2), ('pose', 2), ('lv', 2), ('jrdafplx', 2), ('fcnjmvts', 2), ('chinese', 2), ('relaunching', 2), ('watch', 2), ('elbaqmtp', 2), ('launguage', 2), ('impacting', 2), ('odo', 2), ('piece', 2), ('noticing', 2), ('elapsed', 2), ('meaning', 2), ('prpf', 2), ('deep', 2), ('dive', 2), ('interrupt', 2), ('realized', 2), ('scgtitt', 2), ('osterwalder', 2), ('intel', 2), ('pal', 2), ('sadipta', 2), ('sonia', 2), ('cyvdluja', 2), ('oxrkfpbz', 2), ('plsseald', 2), ('formation', 2), ('throwing', 2), ('vrtybundj', 2), ('ycimqxdn', 2), ('saqbgcpl', 2), ('ybfzcjiq', 2), ('letting', 2), ('accurate', 2), ('accesss', 2), ('jartnine', 2), ('chainverifier', 2), ('godaddy', 2), ('gd', 2), ('nxlzpgfr', 2), ('rlqowmyt', 2), ('licence', 2), ('alarms', 2), ('affect', 2), ('berfkting', 2), ('classified', 2), ('riuvxdas', 2), ('revised', 2), ('revise', 2), ('explanation', 2), ('bersicht', 2), ('rsvminjz', 2), ('tcpqvbae', 2), ('chefghtyn', 2), ('supports', 2), ('premises', 2), ('violation', 2), ('dsn', 2), ('tls', 2), ('relaxed', 2), ('mime', 2), ('favot', 2), ('jacques', 2), ('thread', 2), ('edab', 2), ('aefca', 2), ('slc', 2), ('uriscan', 2), ('prvs', 2), ('forefront', 2), ('nspm', 2), ('sfs', 2), ('mixed', 2), ('dxyvfuhr', 2), ('uyfqgomx', 2), ('reviewed', 2), ('clrgtydia', 2), ('jhxwiply', 2), ('midhcnze', 2), ('elt', 2), ('uploaded', 2), ('magonza', 2), ('evercast', 2), ('broadcast', 2), ('erase', 2), ('trs', 2), ('modules', 2), ('updown', 2), ('bucket', 2), ('doxmlcpr', 2), ('xjheyscu', 2), ('qjiutmel', 2), ('fgvtxeoy', 2), ('wghjkftewj', 2), ('esntuago', 2), ('kwxrdhuv', 2), ('qikvnjzc', 2), ('evmrcqug', 2), ('bcefayom', 2), ('lzhwcgvb', 2), ('lunjuws', 2), ('planet', 2), ('earth', 2), ('hcytr', 2), ('chanthrydru', 2), ('awyrthysm', 2), ('tzradpnj', 2), ('izlotycb', 2), ('printscreen', 2), ('subgtybaryuao', 2), ('zcopc', 2), ('uylvgtfi', 2), ('eovkxgpn', 2), ('staebefertigung', 2), ('gkerqucv', 2), ('bqumyrea', 2), ('difozlav', 2), ('dgbfptos', 2), ('subcontracting', 2), ('layout', 2), ('produce', 2), ('nakagtwsgs', 2), ('unfreeze', 2), ('rethtyuzkd', 2), ('licensed', 2), ('sndaofyw', 2), ('vichtyuky', 2), ('warhtyonack', 2), ('kirvecja', 2), ('dwgs', 2), ('stations', 2), ('disorder', 2), ('ztdgvclp', 2), ('gzcalstq', 2), ('galaxy', 2), ('minimum', 2), ('xkegcqov', 2), ('drctxjqi', 2), ('cowqyjzm', 2), ('refers', 2), ('cleaned', 2), ('unrbafjx', 2), ('reyshakw', 2), ('pcd', 2), ('tims', 2), ('wj', 2), ('abb', 2), ('tent', 2), ('ccc', 2), ('lcow', 2), ('clad', 2), ('standstill', 2), ('subsequent', 2), ('cables', 2), ('rolling', 2), ('aiul', 2), ('malgorzata', 2), ('gugala', 2), ('wpgmkt', 2), ('joanna', 2), ('futur', 2), ('evolution', 2), ('campaign', 2), ('gjbtuwek', 2), ('rgtoger', 2), ('lfgtiu', 2), ('dvd', 2), ('conforma', 2), ('kata', 2), ('sin', 2), ('jnb', 2), ('gurhyqsath', 2), ('ntteam', 2), ('fatgrtyma', 2), ('div', 2), ('xnqzhtwu', 2), ('demjqrfl', 2), ('fkdazsmi', 2), ('yecbrofv', 2), ('boss', 2), ('chandruhdty', 2), ('companyfoundation', 2), ('calculates', 2), ('percentages', 2), ('bswlorek', 2), ('yhdrlgbs', 2), ('bwa', 2), ('brauchen', 2), ('anst', 2), ('hre', 2), ('kurzfristig', 2), ('uns', 2), ('vorbei', 2), ('bringen', 2), ('nnuacyltoe', 2), ('mvhcoqed', 2), ('konjdmwq', 2), ('roaghyunokepc', 2), ('vvhstyap', 2), ('gartryhu', 2), ('cac', 2), ('tue', 2), ('invest', 2), ('bhughjdra', 2), ('pgeknaxy', 2), ('usokqprd', 2), ('zd', 2), ('baker', 2), ('scenario', 2), ('hx', 2), ('ctvaejbo', 2), ('mjcerqwo', 2), ('phlpiops', 2), ('mdflqwxg', 2), ('xqkydoat', 2), ('bveiyclr', 2), ('cantine', 2), ('adapters', 2), ('muqdlobv', 2), ('qflsdahg', 2), ('handled', 2), ('dispatch', 2), ('rohhsyni', 2), ('questioning', 2), ('canadian', 2), ('jesjnlyenm', 2), ('erkennen', 2), ('clip', 2), ('newweaver', 2), ('esprit', 2), ('cam', 2), ('jvshydix', 2), ('rzpmnylt', 2), ('tuning', 2), ('terhyury', 2), ('consuming', 2), ('schhdgtmips', 2), ('sidecar', 2), ('vcenter', 2), ('repository', 2), ('karansb', 2), ('yolktfas', 2), ('fyoxqgvh', 2), ('dms', 2), ('kpro', 2), ('fiber', 2), ('credited', 2), ('datum', 2), ('tigen', 2), ('geben', 2), ('attachement', 2), ('trotz', 2), ('anderer', 2), ('hgrvubzo', 2), ('telefonnummer', 2), ('msdotnet', 2), ('durumu', 2), ('attending', 2), ('kathatryuna', 2), ('vsbhyrt', 2), ('laijuttr', 2), ('peuckbvr', 2), ('tjihmgsv', 2), ('convenient', 2), ('cfokqnhz', 2), ('lizhwdoe', 2), ('wvqgbdhm', 2), ('tjlizqgc', 2), ('unrestricted', 2), ('dvw', 2), ('moves', 2), ('mkuhtyhui', 2), ('zolnubvq', 2), ('ehrqifxp', 2), ('bollmam', 2), ('wv', 2), ('bfb', 2), ('ion', 2), ('ht', 2), ('edd', 2), ('secomea', 2), ('helping', 2), ('gate', 2), ('realize', 2), ('zeiterfassung', 2), ('ersetzen', 2), ('lhqftphfm', 2), ('idoc', 2), ('travelers', 2), ('apusm', 2), ('bv', 2), ('eintragen', 2), ('aktivieren', 2), ('outllok', 2), ('hotspot', 2), ('whqsm', 2), ('gridgetcsvfile', 2), ('sources', 2), ('getcsvfile', 2), ('recommend', 2), ('attack', 2), ('suppress', 2), ('eusa', 2), ('edition', 2), ('controllable', 2), ('expr', 2), ('sanitized', 2), ('illegal', 2), ('fieldname', 2), ('leverage', 2), ('injectncqulao', 2), ('qauighdplicious', 2), ('arbitrary', 2), ('underlying', 2), ('edac', 2), ('evtmcm', 2), ('wjnbvchvsyxjpdhlbdg', 2), ('dptmmcg', 2), ('evtmawvszf', 2), ('wktttrvqgqfnbtfqgpsancnano', 2), ('nfvcbaueftuya', 2), ('ienptknbvchnrduoq', 2), ('fukcbau', 2), ('wzwsnkerplcbdt', 2), ('dqvqojzonlcbau', 2), ('fmvcapktttruxfq', 2), ('bwchlehryyskgrljptsbhzg', 2), ('pvcbovuxmo', 2), ('vsvcbjtlrpigbhzg', 2), ('bmftzwasygvtywlsycxgdxnlcm', 2), ('hbwvglgbwyxnzd', 2), ('yzgasygnyzwf', 2), ('zwrglgbsb', 2), ('dudfrtglgbyzwxvywrfywnsx', 2), ('zsywdglgbpc', 2), ('rpdmvglgblehryywasyhjwx', 2), ('vuycxgcnbfdg', 2), ('jlyxrlzf', 2), ('hdgapifzbtfvfuyaoj', 2), ('bmftzscsj', 2), ('vjdxjpdhlabwfnzw', 2), ('bvbgljescsqfbbu', 2), ('mstk', 2), ('xkcksmcwwldesqevyvfjble', 2), ('vtewsie', 2), ('pvygpkttjtlnfulqgsu', 2), ('utybgywrtaw', 2), ('szwagkhbhcmvudf', 2), ('cmvlx', 2), ('xldmvslhnvcnrfb', 2), ('exbllhvzzxjfawqsupply', 2), ('lksbwquxvrvmgkdesmiwwlcdvjywou', 2), ('vmrunuihvzzxjfawqgrljptsbhzg', 2), ('vyifdirvjfihvzzxjuyw', 2), ('bvbgljescplcdgaxjzdg', 2), ('jayb', 2), ('exblpufkbwluahrtbc', 2), ('pwdldenzdkzpbgv', 2), ('ype', 2), ('urle', 2), ('ncoded', 2), ('mcm', 2), ('wjnbvchv', 2), ('syxjpdhlbdg', 2), ('dptm', 2), ('mcg', 2), ('mawvszf', 2), ('wktttrvqgqfnbtfq', 2), ('gpsancnano', 2), ('nfvcb', 2), ('aueftuya', 2), ('ienptkn', 2), ('bvchnrduoq', 2), ('ukcbau', 2), ('wzwsnkerplcb', 2), ('dqvqojzonlcb', 2), ('fmvcapktttrux', 2), ('bwchlehryysk', 2), ('grljptsbhzg', 2), ('pvcb', 2), ('ovuxmo', 2), ('vsvcb', 2), ('jtlrpigbhzg', 2), ('bmftzwasygvtywl', 2), ('sycxgdxnlcm', 2), ('hbwv', 2), ('glgbwyxnzd', 2), ('yzga', 2), ('sygnyzwf', 2), ('zwrglgb', 2), ('sb', 2), ('dudfrtglgbyzwx', 2), ('vywrfywnsx', 2), ('zsywd', 2), ('glgbpc', 2), ('rpdmv', 2), ('glgblehryywasyhj', 2), ('wx', 2), ('vuycxgcnb', 2), ('fdg', 2), ('jlyxr', 2), ('lzf', 2), ('hdgapifzbtfv', 2), ('fuyaoj', 2), ('bmf', 2), ('tzscsj', 2), ('vjdxjpdhl', 2), ('abwfnzw', 2), ('vbgljescsqfbbu', 2), ('stk', 2), ('xkcksmcwwlde', 2), ('sqevyvfjble', 2), ('vtew', 2), ('pvygpkttjtln', 2), ('fulqgsu', 2), ('utybgywr', 2), ('taw', 2), ('szwagkhb', 2), ('hcmvudf', 2), ('cmv', 2), ('lx', 2), ('xldmvslhnvcnr', 2), ('exbllhvzzxjfawq', 2), ('lksb', 2), ('wquxvrvmgkdesmiw', 2), ('wlcdvjywou', 2), ('vmrun', 2), ('uihvzzxjfawqgrlj', 2), ('ptsbhzg', 2), ('yifdirvjfihvzzxj', 2), ('bvbgl', 2), ('jescplcdgaxjzdg', 2), ('ective', 2), ('exblpufkbwlua', 2), ('hrtbc', 2), ('pwdldenzd', 2), ('kzpbgv', 2), ('forw', 2), ('arded', 2), ('lhutkpxm', 2), ('cluster', 2), ('thomafghk', 2), ('anrufen', 2), ('tzt', 2), ('deine', 2), ('unlinked', 2), ('paystub', 2), ('powered', 2), ('idkfgcnq', 2), ('vjwhmzor', 2), ('olghiveii', 2), ('lack', 2), ('face', 2), ('vijghyduhprga', 2), ('yeghrrajghodu', 2), ('ethernet', 2), ('araghtyu', 2), ('parthyrubhji', 2), ('stefdgthy', 2), ('fxwjhapo', 2), ('ljztkmds', 2), ('sqmabtwn', 2), ('customermaster', 2), ('actions', 2), ('solicito', 2), ('instala', 2), ('agathon', 2), ('combi', 2), ('dbgrtqhs', 2), ('janhytrn', 2), ('ooshstyizen', 2), ('johan', 2), ('kok', 2), ('jofghan', 2), ('kddok', 2), ('hennidgtydhyue', 2), ('booysen', 2), ('gonzales', 2), ('fernandez', 2), ('tqvefyui', 2), ('spain', 2), ('nfc', 2), ('vsdtxwry', 2), ('ngkcdjye', 2), ('vsid', 2), ('anfghyudrejy', 2), ('appointments', 2), ('synchronize', 2), ('xmgptwho', 2), ('fmcxikqz', 2), ('talking', 2), ('zmm', 2), ('gtehdnyushot', 2), ('discription', 2), ('milan', 2), ('grtaoivq', 2), ('validated', 2), ('ekjobdiz', 2), ('ktelzwvg', 2), ('qcxivzag', 2), ('vyucbagx', 2), ('wauhocsk', 2), ('friends', 2), ('sudghhahjkkar', 2), ('estfhycoastrrc', 2), ('sagfhosh', 2), ('karhtyiio', 2), ('ufpzergq', 2), ('zchjbfdehivashankaraiah', 2), ('instalar', 2), ('compare', 2), ('zeichnung', 2), ('dnwfhpyl', 2), ('zqbldipk', 2), ('fabry', 2), ('zeichnungen', 2), ('versehentlich', 2), ('nachdem', 2), ('schon', 2), ('tut', 2), ('leid', 2), ('bisher', 2), ('plfwoagd', 2), ('chtpiazu', 2), ('subscribe', 2), ('dedicated', 2), ('bghakch', 2), ('rod', 2), ('axesnghb', 2), ('cyzuomxa', 2), ('ohio', 2), ('kabel', 2), ('bfqnvezs', 2), ('vwkasnxe', 2), ('jtgmpdcr', 2), ('vvaghjnthl', 2), ('funktion', 2), ('subsystem', 2), ('quyhn', 2), ('grp', 2), ('mghlisha', 2), ('baranwfhrty', 2), ('summit', 2), ('upcoming', 2), ('eben', 2), ('anmelden', 2), ('nun', 2), ('afplnyxb', 2), ('bixsapwu', 2), ('hehr', 2), ('maghyuigie', 2), ('opentext', 2), ('aktuellen', 2), ('folgenden', 2), ('bnsh', 2), ('hqntn', 2), ('apkqmrdu', 2), ('apvpn', 2), ('anlagenbereich', 2), ('liefern', 2), ('siavgtby', 2), ('ordners', 2), ('genannten', 2), ('aufstellen', 2), ('genius', 2), ('pen', 2), ('papier', 2), ('fujitsu', 2), ('vahxnfgl', 2), ('savin', 2), ('lpriokwa', 2), ('coworkers', 2), ('jfcrdavy', 2), ('sxpotjlu', 2), ('edmhihryu', 2), ('laffekr', 2), ('netscape', 2), ('yihelxgp', 2), ('eea', 2), ('earbuds', 2), ('fbhyeksq', 2), ('seghyurghei', 2), ('baurhty', 2), ('recurring', 2), ('proposal', 2), ('bereich', 2), ('besprochen', 2), ('january', 2), ('zupifghd', 2), ('vdqxepun', 2), ('rajyutyi', 2), ('xezjvnyr', 2), ('hmjwknxs', 2), ('obuwfnkm', 2), ('wear', 2), ('mobil', 2), ('vnc', 2), ('ricoh', 2), ('zfd', 2), ('zkln', 2), ('nfsbackup', 2), ('lnssm', 2), ('reinstated', 2), ('illinois', 2), ('bragtydlc', 2), ('lbs', 2), ('lab', 2), ('accrual', 2), ('shhkioaprhkuoash', 2), ('fixes', 2), ('recieve', 2), ('uyjlodhq', 2), ('ymedkatw', 2), ('lghuiezj', 2), ('ekxw', 2), ('hampered', 2), ('explained', 2), ('frafhyuo', 2), ('countries', 2), ('iqmhjlwr', 2), ('jqmxaybi', 2), ('guys', 2), ('vinhytry', 2), ('assumption', 2), ('hzptilsw', 2), ('wusdajqv', 2), ('leseberechtigung', 2), ('aqzcisjy', 2), ('raflghneib', 2), ('xighjacj', 2), ('messvorrichtung', 2), ('steli', 2), ('herstellen', 2), ('berwachung', 2), ('mask', 2), ('mc', 2), ('kuhgtyjvelu', 2), ('reliability', 2), ('ckmeldeterminal', 2), ('gegen', 2), ('kln', 2), ('folgender', 2), ('gwptzvxm', 2), ('sorahdyggs', 2), ('cighytol', 2), ('timnhyt', 2), ('rehtyulds', 2), ('restoring', 2), ('ojgfmvep', 2), ('zbatowgi', 2), ('contributing', 2), ('jundiai', 2), ('dargthy', 2), ('sohfyuimaiah', 2), ('bng', 2), ('qklrdoba', 2), ('qxivmbts', 2), ('wijuiidl', 2), ('association', 2), ('secureserver', 2), ('schtrtgoyhtsdale', 2), ('recibo', 2), ('saindo', 2), ('caracteres', 2), ('implemented', 2), ('todd', 2), ('muywpnof', 2), ('prtikusy', 2), ('xvgzdtqj', 2), ('instructed', 2), ('honest', 2), ('paychecks', 2), ('webfnhtyer', 2), ('einlegen', 2), ('bia', 2), ('traversecitymi', 2), ('fakeav', 2), ('wilmington', 2), ('tps', 2), ('doubleverify', 2), ('applewebkit', 2), ('khtml', 2), ('safari', 2), ('jetzt', 2), ('woche', 2), ('trainers', 2), ('pnroqajb', 2), ('psbyfhkg', 2), ('helpteam', 2), ('ansehen', 2), ('netzwerkkabel', 2), ('ngern', 2), ('xyjkndus', 2), ('carve', 2), ('directed', 2), ('oikhfqyl', 2), ('cabinet', 2), ('lijrvdgh', 2), ('wfhmdsoa', 2), ('quantum', 2), ('khspqlnj', 2), ('npgxuzeq', 2), ('znqcljxt', 2), ('azvoespk', 2), ('sundj', 2), ('bestellungen', 2), ('zaeduhlt', 2), ('jdgsamtv', 2), ('feld', 2), ('gespeichert', 2), ('chte', 2), ('gibt', 2), ('ale', 2), ('majority', 2), ('robankm', 2), ('iehs', 2), ('scroll', 2), ('fniqhjtg', 2), ('qrfuetpw', 2), ('minimize', 2), ('rechnungseingang', 2), ('erhalten', 2), ('liegen', 2), ('somit', 2), ('deshalb', 2), ('translation', 2), ('steinh', 2), ('transit', 2), ('anscheinend', 2), ('netzteil', 2), ('srgtycha', 2), ('klar', 2), ('rdest', 2), ('ebenfalls', 2), ('zugriffsrechte', 2), ('lichtyuiwu', 2), ('leibdrty', 2), ('restriction', 2), ('cplant', 2), ('gap', 2), ('increase', 2), ('ebkfwhgt', 2), ('flapokym', 2), ('zhang', 2), ('bajrpckl', 2), ('cevtiuob', 2), ('phd', 2), ('patent', 2), ('boivin', 2), ('hasghyusan', 2), ('htsnaodb', 2), ('absence', 2), ('davgtgyrh', 2), ('beschichtung', 2), ('retried', 2), ('krbb', 2), ('scfpr', 2), ('torx', 2), ('sar', 2), ('rghkiriuytes', 2), ('retrieval', 2), ('zipped', 2), ('softwares', 2), ('drtb', 2), ('discovered', 2), ('dfrt', 2), ('dofghbmes', 2), ('extending', 2), ('solidworks', 2), ('baoapacg', 2), ('jochegtyhu', 2), ('vvgtycargvc', 2), ('agbighyail', 2), ('producing', 2), ('dailyorderbillingreport', 2), ('mtd', 2), ('fehlgeschlagen', 2), ('ziehen', 2), ('overdue', 2), ('expect', 2), ('moin', 2), ('formulare', 2), ('sky', 2), ('studio', 2), ('debaghjsish', 2), ('pyeothbl', 2), ('agfxelwz', 2), ('arbeitsmedizin', 2), ('metghyjznk', 2), ('roedfghtec', 2), ('aiml', 2), ('financemssg', 2), ('tnorudbf', 2), ('lauftqmd', 2), ('hkjfmcqo', 2), ('gpcxltar', 2), ('pfgyhtu', 2), ('meines', 2), ('umstellung', 2), ('diesen', 2), ('urlaub', 2), ('bekannt', 2), ('schaem', 2), ('eigene', 2), ('dateien', 2), ('mdb', 2), ('alles', 2), ('songhyody', 2), ('imaging', 2), ('thumb', 2), ('abdhtyu', 2), ('comcast', 2), ('dears', 2), ('jvtsgmin', 2), ('firmware', 2), ('regions', 2), ('dump', 2), ('reddy', 2), ('candice', 2), ('helps', 2), ('formats', 2), ('oppurtunities', 2), ('snhdfihytu', 2), ('qjeymnzs', 2), ('cwuospin', 2), ('nbhoxqpe', 2), ('vrtx', 2), ('terralink', 2), ('dhkovprf', 2), ('joining', 2), ('werkzeuge', 2), ('erstellt', 2), ('profil', 2), ('barcelona', 2), ('xos', 2), ('tltz', 2), ('oy', 2), ('qko', 2), ('hn', 2), ('lcbq', 2), ('wd', 2), ('dca', 2), ('eed', 2), ('vatpgsxn', 2), ('dauerhaft', 2), ('bildschirmschoner', 2), ('festplatte', 2), ('stellen', 2), ('dicafxhv', 2), ('wsimcqzt', 2), ('xvwzbdhq', 2), ('books', 2), ('solutioning', 2), ('pagthyuathy', 2), ('afghtyjith', 2), ('vvdfgtyuji', 2), ('judthti', 2), ('zhu', 2), ('subcontract', 2), ('rough', 2), ('enters', 2), ('prepare', 2), ('wendt', 2), ('titanium', 2), ('craigfgh', 2), ('compromise', 2), ('anonymous', 2), ('tss', 2), ('ile', 2), ('hanna', 2), ('esxi', 2), ('ipad', 2), ('oxrnpuys', 2), ('oxizkwmq', 2), ('jayhrt', 2), ('bhatyr', 2), ('extn', 2), ('etiketten', 2), ('salesorg', 2), ('ubdihsop', 2), ('ahyeqpmx', 2), ('outstanding', 2), ('interco', 2), ('stefdgthyo', 2), ('kahrthyeui', 2), ('recording', 2), ('awysinic', 2), ('navigate', 2), ('schneider', 2), ('downs', 2), ('rtpmlwnk', 2), ('unpambrv', 2), ('imperative', 2), ('matheywt', 2), ('ztfn', 2), ('pak', 2), ('framdntye', 2), ('fron', 2), ('luifdsts', 2), ('olhryhira', 2), ('pam', 2), ('hlcrbuqa', 2), ('qznjshwm', 2), ('ojtmnpxc', 2), ('klacwufr', 2), ('delsonpereira', 2), ('kgyboafv', 2), ('tlzsrvgw', 2), ('vagnerlrtopes', 2), ('eeserra', 2), ('quina', 2), ('verificar', 2), ('ybhazlqp', 2), ('zfghsxiw', 2), ('corrections', 2), ('heavily', 2), ('mtdesign', 2), ('manuals', 2), ('utgszjrf', 2), ('pacfvxzk', 2), ('comp', 2), ('reaktivieren', 2), ('cet', 2), ('easier', 2), ('tkuivxrn', 2), ('urdgitsv', 2), ('etlfrucw', 2), ('michjnfyele', 2), ('jenhntyns', 2), ('reversal', 2), ('rhwsmefo', 2), ('tvphyura', 2), ('zkevitua', 2), ('udzkgbwl', 2), ('bgpedtqc', 2), ('imagens', 2), ('quando', 2), ('dalgtylam', 2), ('nv', 2), ('bhqvklgc', 2), ('vscdzjhg', 2), ('fyedqgzt', 2), ('jdqvuhlr', 2), ('recorded', 2), ('nfayqjhg', 2), ('kyswcpei', 2), ('nutzen', 2), ('rden', 2), ('uperformsystem', 2), ('worker', 2), ('communicating', 2), ('outlock', 2), ('schr', 2), ('technische', 2), ('beratung', 2), ('verkauf', 2), ('kundegty', 2), ('jeffrghryeytyf', 2), ('strigtyet', 2), ('jeffrghryeyrghryey', 2), ('strgrtyiet', 2), ('aiobpkzm', 2), ('rmfjwtpl', 2), ('montag', 2), ('anmeldeaccount', 2), ('laufzeitfehler', 2), ('dmitazhw', 2), ('kxbifzoh', 2), ('trx', 2), ('erpsid', 2), ('vvrtgwildj', 2), ('enth', 2), ('entsprechend', 2), ('bo', 2), ('anschlie', 2), ('damit', 2), ('kas', 2), ('fico', 2), ('teilweise', 2), ('kenametal', 2), ('sbfhydeep', 2), ('rekmqxfn', 2), ('zeitwirtschaft', 2), ('beheben', 2), ('scannt', 2), ('berpr', 2), ('samstag', 2), ('verehrte', 2), ('abhilfe', 2), ('hegdthy', 2), ('athjyul', 2), ('dixhtyuit', 2), ('companytm', 2), ('audiocodes', 2), ('hrlwizav', 2), ('elyamqro', 2), ('cajdwtgq', 2), ('mme', 2), ('discuss', 2), ('emb', 2), ('emergency', 2), ('azubi', 2), ('enterprises', 2), ('preferably', 2), ('konnte', 2), ('laufen', 2), ('vvdortddp', 2), ('yaxmwdth', 2), ('xsfgitmq', 2), ('tha', 2), ('transition', 2), ('geehrte', 2), ('fehlerhaft', 2), ('cord', 2), ('anwendungstechnik', 2), ('tortm', 2), ('howthrelte', 2), ('chicago', 2), ('mohgrtyan', 2), ('kaum', 2), ('kuznvase', 2), ('jrxtbuqz', 2), ('iygsxftl', 2), ('hysrbgad', 2), ('yubtgy', 2), ('angebotserstellung', 2), ('benutzername', 2), ('iwazgesl', 2), ('ydgqtpbo', 2), ('gufwhdky', 2), ('cwfzldts', 2), ('dlmukhyn', 2), ('requistions', 2), ('catalogues', 2), ('umzug', 2), ('parrfgyksm', 2), ('hergestellt', 2), ('bhergtyemm', 2), ('speichern', 2), ('maier', 2), ('shunshen', 2), ('paralegal', 2), ('remaining', 2), ('banco', 2), ('jobuacyltoe', 2), ('bghrbie', 2), ('crhyley', 2), ('frederirtck', 2), ('reciever', 2), ('requsted', 2), ('ald', 2), ('isr', 2), ('mainswitch', 2), ('openning', 2), ('tghrloks', 2), ('jbgcvlmf', 2), ('creator', 2), ('reparar', 2), ('workarounds', 2), ('netzlaufwerke', 2), ('tzung', 2), ('war', 2), ('welches', 2), ('benutz', 2), ('namen', 2), ('benutzen', 2), ('sqlmtixr', 2), ('urhbvfgd', 2), ('rds', 2), ('recs', 2), ('responsive', 2), ('kick', 2), ('hearing', 2), ('ak', 2), ('spro', 2), ('ebusaar', 2), ('munnangi', 2), ('serthyei', 2), ('datenbanken', 2), ('kontakt', 2), ('ufgkybsh', 2), ('ijswtdve', 2), ('tooldplcmmaninp', 2), ('lockouts', 2), ('lee', 2), ('decimal', 2), ('subsite', 2), ('flapping', 2), ('inputs', 2), ('num', 2), ('hntl', 2), ('mitctdrh', 2), ('lap', 2), ('downgrade', 2), ('trafficdenied', 2), ('internetsurvey', 2), ('erratasec', 2), ('atlanta', 2), ('cioehrnq', 2), ('wfyhgelz', 2), ('mitarbeiterin', 2), ('atualiza', 2), ('beschichten', 2), ('vvkthyiska', 2), ('east', 2), ('rcbdyslq', 2), ('zuspjbtw', 2), ('zrpemyab', 2), ('xvzwcbha', 2), ('searched', 2), ('wsp', 2), ('fm', 2), ('awysv', 2), ('richthammer', 2), ('michthey', 2), ('sudghnthdra', 2), ('betapfasen', 2), ('dash', 2), ('xzs', 2), ('dnb', 2), ('wwisvc', 2), ('chkdsk', 2), ('percent', 2), ('exploited', 2), ('potentially', 2), ('interfere', 2), ('rows', 2), ('tcode', 2), ('mgndhtillen', 2), ('aler', 2), ('reaches', 2), ('knowledgebase', 2), ('yourself', 2), ('comment', 2), ('oslzvpgk', 2), ('nhwsxgpb', 2), ('zspvxrfk', 2), ('xocyhnkf', 2), ('prb', 2), ('restarts', 2), ('author', 2), ('jquery', 2), ('hinge', 2), ('jiftg', 2), ('anftgup', 2), ('nftgyair', 2), ('preparedness', 2), ('scenarios', 2), ('congratuldhyation', 2), ('highlighting', 2), ('sunil', 2), ('interfacetengigabitethernet', 2), ('yakimp', 2), ('businessobjects', 2), ('menus', 2), ('whzbrusx', 2), ('planners', 2), ('browserproblem', 2), ('upgrades', 2), ('geschlossen', 2), ('presserei', 2), ('ezrsdgfc', 2), ('hofgvwel', 2), ('jnqmvard', 2), ('jiazxvcl', 2), ('blade', 2), ('monitorixepyfbga', 2), ('wtqdyoinware', 2), ('sensors', 2), ('yqlvfkih', 2), ('folbpugd', 2), ('rollout', 2), ('targetlist', 2), ('hibernation', 2), ('flight', 2), ('netviewer', 2), ('classifications', 2), ('suzjhmfa', 2), ('dest', 2), ('gest', 2), ('dtwlrofu', 2), ('etwas', 2), ('quotations', 2), ('periodic', 2), ('recovered', 2), ('lhqw', 2), ('anzeige', 2), ('lit', 2), ('cooperation', 2), ('surely', 2), ('abhay', 2), ('companypzyre', 2), ('mqlsfkre', 2), ('ldnfgt', 2), ('loovexfbjy', 2), ('lmcaqfkz', 2), ('loc', 2), ('commonly', 2), ('programdntymer', 2), ('explicitly', 2), ('coding', 2), ('subroutine', 2), ('mgermanyger', 2), ('wins', 2), ('leveraged', 2), ('propagation', 2), ('ulm', 2), ('broadscanning', 2), ('wz', 2), ('quest', 2), ('zihrtyud', 2), ('eqiyskhm', 2), ('robhyertyjo', 2), ('qiyujevw', 2), ('ogadikxv', 2), ('aghl', 2), ('nazarr', 2), ('israey', 2), ('nahumo', 2), ('tevkia', 2), ('agvw', 2), ('stable', 2), ('volunteer', 2), ('fmjeaoih', 2), ('ndyezlkb', 2), ('steffen', 2), ('ragsbdhryu', 2), ('jagthyin', 2), ('spell', 2), ('wza', 2), ('reassignment', 2), ('fund', 2), ('prtsid', 2), ('brings', 2), ('produkt', 2), ('nvawmlch', 2), ('ubyjolnc', 2), ('webside', 2), ('vvghychamc', 2), ('trueinternet', 2), ('nagfghtyudra', 2), ('cytohwau', 2), ('qfunricw', 2), ('anschalten', 2), ('einstellung', 2), ('patino', 2), ('nlearzwi', 2), ('ukdzstwi', 2), ('cfgxpvzi', 2), ('dvpnfbrc', 2), ('olifgtmpio', 2), ('probe', 2), ('tupsgrnd', 2), ('official', 2), ('lotus', 2), ('zpfitlyu', 2), ('cemvwyso', 2), ('mobiltelefon', 2), ('setzten', 2), ('bdc', 2), ('lokced', 2), ('jdcbiezx', 2), ('mrp', 2), ('weiterleiten', 2), ('umgestellt', 2), ('termin', 2), ('bertragung', 2), ('lochthowe', 2), ('einer', 2), ('moeglich', 2), ('krsxguty', 2), ('odqpwsgi', 2), ('uzavdmoj', 2), ('wrpkolzq', 2), ('isyfngdz', 2), ('flavio', 2), ('maihdlne', 2), ('xodeqlsv', 2), ('dfjbnrem', 2), ('vabqwxlm', 2), ('imeytghj', 2), ('lyrikzep', 2), ('fhkebpyx', 2), ('tqpermnu', 2), ('uvyjpixc', 2), ('kbfcrauw', 2), ('garcia', 2), ('placido', 2), ('skus', 2), ('retaining', 2), ('nxloukai', 2), ('ivxybewh', 2), ('sadghryioshkurtyar', 2), ('hiremath', 2), ('dcbligso', 2), ('mfhquljk', 2), ('mfyivqes', 2), ('cpihaxbs', 2), ('netframdntyework', 2), ('holiday', 2), ('ugi', 2), ('bvfdnale', 2), ('modern', 2), ('prospects', 2), ('tfedground', 2), ('zcqnuawo', 2), ('fqdn', 2), ('olvidley', 2), ('foreseeglobalmodelassessmt', 2), ('foreseemalprobglobalmodel', 2), ('globalmodelversion', 2), ('xac', 2), ('indaituba', 2), ('httpstatuscode', 2), ('dstassetofinterest', 2), ('foreseesrcipgeo', 2), ('franhtyufurt', 2), ('httpcontenttype', 2), ('xcc', 2), ('futureinterest', 2), ('chunked', 2), ('transfe', 2), ('chun', 2), ('ked', 2), ('tureinterest', 2), ('cbf', 2), ('centralsamerirtca', 2), ('fieldsales', 2), ('jre', 2), ('javaw', 2), ('bcc', 2), ('bddb', 2), ('heuristic', 2), ('ffddfee', 2), ('fbf', 2), ('algorithm', 2), ('untrustworthy', 2), ('prevalence', 2), ('fewer', 2), ('sensitivity', 2), ('encrypts', 2), ('demands', 2), ('decrypt', 2), ('campaigns', 2), ('researchers', 2), ('filenames', 2), ('sets', 2), ('persistence', 2), ('ladybi', 2), ('svchost', 2), ('hkcu', 2), ('creates', 2), ('kasperskylab', 2), ('eset', 2), ('avast', 2), ('bitmap', 2), ('unconfirmed', 2), ('speculation', 2), ('botnet', 2), ('distributing', 2), ('bugat', 2), ('dridex', 2), ('emanates', 2), ('shiz', 2), ('shifu', 2), ('conclusive', 2), ('affiliates', 2), ('locate', 2), ('encrypt', 2), ('analysing', 2), ('enumerate', 2), ('qcow', 2), ('vmdk', 2), ('tar', 2), ('bz', 2), ('jpeg', 2), ('sqlite', 2), ('ppsm', 2), ('potm', 2), ('docm', 2), ('wallet', 2), ('xlsm', 2), ('xlsb', 2), ('dotm', 2), ('dotx', 2), ('djvu', 2), ('pptm', 2), ('xltx', 2), ('xltm', 2), ('ppsx', 2), ('ppam', 2), ('docb', 2), ('potx', 2), ('lay', 2), ('msid', 2), ('sldm', 2), ('sldx', 2), ('tiff', 2), ('sqlitedb', 2), ('nome', 2), ('portugu', 2), ('cxl', 2), ('cada', 2), ('aliv', 2), ('connecti', 2), ('mozi', 2), ('lla', 2), ('compati', 2), ('ble', 2), ('indows', 2), ('urlenc', 2), ('oded', 2), ('ngth', 2), ('nom', 2), ('ugu', 2), ('nti', 2), ('lugin', 2), ('carbide', 2), ('mensagem', 2), ('destinat', 2), ('rios', 2), ('esta', 2), ('agora', 2), ('hajghtdul', 2), ('vvfrtgarnb', 2), ('dinner', 2), ('cfkwxlmq', 2), ('jimdghty', 2), ('fim', 2), ('prgthyuulla', 2), ('taqekwrd', 2), ('mdm', 2), ('workgroup', 2), ('hdb', 2), ('chmielewski', 2), ('upajtkbn', 2), ('wzyspovl', 2), ('localhost', 2), ('kebogxzp', 2), ('difnjlkp', 2), ('jegpvyrh', 2), ('kasnhdrb', 2), ('altogether', 2), ('zuteillisten', 2), ('kantenverrunden', 2), ('sammelarbpl', 2), ('doubleklick', 2), ('measurement', 2), ('inserts', 2), ('vikrhtyaskurtyar', 2), ('stepfhryhan', 2), ('urlaubsplanung', 2), ('allgemeines', 2), ('berirtchtswesen', 2), ('gebiet', 2), ('teamcall', 2), ('teammeeting', 2), ('nizholae', 2), ('bjnqikym', 2), ('vkrqojyt', 2), ('slt', 2), ('rushethryli', 2), ('bigrtdfatta', 2), ('jlzsardp', 2), ('kumtcnwi', 2), ('ramdntysey', 2), ('gergrythg', 2), ('pragtyhusas', 2), ('mondhrbaz', 2), ('xwlcqfsr', 2), ('lbcqfnie', 2), ('wqinjkxs', 2), ('azoyklqe', 2), ('customerattributes', 2), ('customeraddress', 2), ('customername', 2), ('pasgryowski', 2), ('dfupksnr', 2), ('drnqjzph', 2), ('epilwzux', 2), ('qjhitbcr', 2), ('sadmin', 2), ('hhnght', 2), ('pressing', 2), ('bjitvswa', 2), ('yrmugfnq', 2), ('enhanced', 2), ('mountpoint', 1), ('sgxqsuojr', 1), ('xwbesorf', 1), ('undocking', 1), ('gentles', 1), ('prtjc', 1), ('wihuyjdo', 1), ('qpogfwkb', 1), ('prqos', 1), ('anwmfvlgenkataramdntyana', 1), ('cnhkypxw', 1), ('lafncksi', 1), ('dlv', 1), ('sobkz', 1), ('umsok', 1), ('kzbws', 1), ('kzvbr', 1), ('ave', 1), ('yno', 1), ('rgtry', 1), ('stamped', 1), ('bapireturnederrorexception', 1), ('handlereturnstructureorreturntableline', 1), ('wt', 1), ('handlereturnstructure', 1), ('generirtcfunction', 1), ('objmod', 1), ('aju', 1), ('checkoutbapi', 1), ('rb', 1), ('downloaddocumentoriginals', 1), ('actionperformed', 1), ('obr', 1), ('api', 1), ('ecprjbod', 1), ('litmjwsy', 1), ('isfadulo', 1), ('etkyjabn', 1), ('bgqpotek', 1), ('cuxakvml', 1), ('stub', 1), ('upitdmhz', 1), ('owupktcg', 1), ('cruzjc', 1), ('bloqueado', 1), ('forma', 1), ('temporal', 1), ('sincronizaci', 1), ('dispositivo', 1), ('vil', 1), ('mediante', 1), ('hasta', 1), ('autorice', 1), ('acceso', 1), ('expert', 1), ('belo', 1), ('wmsg', 1), ('serrver', 1), ('reproduced', 1), ('roughing', 1), ('thryeu', 1), ('xvgftyr', 1), ('tryfuh', 1), ('cnkoflhi', 1), ('abeoucfj', 1), ('vdhfy', 1), ('scrapped', 1), ('djhadkudhd', 1), ('milliseconds', 1), ('angry', 1), ('documenting', 1), ('qtrmxlgv', 1), ('dfruzvam', 1), ('dgrtrkjs', 1), ('wseacnvi', 1), ('azvixyqg', 1), ('jdhdw', 1), ('hckvpary', 1), ('emxbpkwy', 1), ('woodscf', 1), ('hdjdkt', 1), ('wasted', 1), ('cee', 1), ('polska', 1), ('ul', 1), ('krzywoustego', 1), ('siedzib', 1), ('polandiu', 1), ('ka', 1), ('zarejestrowana', 1), ('dzie', 1), ('rejonowym', 1), ('nowe', 1), ('miasto', 1), ('wilda', 1), ('wydzia', 1), ('viii', 1), ('gospodarczy', 1), ('krs', 1), ('pod', 1), ('numerem', 1), ('kapita', 1), ('zak', 1), ('adowy', 1), ('pln', 1), ('abcdri', 1), ('tys', 1), ('fgehvwxb', 1), ('ckxegsqv', 1), ('ijeqpkrz', 1), ('nwtehsyx', 1), ('grauw', 1), ('enhances', 1), ('dfgry', 1), ('zscxqdhoalaramdntyan', 1), ('attendee', 1), ('abcdegy', 1), ('titcket', 1), ('xbdht', 1), ('yrjhd', 1), ('aqjdvexo', 1), ('lmedazjo', 1), ('advisor', 1), ('bdtryh', 1), ('receivng', 1), ('pnp', 1), ('fatal', 1), ('chrtyad', 1), ('thrdy', 1), ('tgyu', 1), ('approving', 1), ('mahcine', 1), ('khdgd', 1), ('hdytrkfiu', 1), ('rfqhoaky', 1), ('rncspbot', 1), ('ikxjfnml', 1), ('kaocbpfr', 1), ('tshljagr', 1), ('mhyiopxr', 1), ('zgmdster', 1), ('bdvcealj', 1), ('xjzcbgnp', 1), ('vfkwscao', 1), ('pushixepyfbga', 1), ('dispatching', 1), ('gstry', 1), ('arexjftu', 1), ('ohxdwngl', 1), ('konnica', 1), ('aufgeh', 1), ('bgdxitwu', 1), ('fdgrty', 1), ('kybgepnj', 1), ('idszleru', 1), ('acknowledgements', 1), ('ourhmqta', 1), ('tjwnqexo', 1), ('accts', 1), ('hdyrugdty', 1), ('receiver', 1), ('pamxszek', 1), ('serirtce', 1), ('renewal', 1), ('versa', 1), ('tryhdty', 1), ('webapps', 1), ('webapp', 1), ('toolt', 1), ('appoval', 1), ('yrada', 1), ('tyss', 1), ('ashdtyf', 1), ('neues', 1), ('cwrikael', 1), ('oanmsecr', 1), ('resetfor', 1), ('mfrsnagc', 1), ('mhakdfjo', 1), ('reso', 1), ('tehdy', 1), ('fcyuqvoj', 1), ('ajqeidlm', 1), ('fgxprnub', 1), ('hlanwgqj', 1), ('tgryhu', 1), ('hschicht', 1), ('tfnzwycd', 1), ('bicohjga', 1), ('icvulkwh', 1), ('udnyietg', 1), ('esrs', 1), ('brdhdd', 1), ('dhwduw', 1), ('persist', 1), ('channels', 1), ('bog', 1), ('choppy', 1), ('hdty', 1), ('yrhxssytu', 1), ('shstrud', 1), ('uploads', 1), ('langsadgtym', 1), ('ywqgrbnx', 1), ('jwnsyzbv', 1), ('tgeyd', 1), ('gdthryd', 1), ('technologist', 1), ('hbcejwsz', 1), ('cejsmdpf', 1), ('createknownaccountssummary', 1), ('replying', 1), ('hdthy', 1), ('hstdd', 1), ('gdthrujt', 1), ('commerirtcal', 1), ('flush', 1), ('winsock', 1), ('intercom', 1), ('ekpsm', 1), ('trhdyd', 1), ('mffbsf', 1), ('tender', 1), ('procurement', 1), ('qalyeunp', 1), ('eiyrcxog', 1), ('jnjxbq', 1), ('trgdyyufs', 1), ('vvgtyrhds', 1), ('vvkuhtdppg', 1), ('nikulatrhdy', 1), ('kroschke', 1), ('kalendereintr', 1), ('thrydufg', 1), ('standing', 1), ('wam', 1), ('udo', 1), ('trhsysba', 1), ('geschikt', 1), ('raum', 1), ('verbinden', 1), ('tkt', 1), ('hors', 1), ('suspision', 1), ('thnak', 1), ('enclsoed', 1), ('approva', 1), ('tgryds', 1), ('cccethxakm', 1), ('thrys', 1), ('hsdbdtt', 1), ('kp', 1), ('yscgjexz', 1), ('hxlbvjgf', 1), ('lhqsv', 1), ('followup', 1), ('associate', 1), ('retired', 1), ('myhrt', 1), ('preserved', 1), ('danie', 1), ('utejhdyd', 1), ('hrm', 1), ('bkzcfmse', 1), ('naslrwdb', 1), ('freeze', 1), ('meter', 1), ('trhsys', 1), ('hrydjs', 1), ('vvtgryhud', 1), ('axhkewnv', 1), ('zpumhlic', 1), ('tgrsyduf', 1), ('cybersecurity', 1), ('thrdyd', 1), ('dhdtwdd', 1), ('thrdyakdj', 1), ('yhtdush', 1), ('jqeczxtn', 1), ('gfjcronyudhakar', 1), ('thtudb', 1), ('ghtysui', 1), ('dnlhsgyo', 1), ('newducsl', 1), ('workplanning', 1), ('bbo', 1), ('possibilites', 1), ('collegue', 1), ('thryduf', 1), ('hddwtra', 1), ('consulting', 1), ('reactivation', 1), ('etdh', 1), ('thsydaas', 1), ('recurrent', 1), ('penn', 1), ('inspectors', 1), ('trhfyd', 1), ('sugajadd', 1), ('rtrhfyd', 1), ('tickert', 1), ('athrdyau', 1), ('qnxfegjw', 1), ('rljdhmwb', 1), ('thrydsss', 1), ('funke', 1), ('bildband', 1), ('vepxdgot', 1), ('poezmwny', 1), ('incomig', 1), ('tck', 1), ('solving', 1), ('western', 1), ('adwjfpbreu', 1), ('addon', 1), ('botton', 1), ('thrydksd', 1), ('trhdaa', 1), ('startpassword', 1), ('mvgr', 1), ('vbapkom', 1), ('timerecording', 1), ('terminals', 1), ('bcom', 1), ('zae', 1), ('wnvlmsjr', 1), ('znbvlygd', 1), ('trhsydsff', 1), ('backorderreports', 1), ('trhadg', 1), ('cnhgysju', 1), ('evaluations', 1), ('degree', 1), ('counterpart', 1), ('theajdlkadyt', 1), ('hrtgsd', 1), ('lqipwdeg', 1), ('dkbmjnvl', 1), ('schetrhsdlw', 1), ('ndthwedwys', 1), ('gqchtedl', 1), ('ye', 1), ('sahtym', 1), ('wanthryg', 1), ('lfal', 1), ('bothms', 1), ('brandtrhee', 1), ('pjdhfitman', 1), ('bachsdadgtadw', 1), ('badges', 1), ('cst', 1), ('pmgzjikq', 1), ('potmrkxy', 1), ('jeftryhf', 1), ('szrglyte', 1), ('stvyhzxn', 1), ('shadakjsdd', 1), ('twejhda', 1), ('asjadjs', 1), ('aeftjxos', 1), ('lhnyofad', 1), ('majsdtnrio', 1), ('thadasgg', 1), ('ccghksdm', 1), ('bsddqd', 1), ('jgdydqqd', 1), ('vmeljsfb', 1), ('ymxejsbd', 1), ('kstdaddaad', 1), ('fievgddtrr', 1), ('hpelnwco', 1), ('byjgvdzf', 1), ('bloqued', 1), ('failes', 1), ('tanks', 1), ('joetrhud', 1), ('eziswfym', 1), ('cehwzojy', 1), ('garthyhtuy', 1), ('kirathrydan', 1), ('pntp', 1), ('zntc', 1), ('bobjee', 1), ('facets', 1), ('exploring', 1), ('inconsistency', 1), ('idb', 1), ('curser', 1), ('trys', 1), ('hxgayczemii', 1), ('oypnxftq', 1), ('collapsed', 1), ('gvxfymjk', 1), ('euioadyf', 1), ('spanish', 1), ('lothryra', 1), ('conflict', 1), ('carthygyrol', 1), ('csn', 1), ('cahnge', 1), ('mgvpoyqd', 1), ('weupycai', 1), ('epqhduro', 1), ('pkwcdbrv', 1), ('fuser', 1), ('efrjkspc', 1), ('sfhbunrp', 1), ('oxkhntpl', 1), ('xwszgidt', 1), ('vxpcnrtw', 1), ('xelhoicd', 1), ('sge', 1), ('sgwipoxns', 1), ('usatdhdal', 1), ('xyedbsnm', 1), ('chtrhysdrystal', 1), ('oxkghdbr', 1), ('dsyvalof', 1), ('sdjdskjdkyr', 1), ('departements', 1), ('thryad', 1), ('bettery', 1), ('dcd', 1), ('bfnvjgxd', 1), ('trqmnpvu', 1), ('druckerfunktionsst', 1), ('admits', 1), ('mistrials', 1), ('rddept', 1), ('hathryrtmut', 1), ('benjamtrhdyin', 1), ('ckmelden', 1), ('ottyhddok', 1), ('thielpwii', 1), ('lobodeid', 1), ('loksdkdjwda', 1), ('edge', 1), ('vp', 1), ('cpmmecial', 1), ('gjbcengineering', 1), ('tooll', 1), ('gidsekfo', 1), ('chucashadqc', 1), ('wsljdqqds', 1), ('illustrated', 1), ('magazine', 1), ('loaction', 1), ('vvjotsgssea', 1), ('fdrf', 1), ('requriment', 1), ('ovnedzxg', 1), ('pylshdvb', 1), ('funktionsst', 1), ('ly', 1), ('inqogkxz', 1), ('rgmslhjn', 1), ('shifting', 1), ('gqdaikbv', 1), ('dnis', 1), ('thsaqsh', 1), ('pthyu', 1), ('wshqqhdqh', 1), ('vasanqi', 1), ('tehsauadd', 1), ('asjdidwni', 1), ('internally', 1), ('collogues', 1), ('putyrh', 1), ('awywkwd', 1), ('awywkjsw', 1), ('tahamt', 1), ('thry', 1), ('ldikdowdfm', 1), ('tomyh', 1), ('oen', 1), ('lxrponic', 1), ('lyszwcxg', 1), ('ranlpbmw', 1), ('djwkylif', 1), ('theeadjjd', 1), ('plugging', 1), ('wester', 1), ('crohuani', 1), ('dtjvhyob', 1), ('eseer', 1), ('hygxzklauthuchidambaramdnty', 1), ('thsadyu', 1), ('dwwlhews', 1), ('anantadth', 1), ('bwhrerattr', 1), ('rpmwh', 1), ('ktthasb', 1), ('dwon', 1), ('kmzucxgq', 1), ('refusing', 1), ('reponding', 1), ('systemic', 1), ('wide', 1), ('encrypted', 1), ('thesdf', 1), ('sdlwfkvach', 1), ('gaeycbwd', 1), ('populated', 1), ('janhduh', 1), ('keehad', 1), ('fvkgaalen', 1), ('gqkedvzu', 1), ('czoniyra', 1), ('dwivethn', 1), ('bertsckaadyd', 1), ('moranm', 1), ('mdwydindy', 1), ('mwdlkloran', 1), ('yrlsguzk', 1), ('fasyiokl', 1), ('xml', 1), ('terminating', 1), ('teamgdnwlkit', 1), ('gsm', 1), ('suplier', 1), ('shkdwd', 1), ('dlwdwd', 1), ('dfetvmzq', 1), ('brxavtzp', 1), ('jgxmafwk', 1), ('walk', 1), ('egjwxhlo', 1), ('weofysln', 1), ('engilsh', 1), ('qkgnwxto', 1), ('dwtivjrp', 1), ('hiatchi', 1), ('acmglkti', 1), ('cwgxrabu', 1), ('glitch', 1), ('unpost', 1), ('reality', 1), ('lomzfqns', 1), ('htznsgdf', 1), ('yhrdw', 1), ('hdld', 1), ('geman', 1), ('jwbsdd', 1), ('ddmefoche', 1), ('lions', 1), ('knicrhtyt', 1), ('layers', 1), ('freybtrhsdl', 1), ('plot', 1), ('lpawhdt', 1), ('doubt', 1), ('wswdd', 1), ('djdwol', 1), ('apacc', 1), ('fragmentation', 1), ('activitiess', 1), ('kba', 1), ('defense', 1), ('hxgayczeen', 1), ('nkssc', 1), ('vavilova', 1), ('quantety', 1), ('pihddltzr', 1), ('werkleitunggermany', 1), ('tejahd', 1), ('easdwmdwrappa', 1), ('reactive', 1), ('dudyhuyv', 1), ('netpath', 1), ('solarwind', 1), ('piltzrnj', 1), ('hkruyqfc', 1), ('aouezihl', 1), ('rrmx', 1), ('purely', 1), ('ecker', 1), ('breaks', 1), ('haughty', 1), ('nsdwd', 1), ('mwdddlleh', 1), ('fenthgh', 1), ('jmxqhrfa', 1), ('vawptbfl', 1), ('fever', 1), ('hwddwwd', 1), ('wdflefrong', 1), ('wdnwe', 1), ('svfuhlnx', 1), ('pojhkxua', 1), ('frpxtsca', 1), ('dnqdqld', 1), ('sasqkjqh', 1), ('lwddkqddq', 1), ('vvtdfettc', 1), ('okmhzgcq', 1), ('wzvetbqa', 1), ('talent', 1), ('kristina', 1), ('cope', 1), ('mtdyuhki', 1), ('fdnrxaci', 1), ('dalmdwppi', 1), ('fourth', 1), ('appeares', 1), ('theecanse', 1), ('wdleell', 1), ('skads', 1), ('wdlmdwwck', 1), ('jdcbiezxs', 1), ('hddwdw', 1), ('lwdwdwdr', 1), ('customization', 1), ('plam', 1), ('diwhdd', 1), ('jwddkwor', 1), ('dwwkd', 1), ('wdjwd', 1), ('wdnwk', 1), ('wdwmd', 1), ('wdkfww', 1), ('whwdiuw', 1), ('wdnwwl', 1), ('kwfwdw', 1), ('wdkwdwd', 1), ('dymo', 1), ('labelwriter', 1), ('turbo', 1), ('gages', 1), ('membership', 1), ('bdwdwarbara', 1), ('toolroom', 1), ('wwdyuan', 1), ('yqddquanw', 1), ('guy', 1), ('laid', 1), ('ouutlook', 1), ('systemen', 1), ('aurwddwacher', 1), ('retirement', 1), ('pension', 1), ('instuctrion', 1), ('superiors', 1), ('zust', 1), ('ndigen', 1), ('waldjrrm', 1), ('uskydftv', 1), ('sgwbfkjz', 1), ('itbof', 1), ('pane', 1), ('addins', 1), ('vale', 1), ('urgente', 1), ('customizing', 1), ('img', 1), ('xhlg', 1), ('pmm', 1), ('atm', 1), ('variables', 1), ('iformed', 1), ('ina', 1), ('cosid', 1), ('pistol', 1), ('webr', 1), ('gers', 1), ('gogtyektdgwo', 1), ('sandblasting', 1), ('ringtone', 1), ('mathes', 1), ('uexodhqa', 1), ('txqoifsl', 1), ('satisfied', 1), ('optional', 1), ('cnctf', 1), ('hedjdbwlmut', 1), ('nwwiebler', 1), ('unzfipac', 1), ('opwzmlbc', 1), ('awddmwdol', 1), ('mwddwansuke', 1), ('sqqqd', 1), ('zlkmlwdwdade', 1), ('nidqknwjktin', 1), ('dewkiodshpande', 1), ('temperory', 1), ('workmen', 1), ('nxcfastp', 1), ('xnwtyebg', 1), ('vvandwkjis', 1), ('hkydrfdw', 1), ('fregabe', 1), ('chrashes', 1), ('wqw', 1), ('ksdvp', 1), ('descriptions', 1), ('beachten', 1), ('withn', 1), ('hxgayczedistributor', 1), ('hxgayczecompany', 1), ('csscdddwsawdrill', 1), ('sloved', 1), ('rk', 1), ('mesz', 1), ('qmsoft', 1), ('toolonicserviceagent', 1), ('fsp', 1), ('bpa', 1), ('bpc', 1), ('jkddwkwd', 1), ('ngtr', 1), ('qzkyugce', 1), ('etsmnuba', 1), ('dondwdgj', 1), ('kwddwdw', 1), ('hudfefwe', 1), ('gross', 1), ('kgs', 1), ('ksjfye', 1), ('fekfeealleh', 1), ('vqefplhm', 1), ('mfpjaleo', 1), ('yandfgs', 1), ('zvjwleuo', 1), ('tdfqgcal', 1), ('trtgoywdd', 1), ('povirttch', 1), ('cmifnspg', 1), ('icauzxfh', 1), ('bomsdgar', 1), ('ace', 1), ('owenssdcl', 1), ('wsjkbw', 1), ('owwddwens', 1), ('qhtvlrxe', 1), ('knows', 1), ('gear', 1), ('lkzddens', 1), ('dint', 1), ('flea', 1), ('ersuche', 1), ('mechmet', 1), ('sllwdw', 1), ('wsjsoiwd', 1), ('yw', 1), ('ycgkinov', 1), ('czoparqg', 1), ('johddnthay', 1), ('welwsswbtwe', 1), ('keddsdn', 1), ('wethrybb', 1), ('bdm', 1), ('kellkwdy', 1), ('grab', 1), ('undating', 1), ('duplicating', 1), ('yrhackgt', 1), ('sfhxckgq', 1), ('francestrhuco', 1), ('uvdqkbgi', 1), ('suabfdci', 1), ('mesaage', 1), ('krthdelly', 1), ('sthytachnik', 1), ('datacntr', 1), ('rolled', 1), ('konica', 1), ('potthryzler', 1), ('benutzerkennung', 1), ('potsffwzlo', 1), ('systemfehler', 1), ('potzlow', 1), ('doyhtuug', 1), ('endlkglfeghart', 1), ('switchover', 1), ('winwip', 1), ('ksff', 1), ('rpgcdbfa', 1), ('reuwibpt', 1), ('userlogin', 1), ('hellow', 1), ('pildladjadga', 1), ('evry', 1), ('bokrgadgsu', 1), ('esdwduobrlcn', 1), ('wibovsru', 1), ('ulmqyhsk', 1), ('drwfubia', 1), ('cphemg', 1), ('emealeitung', 1), ('hjokrfpv', 1), ('fhpaxsqc', 1), ('lzvdyouh', 1), ('imqgfadb', 1), ('pandethrypv', 1), ('kirtyywpuo', 1), ('dcksxjeq', 1), ('oldest', 1), ('ckfobaxd', 1), ('wgnejyvt', 1), ('creojvdh', 1), ('ciblyskg', 1), ('belgium', 1), ('callsariving', 1), ('kubiadfffk', 1), ('costumers', 1), ('polish', 1), ('industriekontrollmonitor', 1), ('dhthykts', 1), ('anzuschlie', 1), ('gray', 1), ('alternative', 1), ('daf', 1), ('czech', 1), ('republic', 1), ('chrsddiwds', 1), ('dwdbertfsych', 1), ('tests', 1), ('consigment', 1), ('relased', 1), ('petqkjra', 1), ('ksxchbaf', 1), ('rhquvzfm', 1), ('ydvmibwn', 1), ('eljtsdar', 1), ('donggle', 1), ('fwypxqcs', 1), ('twsqoimy', 1), ('ladies', 1), ('gentlemen', 1), ('bde', 1), ('passowrd', 1), ('kimtc', 1), ('dkklddww', 1), ('lqdwjdwd', 1), ('plesae', 1), ('bag', 1), ('adopter', 1), ('deilvery', 1), ('nwlhcfpa', 1), ('zdlfmthq', 1), ('waitgr', 1), ('chdffong', 1), ('sdfgwong', 1), ('lauredwwden', 1), ('hwffiglhkins', 1), ('kthassi', 1), ('dbkdwwd', 1), ('wdfwsggalleh', 1), ('phvwitud', 1), ('manifest', 1), ('ships', 1), ('wtykmnlg', 1), ('xamesrpfrop', 1), ('zurtxjbd', 1), ('gaotwcfd', 1), ('brthryian', 1), ('kiduhltr', 1), ('ofrdvnux', 1), ('asae', 1), ('yhteijwf', 1), ('llwlfazo', 1), ('dwmnad', 1), ('macwdlmwkey', 1), ('objecthandleonstack', 1), ('retrawbytes', 1), ('getrawdata', 1), ('generatehash', 1), ('hashtype', 1), ('getassemblyinfointernal', 1), ('getassemblyinfo', 1), ('getscriptresourceurlimpl', 1), ('assemblyresourcelists', 1), ('resourcename', 1), ('cultureinfo', 1), ('culture', 1), ('geturlfromname', 1), ('icontrol', 1), ('scriptmanagercontrol', 1), ('geturlinternal', 1), ('geturl', 1), ('registeruniquescripts', 1), ('uniquescripts', 1), ('registerscripts', 1), ('onpageprerendercomplete', 1), ('onprerendercomplete', 1), ('processrequestmain', 1), ('includestagesbeforeasyncpoint', 1), ('includestagesafterasyncpoint', 1), ('waited', 1), ('categories', 1), ('metric', 1), ('flo', 1), ('depletes', 1), ('beyhtcykea', 1), ('corsthroc', 1), ('envoy', 1), ('partir', 1), ('outil', 1), ('cran', 1), ('orvsydzf', 1), ('rbqtpdaz', 1), ('haved', 1), ('knrlepglper', 1), ('reinstate', 1), ('thetadkg', 1), ('reeive', 1), ('tackle', 1), ('sharepont', 1), ('schddklnes', 1), ('dpo', 1), ('walddkrrm', 1), ('mdiwjd', 1), ('wthaldlmdsrop', 1), ('benethrytte', 1), ('cthoursook', 1), ('bplnyedg', 1), ('vobluewg', 1), ('ghost', 1), ('iii', 1), ('dathniel', 1), ('fangtry', 1), ('openorderbook', 1), ('accdb', 1), ('iwifi', 1), ('slimware', 1), ('detachable', 1), ('hopes', 1), ('dkxlpvnr', 1), ('narxcgjh', 1), ('odbcdrivermanagerlibrary', 1), ('rzxfgmcu', 1), ('xprwayoc', 1), ('redirect', 1), ('worddocument', 1), ('edited', 1), ('zeugnis', 1), ('proint', 1), ('eweausbildung', 1), ('jannek', 1), ('ndling', 1), ('extremly', 1), ('possilbe', 1), ('reacts', 1), ('extremely', 1), ('cwivnxuk', 1), ('izmxqfud', 1), ('qkspyrdms', 1), ('feira', 1), ('outubro', 1), ('ongumpdz', 1), ('pjkrfmbc', 1), ('ferguss', 1), ('gustaco', 1), ('filipim', 1), ('requena', 1), ('gustavo', 1), ('streamline', 1), ('vvtathadnda', 1), ('separatelly', 1), ('mokolthrl', 1), ('literate', 1), ('lhmxposv', 1), ('seat', 1), ('liable', 1), ('mwtrouyl', 1), ('joerg', 1), ('passiep', 1), ('jathyrsy', 1), ('ues', 1), ('township', 1), ('pinkow', 1), ('announcement', 1), ('bertes', 1), ('inin', 1), ('noc', 1), ('proactive', 1), ('interactive', 1), ('blast', 1), ('monitored', 1), ('inlet', 1), ('ambient', 1), ('lpas', 1), ('cabane', 1), ('pluytd', 1), ('erthryika', 1), ('plaunyud', 1), ('pxvjczdt', 1), ('kizsjfpq', 1), ('ztax', 1), ('taps', 1), ('henvrkuo', 1), ('nogrfadw', 1), ('instrument', 1), ('sever', 1), ('workcentre', 1), ('rta', 1), ('fduinmtw', 1), ('yofhirjs', 1), ('hixsodl', 1), ('hydluapo', 1), ('qbgclmit', 1), ('eulsvchi', 1), ('rqflkeuc', 1), ('cthryhris', 1), ('kovaddcth', 1), ('ksxjcvze', 1), ('ognyetrp', 1), ('cannnot', 1), ('tiaghry', 1), ('santhuy', 1), ('ldgm', 1), ('bcxfhekz', 1), ('bplfrnis', 1), ('zneyrlhg', 1), ('bfiwanze', 1), ('bettymcdanghtnuell', 1), ('mcdythanbm', 1), ('pvn', 1), ('ajnpuqym', 1), ('gvoncems', 1), ('branding', 1), ('jefguyza', 1), ('mkhwcnes', 1), ('syhtu', 1), ('pozdrsavom', 1), ('precall', 1), ('cmor', 1), ('costing', 1), ('ppc', 1), ('itemization', 1), ('split', 1), ('dedalus', 1), ('vajtodny', 1), ('ldmwqubi', 1), ('sovqeynk', 1), ('lehrjahr', 1), ('cspkeyza', 1), ('bzpyfcki', 1), ('dwight', 1), ('stibo', 1), ('drtawings', 1), ('simply', 1), ('visitors', 1), ('rtbkimey', 1), ('cfsqwtdv', 1), ('xikojdym', 1), ('rgazclmi', 1), ('fuf', 1), ('sny', 1), ('mdvlkbac', 1), ('uhefoqtg', 1), ('plese', 1), ('tempor', 1), ('anmeldefehler', 1), ('dvi', 1), ('syhunil', 1), ('krishnyhda', 1), ('mpvasqwy', 1), ('rotkyeja', 1), ('netzger', 1), ('inkjet', 1), ('imaje', 1), ('surrender', 1), ('verly', 1), ('yotyhga', 1), ('narthdyhy', 1), ('serverprobleme', 1), ('qm', 1), ('anwendbar', 1), ('adapoter', 1), ('unpliugged', 1), ('batter', 1), ('hvzlqthr', 1), ('lthyqzns', 1), ('gordon', 1), ('leach', 1), ('akirtyethsyd', 1), ('vvsthryomaa', 1), ('ned', 1), ('nunber', 1), ('nothrdaj', 1), ('xcircuit', 1), ('suspecting', 1), ('thaybd', 1), ('mhasttdd', 1), ('rohthsit', 1), ('mhdyhtya', 1), ('plvnuxmry', 1), ('theft', 1), ('fpsf', 1), ('pmw', 1), ('bcv', 1), ('nesner', 1), ('dispatched', 1), ('hierf', 1), ('dbif', 1), ('rsql', 1), ('ajlbguzn', 1), ('fxrwivyg', 1), ('ujzhflpn', 1), ('oibnxrvq', 1), ('purartnpn', 1), ('cuzhydjl', 1), ('queretaro', 1), ('unfortunate', 1), ('globaleng', 1), ('gtc', 1), ('sewilrxm', 1), ('cbikymvf', 1), ('rlphwiqn', 1), ('zagvbkro', 1), ('aao', 1), ('melisdfysa', 1), ('rgrtrs', 1), ('ducyua', 1), ('hqn', 1), ('natytse', 1), ('sylyhtsvesuyter', 1), ('aircap', 1), ('idf', 1), ('yhtdon', 1), ('zdyhazula', 1), ('obtaining', 1), ('suthye', 1), ('kinght', 1), ('sntvfpbl', 1), ('vtokgley', 1), ('stuarthsyt', 1), ('onbehalf', 1), ('dxnzkcuh', 1), ('eqdgoxap', 1), ('whilst', 1), ('bernig', 1), ('hcyemunl', 1), ('lnecsgpd', 1), ('mathrv', 1), ('macyhtkey', 1), ('lcvl', 1), ('cuthyunniy', 1), ('efdhrlwv', 1), ('aoxtugzr', 1), ('aedwrpvo', 1), ('qbpafrsx', 1), ('aqritplu', 1), ('beuflorc', 1), ('bnsaqekm', 1), ('xoglfeij', 1), ('vnizrdeb', 1), ('rgtsm', 1), ('filed', 1), ('raid', 1), ('maihtyrhu', 1), ('suniythulkuujmar', 1), ('vvkhyhums', 1), ('dhad', 1), ('wuz', 1), ('yup', 1), ('dhoalycb', 1), ('igdnsjhz', 1), ('awnftgev', 1), ('hint', 1), ('generates', 1), ('positive', 1), ('younger', 1), ('batuhan', 1), ('gueduel', 1), ('szockfpj', 1), ('izohlgcq', 1), ('qdztknml', 1), ('hpcxnyrq', 1), ('djhznybt', 1), ('dyqekzuc', 1), ('urpbyoke', 1), ('vwcbhmds', 1), ('ybxsujwi', 1), ('yzwanorl', 1), ('shaking', 1), ('proms', 1), ('imjwbogq', 1), ('xfizlnap', 1), ('dbednyuarski', 1), ('pgacouel', 1), ('mpgfkxwr', 1), ('horrible', 1), ('mother', 1), ('compute', 1), ('amazonaws', 1), ('standby', 1), ('mcae', 1), ('saravthsyana', 1), ('qgvbalix', 1), ('smehqkyj', 1), ('cesco', 1), ('pound', 1), ('bnoupaki', 1), ('cpeioxdz', 1), ('kambthryes', 1), ('exlbkpoj', 1), ('vrkoqaje', 1), ('jidhewlg', 1), ('jufskody', 1), ('olthyivectr', 1), ('tegdtyyp', 1), ('ethd', 1), ('yhtheehey', 1), ('mithyke', 1), ('tayjuoylor', 1), ('arithel', 1), ('shfsako', 1), ('mullthyed', 1), ('marthhty', 1), ('retiring', 1), ('urgency', 1), ('thsgy', 1), ('todthyd', 1), ('renytrner', 1), ('initialization', 1), ('loader', 1), ('uwjchqor', 1), ('anira', 1), ('budighfl', 1), ('izbxvary', 1), ('initialize', 1), ('quaraintined', 1), ('accsess', 1), ('rcmziuob', 1), ('xhblozqe', 1), ('accses', 1), ('incomming', 1), ('reprint', 1), ('changi', 1), ('lane', 1), ('atleast', 1), ('wsabzycm', 1), ('pbhrmnyz', 1), ('improve', 1), ('automatical', 1), ('transfers', 1), ('automaticaly', 1), ('stransfers', 1), ('dosen', 1), ('aunkpchr', 1), ('qsyvrmjl', 1), ('mates', 1), ('willk', 1), ('rlich', 1), ('spots', 1), ('qhyoiwls', 1), ('uynrhiva', 1), ('xaykwtrf', 1), ('amlswjfr', 1), ('krnzfoct', 1), ('mnjbuedp', 1), ('priter', 1), ('prqmp', 1), ('feathers', 1), ('punches', 1), ('annyhtie', 1), ('zhothu', 1), ('grir', 1), ('ice', 1), ('routes', 1), ('athyndy', 1), ('eartyp', 1), ('discrepancy', 1), ('kmzwjdsb', 1), ('ejgnafcy', 1), ('ebm', 1), ('eemwx', 1), ('winprodnc', 1), ('wunthyder', 1), ('mdghayi', 1), ('redytudy', 1), ('dhec', 1), ('ordinate', 1), ('rwuqydvo', 1), ('anecdfps', 1), ('masters', 1), ('override', 1), ('zlp', 1), ('nett', 1), ('crops', 1), ('edgemaster', 1), ('hone', 1), ('ebpwcfla', 1), ('qoxvpbam', 1), ('zdgnlqkc', 1), ('zfjbpntg', 1), ('kenny', 1), ('pkj', 1), ('prakaythsh', 1), ('kujigalore', 1), ('xzbtcfar', 1), ('gilbrmuyt', 1), ('globalmfg', 1), ('gethyoff', 1), ('schoemerujt', 1), ('mccoyimgs', 1), ('bee', 1), ('actively', 1), ('combxind', 1), ('bvptuqxr', 1), ('revelj', 1), ('einw', 1), ('hlen', 1), ('pennsylvania', 1), ('waynesboro', 1), ('stoppage', 1), ('dialed', 1), ('clue', 1), ('companyssecure', 1), ('intellectual', 1), ('lryturhyyth', 1), ('ryhunan', 1), ('turns', 1), ('ssp', 1), ('tmunkaiv', 1), ('nlgkmpaq', 1), ('outrnkca', 1), ('druker', 1), ('xahuklgm', 1), ('dqvkfjlb', 1), ('programdntyms', 1), ('ieas', 1), ('logical', 1), ('begun', 1), ('miioperatordev', 1), ('miioperatorqa', 1), ('barrtyh', 1), ('unsubscribe', 1), ('quarterly', 1), ('gurublxkizmh', 1), ('nvodbrfluppasadabasavaraj', 1), ('qaohugxw', 1), ('briefkopf', 1), ('axcbfuqo', 1), ('yiagubvh', 1), ('simcard', 1), ('holder', 1), ('prithivrtyaj', 1), ('vvyhtyumasp', 1), ('gmwdvrou', 1), ('aupnvems', 1), ('rtgdcoun', 1), ('pngufmvq', 1), ('efodqiuh', 1), ('tpfnzkli', 1), ('rcwpvkyb', 1), ('exgjscql', 1), ('bhty', 1), ('thvnfs', 1), ('anyhusppa', 1), ('wcrbmgon', 1), ('kcudbnrw', 1), ('faxen', 1), ('schl', 1), ('fehl', 1), ('faxing', 1), ('xbsckemt', 1), ('durnfyxb', 1), ('appricatehub', 1), ('hie', 1), ('visited', 1), ('problme', 1), ('pauhtul', 1), ('phillyhuip', 1), ('wsboedtj', 1), ('yvlswgxb', 1), ('errormessage', 1), ('female', 1), ('accound', 1), ('mthyn', 1), ('xiyhtu', 1), ('blmvcuya', 1), ('heklonfc', 1), ('jtyhnifer', 1), ('luntu', 1), ('urfdkvei', 1), ('bfiulzto', 1), ('ayhtrvin', 1), ('yzhao', 1), ('oqmiabtv', 1), ('spridlbm', 1), ('mathgie', 1), ('ztyhng', 1), ('ryajizsq', 1), ('roezpsav', 1), ('alvrhn', 1), ('twhyang', 1), ('evlburfd', 1), ('xbusnyam', 1), ('prathryep', 1), ('margin', 1), ('closure', 1), ('jqpvitdw', 1), ('vitsrceq', 1), ('lilp', 1), ('workig', 1), ('mini', 1), ('ckswypji', 1), ('vrucgqna', 1), ('motorola', 1), ('moto', 1), ('req', 1), ('yeyhtung', 1), ('kimthy', 1), ('vvkuimtyu', 1), ('sudghhahjkkarreddy', 1), ('datateam', 1), ('englehart', 1), ('toolpasswordmanager', 1), ('mthyike', 1), ('voyyhuek', 1), ('aza', 1), ('ze', 1), ('outlooke', 1), ('fuydxemo', 1), ('fntmbpla', 1), ('tyyhtuler', 1), ('hiyhllt', 1), ('vnjwsadx', 1), ('iltywzjm', 1), ('fidleyhtjp', 1), ('bright', 1), ('warms', 1), ('lacl', 1), ('hfmp', 1), ('cannont', 1), ('frustrating', 1), ('mysterious', 1), ('rckfthy', 1), ('administra', 1), ('geoyhtrge', 1), ('wuryhtudack', 1), ('tracked', 1), ('oxlqvika', 1), ('justrgun', 1), ('rohit', 1), ('wjdatzyv', 1), ('bhkoldjv', 1), ('ncoileu', 1), ('boeyhthm', 1), ('dpyvjcxr', 1), ('lfml', 1), ('itjzudor', 1), ('ybtmorxp', 1), ('nderen', 1), ('nderildi', 1), ('ekim', 1), ('sal', 1), ('kime', 1), ('konu', 1), ('hiyhugins', 1), ('haiwei', 1), ('liang', 1), ('patience', 1), ('intepmov', 1), ('imjukbqhing', 1), ('qkspyrdm', 1), ('hiapth', 1), ('jk', 1), ('unsupported', 1), ('exchangeable', 1), ('twdq', 1), ('cruz', 1), ('qavdrpfu', 1), ('shopflor', 1), ('replicates', 1), ('rudiment', 1), ('latest', 1), ('atttached', 1), ('lvdyrqfc', 1), ('pfnmjsok', 1), ('webyutelc', 1), ('ztswnaom', 1), ('lrapiwex', 1), ('othyoiz', 1), ('alfa', 1), ('rollo', 1), ('matic', 1), ('franhtyuj', 1), ('cksetzten', 1), ('brook', 1), ('hurry', 1), ('notic', 1), ('hourglass', 1), ('void', 1), ('jay', 1), ('airwaybill', 1), ('axis', 1), ('dizquolf', 1), ('hlykecxa', 1), ('wkpnlvts', 1), ('oumeaxcz', 1), ('heidi', 1), ('documented', 1), ('unacceptable', 1), ('strategic', 1), ('mazurjw', 1), ('xbfcitlh', 1), ('ntulmcpq', 1), ('theydbar', 1), ('brrgtyanthet', 1), ('perry', 1), ('buyoipdj', 1), ('fceymwtz', 1), ('toll', 1), ('xwelumfz', 1), ('louis', 1), ('ensured', 1), ('ipv', 1), ('initially', 1), ('denghyrt', 1), ('fdmaluyo', 1), ('kpr', 1), ('gdcuhzqw', 1), ('coppthsy', 1), ('humthyphk', 1), ('pinter', 1), ('yrjekzqv', 1), ('automatci', 1), ('kslhobgj', 1), ('cyhvefna', 1), ('yorgbnpa', 1), ('ndigthpj', 1), ('sitepages', 1), ('ipglathybel', 1), ('appair', 1), ('imp', 1), ('vvhthyoffc', 1), ('vvbthryhn', 1), ('vvwtyeidt', 1), ('vvftgors', 1), ('vvnergtubj', 1), ('vvthygschj', 1), ('bfr', 1), ('prtoplant', 1), ('kubyhtuaa', 1), ('encounters', 1), ('timeouts', 1), ('plqbesvo', 1), ('uopaexic', 1), ('dabhrujir', 1), ('thy', 1), ('ryljar', 1), ('fts', 1), ('troubling', 1), ('remedial', 1), ('busienss', 1), ('umykjweg', 1), ('jpwrfuhk', 1), ('krugew', 1), ('bugs', 1), ('terminates', 1), ('arcgonvy', 1), ('jam', 1), ('vroxskje', 1), ('bixtmefd', 1), ('fdd', 1), ('vithrkas', 1), ('xiwegtas', 1), ('ygrfbzon', 1), ('preview', 1), ('qwvpgayb', 1), ('amniujsh', 1), ('xdvwitpm', 1), ('zscxqdho', 1), ('bathylardb', 1), ('ashtusis', 1), ('pyhuule', 1), ('phufsav', 1), ('yhru', 1), ('manyhsu', 1), ('ayujdm', 1), ('linz', 1), ('ppm', 1), ('reviews', 1), ('micheyi', 1), ('gyhus', 1), ('logoff', 1), ('hecked', 1), ('wanayht', 1), ('xztveoqs', 1), ('zyrnqiav', 1), ('lhqksbd', 1), ('rjeyfxlg', 1), ('ltfskygw', 1), ('smdbqnef', 1), ('notepad', 1), ('oyunatye', 1), ('ngjztqai', 1), ('xqjzpvru', 1), ('dpozkmie', 1), ('vjuybcwz', 1), ('proc', 1), ('counts', 1), ('dow', 1), ('ilhcgoqf', 1), ('xlibynvc', 1), ('allocate', 1), ('biyhll', 1), ('kthyarg', 1), ('smhepfdn', 1), ('aypgzieh', 1), ('resset', 1), ('hgudmrta', 1), ('vidzoqfl', 1), ('dene', 1), ('duane', 1), ('captured', 1), ('orelikon', 1), ('balzers', 1), ('feb', 1), ('lij', 1), ('syte', 1), ('jinxyhdi', 1), ('luji', 1), ('vvlixthy', 1), ('jacyhky', 1), ('liuhyt', 1), ('weqocbnu', 1), ('eoixcpvy', 1), ('caoryhuq', 1), ('vhjkdqop', 1), ('tkhafgrc', 1), ('qhjkxoyw', 1), ('lgiovknd', 1), ('vythytalyst', 1), ('prerequisites', 1), ('filler', 1), ('lra', 1), ('remedy', 1), ('subtask', 1), ('decommissioning', 1), ('mcgee', 1), ('subbathykrisyuhnyrt', 1), ('shhuivashankar', 1), ('easyterritory', 1), ('builder', 1), ('resend', 1), ('resign', 1), ('plase', 1), ('unlook', 1), ('chrithysgd', 1), ('pstn', 1), ('converion', 1), ('exiting', 1), ('navdgtya', 1), ('kuhyakose', 1), ('zevpkogu', 1), ('khfgharla', 1), ('fqdgotvx', 1), ('septemer', 1), ('zcp', 1), ('mandt', 1), ('sum', 1), ('toengineering', 1), ('solman', 1), ('theses', 1), ('charms', 1), ('cal', 1), ('pointing', 1), ('reichlhdyl', 1), ('hellej', 1), ('toolbar', 1), ('ctxjsolz', 1), ('kq', 1), ('carahcters', 1), ('pjl', 1), ('persits', 1), ('nia', 1), ('ojhiaubp', 1), ('lovgirtm', 1), ('paint', 1), ('winzip', 1), ('keyboard', 1), ('stocktransfer', 1), ('mgahlpwx', 1), ('jwtfpaxh', 1), ('webportal', 1), ('corectly', 1), ('treatment', 1), ('krilnmcs', 1), ('chpkeyqb', 1), ('tryed', 1), ('alook', 1), ('sless', 1), ('comfortable', 1), ('imwveudk', 1), ('mykcourx', 1), ('plans', 1), ('disrupted', 1), ('downtime', 1), ('tcbonyes', 1), ('gpfacron', 1), ('bnthygl', 1), ('pdu', 1), ('hdjm', 1), ('bhrtty', 1), ('pdlc', 1), ('closes', 1), ('ugyothfz', 1), ('ambals', 1), ('oclock', 1), ('ttemplates', 1), ('redo', 1), ('dinthyesh', 1), ('ethyxekirty', 1), ('etyhumpdil', 1), ('exekirty', 1), ('empkirty', 1), ('blanks', 1), ('nagdyiyst', 1), ('acount', 1), ('sykpe', 1), ('belwo', 1), ('delthybid', 1), ('gjisfonb', 1), ('odwfhmze', 1), ('hohlbfgtu', 1), ('provision', 1), ('gflewxmn', 1), ('qnxhoryg', 1), ('tghkris', 1), ('wickhamtf', 1), ('tagen', 1), ('wichtige', 1), ('dokumente', 1), ('speicherort', 1), ('kopieren', 1), ('ltig', 1), ('linking', 1), ('fvwhyenp', 1), ('juxitrbf', 1), ('vebckrgh', 1), ('computing', 1), ('priced', 1), ('suppose', 1), ('cancelation', 1), ('zmc', 1), ('antjuyhony', 1), ('myportal', 1), ('sre', 1), ('issie', 1), ('scrolled', 1), ('rfa', 1), ('analyseger', 1), ('laboratory', 1), ('srqyfjxz', 1), ('lnagtjzi', 1), ('cqargubj', 1), ('krdxbfqh', 1), ('vvgoythttu', 1), ('pgrvqtdo', 1), ('hgpymwxb', 1), ('voethrylke', 1), ('malefunktion', 1), ('aiiw', 1), ('doctypes', 1), ('rayhtuorv', 1), ('drhykngx', 1), ('oxviygdb', 1), ('shryresh', 1), ('rdyrty', 1), ('yzeakbrl', 1), ('npxbkojl', 1), ('ldg', 1), ('misuhet', 1), ('milsytr', 1), ('shippping', 1), ('hxgayczekurtyar', 1), ('housekeep', 1), ('feasibility', 1), ('gpfsfs', 1), ('keshyslsj', 1), ('devt', 1), ('toolscal', 1), ('approx', 1), ('gneral', 1), ('infomation', 1), ('staerted', 1), ('liefert', 1), ('corner', 1), ('bwhrchgr', 1), ('townhall', 1), ('preparation', 1), ('rtro', 1), ('nicdhylas', 1), ('hidhys', 1), ('ddwjm', 1), ('lots', 1), ('ascending', 1), ('alphabets', 1), ('nightmare', 1), ('alloy', 1), ('halo', 1), ('haajksjp', 1), ('myth', 1), ('franjuz', 1), ('urbghty', 1), ('bowtniuy', 1), ('afgdmesz', 1), ('wharehouse', 1), ('batia', 1), ('shloosh', 1), ('huji', 1), ('uhytry', 1), ('yayuel', 1), ('sayatgr', 1), ('sdilxrfk', 1), ('jhdythua', 1), ('htayhil', 1), ('udmbwocs', 1), ('kegsjdva', 1), ('lgeuniqf', 1), ('ijsnyxgf', 1), ('diehm', 1), ('atdclmyi', 1), ('wqxzaysu', 1), ('websty', 1), ('tooled', 1), ('clint', 1), ('cke', 1), ('disturbance', 1), ('suppor', 1), ('timeclock', 1), ('clhr', 1), ('toollant', 1), ('elituyt', 1), ('byhtu', 1), ('kicks', 1), ('potentional', 1), ('marc', 1), ('xmjwanes', 1), ('naffwflie', 1), ('hyhtard', 1), ('kronsnwdg', 1), ('agthynew', 1), ('laijuttryhr', 1), ('owtlmpuv', 1), ('oicrjsfh', 1), ('donwloaded', 1), ('libraries', 1), ('wfzgituk', 1), ('rxvqzopt', 1), ('corporateaccounting', 1), ('misc', 1), ('zlftrkpq', 1), ('liz', 1), ('domasky', 1), ('trupthyti', 1), ('royhtub', 1), ('haujtimpton', 1), ('spillage', 1), ('pfner', 1), ('hoepftyhum', 1), ('fore', 1), ('qzixratf', 1), ('wrygjncl', 1), ('poruxnwb', 1), ('yfaqhceo', 1), ('hwbukcsm', 1), ('hwobikcv', 1), ('fusion', 1), ('middleware', 1), ('szewiguc', 1), ('nvajphfm', 1), ('renewing', 1), ('upadate', 1), ('acccount', 1), ('reissued', 1), ('outs', 1), ('annoyed', 1), ('hesitate', 1), ('dbryhtuown', 1), ('definition', 1), ('charatcher', 1), ('indications', 1), ('doxiqkws', 1), ('uvrzcqmf', 1), ('apokrfjv', 1), ('mdiepcul', 1), ('donnathyr', 1), ('welling', 1), ('distrtgoyed', 1), ('pande', 1), ('bussy', 1), ('icyxtqej', 1), ('lqsjrgzt', 1), ('hyeonthygwon', 1), ('lethre', 1), ('verggermany', 1), ('favorite', 1), ('gurts', 1), ('lrupiepens', 1), ('responsibilities', 1), ('gus', 1), ('correspondences', 1), ('zwrypjqv', 1), ('bathishry', 1), ('fashion', 1), ('schnafk', 1), ('jaytya', 1), ('deactivation', 1), ('fue', 1), ('ipydfcqo', 1), ('kdxsquzn', 1), ('tgafnyzb', 1), ('hnevrcuj', 1), ('dauert', 1), ('extrem', 1), ('lange', 1), ('manchmal', 1), ('copiers', 1), ('minolta', 1), ('peathryucoj', 1), ('ahujajtyhur', 1), ('ezwcpqrh', 1), ('bnwqaglk', 1), ('emal', 1), ('outrlook', 1), ('gard', 1), ('yathryu', 1), ('asistance', 1), ('highly', 1), ('spends', 1), ('readd', 1), ('kothyherr', 1), ('insists', 1), ('eigentlich', 1), ('jedoch', 1), ('keinerlei', 1), ('erinnerungsinformationen', 1), ('komme', 1), ('deswegen', 1), ('gedr', 1), ('offene', 1), ('tun', 1), ('prognose', 1), ('salesperson', 1), ('alparslanthyr', 1), ('sagfhoshgzpkmilu', 1), ('skwbuvjyheelavant', 1), ('fahdlecz', 1), ('ubcszaohygexqab', 1), ('cygexqab', 1), ('culixwse', 1), ('pmrvxbnjhivaramdntyaiah', 1), ('hpeknoam', 1), ('yrfowmva', 1), ('wake', 1), ('powersave', 1), ('kirtyling', 1), ('worry', 1), ('virtualcenter', 1), ('trn', 1), ('hdt', 1), ('kmc', 1), ('fehlende', 1), ('monatswechsel', 1), ('ausgabe', 1), ('ausgeschaltet', 1), ('unregistered', 1), ('moryctrhbkm', 1), ('plvnuxmrnoj', 1), ('deppt', 1), ('performances', 1), ('wethruiberg', 1), ('relocated', 1), ('vlpfgjyz', 1), ('dvzrfsbo', 1), ('dartnl', 1), ('porwrloisky', 1), ('poloidgthyl', 1), ('danl', 1), ('poloisky', 1), ('federal', 1), ('wip', 1), ('vvwhtyuy', 1), ('mikdhyu', 1), ('lihy', 1), ('rolcgqhx', 1), ('ehndjmlv', 1), ('marty', 1), ('nevins', 1), ('nevinmw', 1), ('fallowing', 1), ('koqntham', 1), ('sqiuctfl', 1), ('jochgthen', 1), ('grethyg', 1), ('omiwzbue', 1), ('auvolfhp', 1), ('tiresome', 1), ('ldvl', 1), ('peoples', 1), ('ccfterguss', 1), ('wire', 1), ('rick', 1), ('orelli', 1), ('biintll', 1), ('tujutnis', 1), ('convenience', 1), ('participant', 1), ('detetection', 1), ('duplicates', 1), ('dup', 1), ('taylthyuoaj', 1), ('manuf', 1), ('early', 1), ('ylfqrzxg', 1), ('jmakitug', 1), ('utilizing', 1), ('labeling', 1), ('hdw', 1), ('vamthrsee', 1), ('jmrukcfq', 1), ('rdyuxomp', 1), ('jmusidzr', 1), ('sratdeol', 1), ('mast', 1), ('func', 1), ('lillanna', 1), ('ujxvrlzg', 1), ('pkaegicn', 1), ('replay', 1), ('tahat', 1), ('determinated', 1), ('chanhes', 1), ('pse', 1), ('gqcyomwf', 1), ('opjaiwcu', 1), ('dqowbefk', 1), ('prgxwzco', 1), ('erpsys', 1), ('bothering', 1), ('arrived', 1), ('hpmjtgik', 1), ('blrmfvyh', 1), ('vvamirsdwnp', 1), ('kuhyndan', 1), ('qjgnkhso', 1), ('vahgolwx', 1), ('upto', 1), ('december', 1), ('billed', 1), ('ziv', 1), ('vvsallz', 1), ('upservice', 1), ('zifujpvr', 1), ('makiosjc', 1), ('kxapdhnm', 1), ('undocked', 1), ('arranging', 1), ('attrachment', 1), ('replies', 1), ('qux', 1), ('qiwthyang', 1), ('oemcold', 1), ('nahytu', 1), ('wxdvjoct', 1), ('ckxwtoam', 1), ('errir', 1), ('toolting', 1), ('yhe', 1), ('sathyrui', 1), ('shiragavi', 1), ('inspiron', 1), ('replied', 1), ('zbpdhxvk', 1), ('nowxjztk', 1), ('aghynilthykurtyar', 1), ('gorlithy', 1), ('wfgtyill', 1), ('hannathry', 1), ('widespread', 1), ('spelling', 1), ('fre', 1), ('etvzjmhx', 1), ('mint', 1), ('rubber', 1), ('separating', 1), ('reattached', 1), ('super', 1), ('glue', 1), ('vpv', 1), ('memotech', 1), ('ken', 1), ('mstipsolutions', 1), ('constance', 1), ('wgtyillsford', 1), ('euromote', 1), ('personel', 1), ('martha', 1), ('began', 1), ('kyefsrjc', 1), ('eadmpzcn', 1), ('ejected', 1), ('mityhuch', 1), ('ervuyin', 1), ('rlmbxeso', 1), ('ulmkxdfi', 1), ('sds', 1), ('lives', 1), ('vvblothryor', 1), ('vvbloor', 1), ('vidya', 1), ('lbdl', 1), ('ons', 1), ('dudekm', 1), ('signs', 1), ('vvblorytor', 1), ('qvhixotw', 1), ('rxutkyha', 1), ('ktr', 1), ('tro', 1), ('vvparthyrra', 1), ('spilled', 1), ('vvnewthey', 1), ('reoccurring', 1), ('scam', 1), ('crime', 1), ('cybercrime', 1), ('measures', 1), ('mandated', 1), ('suspended', 1), ('outloook', 1), ('invites', 1), ('filling', 1), ('resending', 1), ('confusion', 1), ('gleich', 1), ('workcenter', 1), ('surthryr', 1), ('stragiht', 1), ('surveying', 1), ('jhr', 1), ('addiitional', 1), ('wvdgopybrumugam', 1), ('mgyhnsat', 1), ('aswubnyd', 1), ('dargthya', 1), ('jayartheuy', 1), ('tcqpyuei', 1), ('becoxvqkahadikar', 1), ('remarkhtys', 1), ('csm', 1), ('toolent', 1), ('fanuc', 1), ('ladder', 1), ('dwg', 1), ('verz', 1), ('gerung', 1), ('pinged', 1), ('messecke', 1), ('lmwohkbd', 1), ('ucziatex', 1), ('nmtszgbr', 1), ('wnthvqgm', 1), ('elcpduzg', 1), ('eujpstxi', 1), ('rudolf', 1), ('kennmetal', 1), ('sticks', 1), ('broke', 1), ('transfereinheit', 1), ('refrence', 1), ('tkjypfze', 1), ('jxompytk', 1), ('synchronizer', 1), ('developer', 1), ('hxgayczecp', 1), ('jmvnxtgc', 1), ('kvhxntqp', 1), ('encountring', 1), ('fkuqjwit', 1), ('jgcsaqzi', 1), ('channelerror', 1), ('connectionreset', 1), ('kate', 1), ('mhikeucr', 1), ('quaixnbe', 1), ('negatively', 1), ('tami', 1), ('veiw', 1), ('storm', 1), ('storms', 1), ('lwgytuxq', 1), ('qspdztiw', 1), ('brandeerthy', 1), ('grayed', 1), ('sheehy', 1), ('oabdfcnk', 1), ('xeuhkoqa', 1), ('carolutyu', 1), ('magyarics', 1), ('prepull', 1), ('usanet', 1), ('vvmagyc', 1), ('hzmxwdrs', 1), ('tcbjyqps', 1), ('pcjtisrv', 1), ('havyuwds', 1), ('vacations', 1), ('ranjhruy', 1), ('ergebnis', 1), ('evakuierungs', 1), ('bung', 1), ('sintering', 1), ('toolrently', 1), ('anytime', 1), ('ltsqkane', 1), ('ycgwexdf', 1), ('xamcuong', 1), ('maliowbg', 1), ('cltnwazh', 1), ('nmyzehow', 1), ('gnlcripo', 1), ('ayuda', 1), ('conseciones', 1), ('marfhtyios', 1), ('talked', 1), ('hatryu', 1), ('bau', 1), ('fuyidkbv', 1), ('unbale', 1), ('ccftv', 1), ('turnover', 1), ('satisfy', 1), ('och', 1), ('med', 1), ('nlig', 1), ('lsning', 1), ('brxaqlwn', 1), ('auzroqes', 1), ('kantthyhn', 1), ('uninstallation', 1), ('adwares', 1), ('needing', 1), ('verena', 1), ('irrecular', 1), ('lax', 1), ('samacocuntname', 1), ('bihrtyull', 1), ('thadhylman', 1), ('janhetgdyu', 1), ('libarary', 1), ('spl', 1), ('payt', 1), ('aidw', 1), ('petljhxi', 1), ('bocxgins', 1), ('microphones', 1), ('engl', 1), ('bestellnumer', 1), ('wollschl', 1), ('immediatly', 1), ('bankruped', 1), ('silent', 1), ('bzwrchnd', 1), ('ysfiwvmo', 1), ('vvrassyhrt', 1), ('irgsthy', 1), ('daisy', 1), ('huang', 1), ('relese', 1), ('nwzhlktu', 1), ('plktredg', 1), ('grbhybrdg', 1), ('anticipation', 1), ('oqlcdvwi', 1), ('pulcqkzo', 1), ('mobilen', 1), ('inhalte', 1), ('vorr', 1), ('quarant', 1), ('gestellt', 1), ('aktion', 1), ('durchf', 1), ('hren', 1), ('inhalt', 1), ('automatisch', 1), ('heruntergeladen', 1), ('sobald', 1), ('fvqfj', 1), ('kdgu', 1), ('kqelgbis', 1), ('stiarhlu', 1), ('resets', 1), ('productio', 1), ('jamhdtyes', 1), ('kinhytudel', 1), ('ug', 1), ('manuel', 1), ('recommits', 1), ('prjuysva', 1), ('vpbudksy', 1), ('naveuythen', 1), ('dyhtruutt', 1), ('clearance', 1), ('apper', 1), ('lathe', 1), ('shortcuts', 1), ('wkqjcfgy', 1), ('vsknlfri', 1), ('metalworking', 1), ('zm', 1), ('stope', 1), ('series', 1), ('organizer', 1), ('mehrota', 1), ('nextgen', 1), ('ohljvzpn', 1), ('phwdxqev', 1), ('ocsnugeh', 1), ('ksvlowjd', 1), ('ptyzxscl', 1), ('ahmbnsoi', 1), ('aditya', 1), ('choragudi', 1), ('gotbumak', 1), ('ymdqokfp', 1), ('ruy', 1), ('frota', 1), ('estaxpnz', 1), ('mqhrvjkd', 1), ('ufiatosg', 1), ('ynlqrebs', 1), ('hwfoqjdu', 1), ('harman', 1), ('yijgokrn', 1), ('frmyejbx', 1), ('weclfnhx', 1), ('tgzbklec', 1), ('egnwtvch', 1), ('raouf', 1), ('benamor', 1), ('lpnzjimdghtyy', 1), ('mwtvondq', 1), ('nqepkugo', 1), ('knqmscrw', 1), ('sdtoezjb', 1), ('mccoy', 1), ('suspects', 1), ('arranged', 1), ('proces', 1), ('cest', 1), ('indra', 1), ('vvrajai', 1), ('indrakurtyar', 1), ('rajanna', 1), ('zcae', 1), ('curr', 1), ('karghyuens', 1), ('newest', 1), ('edv', 1), ('wurdack', 1), ('steinich', 1), ('druckauftrag', 1), ('moblews', 1), ('mobley', 1), ('ebs', 1), ('thick', 1), ('reddakv', 1), ('delviery', 1), ('quanttiy', 1), ('tootal', 1), ('azyfsrqh', 1), ('wkavqigu', 1), ('kgueyiwp', 1), ('cjlonvme', 1), ('irj', 1), ('attribudes', 1), ('ppoma', 1), ('ppb', 1), ('auslieferbereich', 1), ('invoice', 1), ('vvgraec', 1), ('ulrike', 1), ('mann', 1), ('countersinking', 1), ('wehlauer', 1), ('cxltnjuk', 1), ('hkdefraw', 1), ('xfbc', 1), ('eawn', 1), ('eeo', 1), ('ecatel', 1), ('foreseemaliciousprobability', 1), ('dolder', 1), ('vendorsigid', 1), ('negativeevaluationthreshold', 1), ('positiveevaluationthreshold', 1), ('modelversion', 1), ('classifiertype', 1), ('naivebayes', 1), ('annotatorlist', 1), ('nb', 1), ('zpononpo', 1), ('zzsdspc', 1), ('realised', 1), ('buckets', 1), ('sit', 1), ('reduce', 1), ('droracle', 1), ('unix', 1), ('sudo', 1), ('trayton', 1), ('neal', 1), ('snyder', 1), ('gptmrqzu', 1), ('muiqteyf', 1), ('griener', 1), ('nmqgrkex', 1), ('ldeizfrm', 1), ('woman', 1), ('koburvmc', 1), ('jwzlebap', 1), ('rie', 1), ('netperfmon', 1), ('executed', 1), ('noi', 1), ('prtsg', 1), ('reisenkostenabrechnung', 1), ('anmeldungen', 1), ('ruenzm', 1), ('deloro', 1), ('bers', 1), ('suchfunkton', 1), ('murakt', 1), ('protel', 1), ('qgrbdnoc', 1), ('dgupnhxv', 1), ('vvkujup', 1), ('sae', 1), ('prodcution', 1), ('yae', 1), ('smitrtgcj', 1), ('soplant', 1), ('xqtldrcs', 1), ('ojgrpafb', 1), ('logfile', 1), ('glf', 1), ('owe', 1), ('dont', 1), ('rechecking', 1), ('migrate', 1), ('srvlavpwdrprd', 1), ('vkm', 1), ('nihtykki', 1), ('mcgyuouald', 1), ('johhdyanna', 1), ('uzvsnlbd', 1), ('dfgtyon', 1), ('stasrty', 1), ('vsbtygin', 1), ('oufhtbas', 1), ('helical', 1), ('zke', 1), ('czsmnbdi', 1), ('ispdhfer', 1), ('zfiu', 1), ('hardkopy', 1), ('lxv', 1), ('lxvunpiz', 1), ('passsw', 1), ('almrgtyeiba', 1), ('gvtbduyf', 1), ('gdblxiva', 1), ('lisfgta', 1), ('geitrhybler', 1), ('gei', 1), ('ler', 1), ('vvkertgipn', 1), ('foulgnmdia', 1), ('pgsqwrumh', 1), ('projekt', 1), ('eingabe', 1), ('keyhuerthi', 1), ('vtyr', 1), ('reddatrhykv', 1), ('sox', 1), ('assessments', 1), ('priflhtret', 1), ('quaterly', 1), ('leasing', 1), ('oppressors', 1), ('kfdyzexr', 1), ('hnbetvfk', 1), ('uarnkqps', 1), ('gufcjxma', 1), ('divide', 1), ('dividing', 1), ('sayg', 1), ('lar', 1), ('zla', 1), ('cnw', 1), ('grays', 1), ('grauschleiher', 1), ('anrgtdy', 1), ('bofffgtyin', 1), ('dubpgacz', 1), ('kjzhilng', 1), ('college', 1), ('fort', 1), ('connetction', 1), ('defence', 1), ('confused', 1), ('deducted', 1), ('incl', 1), ('separated', 1), ('amunt', 1), ('minus', 1), ('paymant', 1), ('mandgtryjuth', 1), ('onbugvhi', 1), ('vzjfgckt', 1), ('phfduvwl', 1), ('yqnaucep', 1), ('zhudrs', 1), ('otpkzifh', 1), ('gywinoml', 1), ('exceed', 1), ('warrrtyen', 1), ('wmybrona', 1), ('qvwhpamb', 1), ('inconvenient', 1), ('occasions', 1), ('transferring', 1), ('lowe', 1), ('robhyertyjs', 1), ('scthyott', 1), ('lortwe', 1), ('trup', 1), ('recode', 1), ('piper', 1), ('southeast', 1), ('toolher', 1), ('kindest', 1), ('authenication', 1), ('consult', 1), ('zarthyc', 1), ('mithycs', 1), ('ockwafib', 1), ('wftboqry', 1), ('pomjgvte', 1), ('goswvnci', 1), ('telef', 1), ('nica', 1), ('deducting', 1), ('subtract', 1), ('rzonkfua', 1), ('yidvloun', 1), ('behaving', 1), ('resume', 1), ('prst', 1), ('perfectly', 1), ('behaved', 1), ('todfrm', 1), ('brgyake', 1), ('zcudbnyq', 1), ('rdxzgpej', 1), ('expecting', 1), ('dreaming', 1), ('pfgia', 1), ('erin', 1), ('usernames', 1), ('edclhpkf', 1), ('ahjklpxm', 1), ('mcelrnr', 1), ('sanddry', 1), ('mcgfrtann', 1), ('mcgatnsl', 1), ('restrictions', 1), ('ergtyic', 1), ('wrtyvis', 1), ('unreadable', 1), ('pitcure', 1), ('invlaid', 1), ('infact', 1), ('znxcupyi', 1), ('bhrwyxgu', 1), ('gofmxlun', 1), ('kxcfrobq', 1), ('micthle', 1), ('conneciton', 1), ('letyenm', 1), ('pwc', 1), ('dvsrepro', 1), ('zrfc', 1), ('kthvr', 1), ('sertce', 1), ('literature', 1), ('exporting', 1), ('pgid', 1), ('rpbdvgoy', 1), ('hxasnzjc', 1), ('structure', 1), ('achim', 1), ('grey', 1), ('tranpertation', 1), ('upate', 1), ('qyidkvap', 1), ('cxnfdjpk', 1), ('passoword', 1), ('qekdgaim', 1), ('wagshrzl', 1), ('ukasz', 1), ('kutnik', 1), ('hey', 1), ('mutoralkv', 1), ('mod', 1), ('chipset', 1), ('motherbankrd', 1), ('esg', 1), ('menue', 1), ('cart', 1), ('okycwstu', 1), ('tvrnbgfs', 1), ('bundles', 1), ('bundle', 1), ('crt', 1), ('eggert', 1), ('karl', 1), ('bnmdslzh', 1), ('qyinrmaf', 1), ('customet', 1), ('ytwmgpbk', 1), ('cpawsihk', 1), ('edmlx', 1), ('efyumrls', 1), ('gqjcbufx', 1), ('tu', 1), ('melerowicz', 1), ('melthryerj', 1), ('sugisdfy', 1), ('woxrljif', 1), ('qymrszdk', 1), ('dkinobsv', 1), ('wymgzcrh', 1), ('uy', 1), ('sujitra', 1), ('wagfrtneh', 1), ('viotto', 1), ('ldiw', 1), ('tzornbldf', 1), ('passwprd', 1), ('crete', 1), ('jost', 1), ('mynfoicj', 1), ('pirces', 1), ('ckmeldung', 1), ('thiw', 1), ('aoea', 1), ('uthagtpgc', 1), ('geetha', 1), ('thrice', 1), ('bestell', 1), ('navigation', 1), ('angef', 1), ('yc', 1), ('zc', 1), ('erpbasis', 1), ('cedroapx', 1), ('blsktzgq', 1), ('radyhthika', 1), ('chnbghyg', 1), ('hertel', 1), ('katfrthy', 1), ('cighyillo', 1), ('kathleen', 1), ('cirillo', 1), ('diagnose', 1), ('registrar', 1), ('hybrid', 1), ('standalone', 1), ('customize', 1), ('bkmk', 1), ('spfrecords', 1), ('unauthenticated', 1), ('dmarc', 1), ('hop', 1), ('mapi', 1), ('cipher', 1), ('ecdhe', 1), ('headers', 1), ('dkim', 1), ('bh', 1), ('khj', 1), ('sgrkpfsbsxqqzvxrpkmbumy', 1), ('bgcvme', 1), ('uxr', 1), ('jlaemyl', 1), ('oakf', 1), ('tpv', 1), ('vdn', 1), ('wyalrvmwpvlt', 1), ('leelrpzts', 1), ('atf', 1), ('lwqaiwmpjkyor', 1), ('nks', 1), ('ricx', 1), ('tqctyws', 1), ('prvdutl', 1), ('ghty', 1), ('bfc', 1), ('uvw', 1), ('mapgzzbtnqsozevisyen', 1), ('tqkq', 1), ('mumxccbhz', 1), ('sun', 1), ('adiu', 1), ('lhl', 1), ('ozbuesgkwoqwhf', 1), ('waakvcg', 1), ('efc', 1), ('tnef', 1), ('correlator', 1), ('mailfrom', 1), ('messagesentrepresentingtype', 1), ('meetingforward', 1), ('originating', 1), ('zwkfzfqhveyrfplvnz', 1), ('nkjsmck', 1), ('qwehil', 1), ('qgumjnxnqprprqkdhi', 1), ('rvjj', 1), ('uc', 1), ('nejqa', 1), ('fzgtz', 1), ('jij', 1), ('acwuv', 1), ('puzdunk', 1), ('hjwjqwvmyqk', 1), ('tovjk', 1), ('vrsw', 1), ('nbmikos', 1), ('pagfw', 1), ('tlikizopxoin', 1), ('nwzgqwxh', 1), ('dfht', 1), ('kz', 1), ('mql', 1), ('pajmwsrgslv', 1), ('qwosb', 1), ('oqmwgxvfmlsw', 1), ('vgi', 1), ('xwksid', 1), ('qzzitpqy', 1), ('dri', 1), ('ottgk', 1), ('xzayeeyzavg', 1), ('eudngmojgvhdtnqwuio', 1), ('ybnp', 1), ('xet', 1), ('grz', 1), ('dsb', 1), ('hkcy', 1), ('iuez', 1), ('nrs', 1), ('wzh', 1), ('jzgmtanpptqedo', 1), ('yq', 1), ('yep', 1), ('pocqfthezol', 1), ('pfhxzkwojpdnpwkzpomtdksxjgwu', 1), ('zrcjk', 1), ('gbfmvjr', 1), ('gvlyai', 1), ('kcekgjrnwpozuhpv', 1), ('erkfxyf', 1), ('nfjd', 1), ('vsr', 1), ('kxcifcv', 1), ('ddkrkgwwnfnaklxdg', 1), ('cheincs', 1), ('ucpegyhr', 1), ('jxtdz', 1), ('emaklov', 1), ('pefvrpt', 1), ('qnxldyya', 1), ('iuersial', 1), ('rrranhcq', 1), ('jepzug', 1), ('qvggjsdvnk', 1), ('ibmjw', 1), ('qnt', 1), ('jhj', 1), ('habfqwviwctkdxqupnpbikhjtjiylmevfzllvnwoggkaenkvbsoltry', 1), ('exff', 1), ('prcp', 1), ('ejfgu', 1), ('hih', 1), ('ognmjetr', 1), ('okkpaemfapugfinxzgn', 1), ('bwwvkifmzcsfrcsre', 1), ('wvfocm', 1), ('gqhn', 1), ('uow', 1), ('qrijelr', 1), ('yuhm', 1), ('lbonhp', 1), ('mlumfit', 1), ('atgzwedr', 1), ('orp', 1), ('nclbrvinpcrdn', 1), ('pzm', 1), ('xfd', 1), ('qqr', 1), ('poembzawfucemmdkg', 1), ('znyd', 1), ('bdftagjq', 1), ('czljtqjboyohmsd', 1), ('kaw', 1), ('guwxlcemjtijzkysxvmp', 1), ('dwujtas', 1), ('efca', 1), ('cfa', 1), ('sfv', 1), ('sfp', 1), ('scl', 1), ('fpr', 1), ('ptr', 1), ('infonorecords', 1), ('designate', 1), ('spamdiagnosticoutput', 1), ('spamdiagnosticmetadata', 1), ('namp', 1), ('originatororg', 1), ('originalarrivaltime', 1), ('fromentityheader', 1), ('hosted', 1), ('crosstenantheadersstamped', 1), ('eagsm', 1), ('markhtyingre', 1), ('serevrs', 1), ('dcaokyph', 1), ('vrdnocxs', 1), ('allert', 1), ('herghan', 1), ('joothyst', 1), ('exclusion', 1), ('traced', 1), ('crmdynamics', 1), ('safely', 1), ('neerthyu', 1), ('agrtywal', 1), ('vxhyftae', 1), ('tbkyfdli', 1), ('helo', 1), ('psid', 1), ('ebikdrqw', 1), ('empubwxo', 1), ('zbxljotu', 1), ('cbunzrak', 1), ('listen', 1), ('jogtse', 1), ('mhytusa', 1), ('hydra', 1), ('johjkse', 1), ('luartusa', 1), ('rahymos', 1), ('bajio', 1), ('exclusive', 1), ('disclosed', 1), ('conforming', 1), ('legislation', 1), ('reproduction', 1), ('shippingarea', 1), ('onfiirm', 1), ('maquinados', 1), ('tecnologia', 1), ('hidra', 1), ('saludos', 1), ('mailo', 1), ('bvpglyzh', 1), ('dyhusejm', 1), ('lineproto', 1), ('oweklxnm', 1), ('ubayizsq', 1), ('disagree', 1), ('careful', 1), ('tpfghtlugn', 1), ('mazak', 1), ('jurten', 1), ('setgyrt', 1), ('proposals', 1), ('tmb', 1), ('preserve', 1), ('trees', 1), ('aqzz', 1), ('erpquery', 1), ('memepr', 1), ('outlet', 1), ('ugyawsjv', 1), ('ypgjirlm', 1), ('doens', 1), ('sutyu', 1), ('magerjtyhd', 1), ('yadavtghya', 1), ('schgtewmik', 1), ('obdphylz', 1), ('qaeicrkz', 1), ('qbjmoihg', 1), ('nbgvyqac', 1), ('prefer', 1), ('maerza', 1), ('dds', 1), ('dss', 1), ('unterhaltung', 1), ('jfhytu', 1), ('mthyuleng', 1), ('periodically', 1), ('fce', 1), ('jinf', 1), ('vhw', 1), ('mess', 1), ('helftgyldt', 1), ('qoybxkfh', 1), ('dwcmxuea', 1), ('jncvkrzm', 1), ('thjquiyl', 1), ('jczwxvdn', 1), ('pexuklry', 1), ('blapmcwk', 1), ('dgrkbnua', 1), ('vvkusgtms', 1), ('qwghlvdx', 1), ('pjwvdiuz', 1), ('workman', 1), ('faerfrtbj', 1), ('modul', 1), ('marcom', 1), ('proofreading', 1), ('dropdox', 1), ('temprature', 1), ('aunpdmlj', 1), ('kzhapcve', 1), ('fpedscxo', 1), ('acuvyqnx', 1), ('recieving', 1), ('accont', 1), ('ecp', 1), ('rgtyob', 1), ('lafgseimer', 1), ('pjcompanyfeg', 1), ('wnrcyaks', 1), ('vmdwslkj', 1), ('exvcknbp', 1), ('angyta', 1), ('hgywselena', 1), ('brescsfgryiani', 1), ('acgyuna', 1), ('hartwell', 1), ('cadkey', 1), ('totmannhandys', 1), ('dps', 1), ('haveing', 1), ('cleanup', 1), ('lmdl', 1), ('faceing', 1), ('prtqz', 1), ('bonhyb', 1), ('knepkhsw', 1), ('resplved', 1), ('smhdyhtis', 1), ('constant', 1), ('lehhywsmat', 1), ('jonnht', 1), ('portfolio', 1), ('marcel', 1), ('aef', 1), ('abba', 1), ('coo', 1), ('kie', 1), ('encodi', 1), ('addr', 1), ('arpa', 1), ('anubisnetworks', 1), ('beffaf', 1), ('ambiance', 1), ('connec', 1), ('tion', 1), ('ffaf', 1), ('mbiance', 1), ('footer', 1), ('lcowx', 1), ('brgtyad', 1), ('ahdwqrson', 1), ('gthxezqp', 1), ('ainuhbmk', 1), ('physically', 1), ('density', 1), ('nderungen', 1), ('dvzlq', 1), ('kadjuwqama', 1), ('bekommt', 1), ('xhnmygfp', 1), ('bnpehyku', 1), ('happended', 1), ('skirtylport', 1), ('messmachine', 1), ('berirtch', 1), ('zedghkler', 1), ('bwgldaoe', 1), ('aczyfqjr', 1), ('kwehgxts', 1), ('agdsqbwv', 1), ('palyer', 1), ('gwmspqeo', 1), ('vwfetaqg', 1), ('extracting', 1), ('significantly', 1), ('xnlapdeq', 1), ('wupaeqlv', 1), ('structural', 1), ('mike', 1), ('tooltors', 1), ('oscar', 1), ('usero', 1), ('ivbkzcma', 1), ('nrehuqpa', 1), ('rtjwbuev', 1), ('gfpwdetq', 1), ('cvltebaj', 1), ('yzmcfxah', 1), ('rostuhhwr', 1), ('exepnse', 1), ('qdbfemro', 1), ('mcsqzlvd', 1), ('hctduems', 1), ('znalhivf', 1), ('sur', 1), ('domin', 1), ('surname', 1), ('wdgebvpzagavan', 1), ('slzhuipc', 1), ('sqntcber', 1), ('xomkhzrq', 1), ('vytqlphd', 1), ('boxrlpec', 1), ('fnkhwytl', 1), ('exits', 1), ('decommissioned', 1), ('isue', 1), ('morhyerw', 1), ('efqhmwpj', 1), ('hivumtfz', 1), ('zsdslsum', 1), ('xawlkiey', 1), ('lauthry', 1), ('globally', 1), ('comprehend', 1), ('coumikzb', 1), ('ubfcwegt', 1), ('viruhytph', 1), ('mud', 1), ('sludgeshipping', 1), ('hoscgthke', 1), ('metallurgy', 1), ('langhdte', 1), ('schneidk', 1), ('rper', 1), ('hbyoyer', 1), ('befertigung', 1), ('dahytrda', 1), ('retz', 1), ('kowski', 1), ('chsbhuo', 1), ('context', 1), ('cells', 1), ('aljbtwsh', 1), ('lepkbgix', 1), ('nehsytwrrej', 1), ('grwtfer', 1), ('damuphws', 1), ('chnagdrtymk', 1), ('eples', 1), ('chahdtyru', 1), ('sdlixwmb', 1), ('zvygmnco', 1), ('hwbipgfq', 1), ('sqiyfdax', 1), ('lanhuage', 1), ('ontario', 1), ('nwfoucba', 1), ('dzbujamc', 1), ('samsungsmg', 1), ('teamsales', 1), ('allocations', 1), ('retroactively', 1), ('summing', 1), ('adjustments', 1), ('hugely', 1), ('newer', 1), ('endkontrolle', 1), ('overloaded', 1), ('quer', 1), ('sad', 1), ('nozahtbr', 1), ('ubznqpsy', 1), ('byrljshv', 1), ('bwvmophd', 1), ('jfwvuzdn', 1), ('xackgvmd', 1), ('radgthika', 1), ('daserf', 1), ('leengineering', 1), ('zhm', 1), ('backdoor', 1), ('phoning', 1), ('urumqi', 1), ('mpatible', 1), ('ache', 1), ('hvu', 1), ('fcf', 1), ('lhqfinglbalfy', 1), ('agree', 1), ('jave', 1), ('sreenshot', 1), ('maschinenstillstand', 1), ('kesm', 1), ('fernwartung', 1), ('teichgr', 1), ('huges', 1), ('zdus', 1), ('leantracker', 1), ('pandepv', 1), ('variation', 1), ('raises', 1), ('ccif', 1), ('woehyl', 1), ('ssid', 1), ('edimax', 1), ('xgrhplvk', 1), ('coejktzn', 1), ('babhjbu', 1), ('gdgy', 1), ('generally', 1), ('beathe', 1), ('ibtvlfah', 1), ('dtlwscma', 1), ('bakheyr', 1), ('rjuyihes', 1), ('cegtcily', 1), ('jeshyensky', 1), ('wynhtydf', 1), ('twimc', 1), ('ruben', 1), ('castro', 1), ('teaming', 1), ('qipzctgs', 1), ('wjhtbpfr', 1), ('equal', 1), ('portelance', 1), ('chhyene', 1), ('dolhyt', 1), ('ianqdhmu', 1), ('camoysfq', 1), ('porteta', 1), ('hovering', 1), ('occasional', 1), ('spikes', 1), ('rudras', 1), ('mirror', 1), ('goyvcped', 1), ('sxbgiajh', 1), ('michbhuael', 1), ('laugdghjhlin', 1), ('aacount', 1), ('ki', 1), ('wbs', 1), ('profitability', 1), ('vvshyuwb', 1), ('brandhyht', 1), ('muthdyrta', 1), ('ewa', 1), ('ngfedxrp', 1), ('oirmgqcs', 1), ('hodgek', 1), ('andhtyju', 1), ('volkhd', 1), ('schedulerbhml', 1), ('qualify', 1), ('leadin', 1), ('claims', 1), ('thro', 1), ('needfufl', 1), ('sanmhty', 1), ('mahatndhyua', 1), ('kk', 1), ('sqlcuhep', 1), ('railgnfb', 1), ('replication', 1), ('ld', 1), ('zredeploy', 1), ('gbirhjat', 1), ('fptbrhwv', 1), ('optical', 1), ('seltxfkw', 1), ('ontgxqwy', 1), ('addreess', 1), ('tring', 1), ('procache', 1), ('cscache', 1), ('comunication', 1), ('approximate', 1), ('trip', 1), ('milli', 1), ('imaginal', 1), ('stockmanagement', 1), ('hinderance', 1), ('tofinance', 1), ('officer', 1), ('cursors', 1), ('opposite', 1), ('lombab', 1), ('vrcqhnty', 1), ('ajomhkfv', 1), ('nxa', 1), ('epj', 1), ('rsir', 1), ('unmarkhty', 1), ('prelim', 1), ('sauber', 1), ('seiten', 1), ('wedgrtyh', 1), ('ahsnwtey', 1), ('gesamte', 1), ('anlegt', 1), ('nochmals', 1), ('ckgestzt', 1), ('vermutlich', 1), ('mnlazfsr', 1), ('mtqrkhnx', 1), ('ramdntythanjeshkurtyar', 1), ('radjkanjesh', 1), ('sidrthy', 1), ('zpwgoqju', 1), ('sadjuetha', 1), ('shwyhdtu', 1), ('sot', 1), ('vignbhyesh', 1), ('etr', 1), ('imposible', 1), ('therefor', 1), ('hagemeyer', 1), ('oci', 1), ('vide', 1), ('debited', 1), ('berechnungsprogramdntym', 1), ('sicherungsdatei', 1), ('produktionsinfos', 1), ('zlx', 1), ('verlauf', 1), ('stehen', 1), ('kudhnyuhnm', 1), ('kesrgtyu', 1), ('rockehsty', 1), ('koiergyvh', 1), ('simekdty', 1), ('vergleichsmitarbeiter', 1), ('beckshtywsnh', 1), ('beckshtyw', 1), ('instaliert', 1), ('instalieren', 1), ('qiscgfjv', 1), ('kxfdsijv', 1), ('rickjdt', 1), ('nfybpxdg', 1), ('kinsella', 1), ('ainw', 1), ('studies', 1), ('umstellen', 1), ('konfigurieren', 1), ('chargeur', 1), ('poste', 1), ('octophon', 1), ('prof', 1), ('ladeschale', 1), ('mnr', 1), ('prozessflussverantwortliche', 1), ('scdp', 1), ('awa', 1), ('abgelaufen', 1), ('einleiten', 1), ('vadnhyt', 1), ('stegyhui', 1), ('manjhyt', 1), ('vbap', 1), ('zzcmpgn', 1), ('considered', 1), ('jahtyuj', 1), ('reiceve', 1), ('cihaz', 1), ('modeli', 1), ('kimli', 1), ('hnorau', 1), ('era', 1), ('letim', 1), ('sistemi', 1), ('kullan', 1), ('arac', 1), ('eri', 1), ('nedeni', 1), ('workshop', 1), ('ifbg', 1), ('cfc', 1), ('fulfillment', 1), ('stands', 1), ('swathi', 1), ('coordinate', 1), ('vbkpf', 1), ('nieconn', 1), ('nfe', 1), ('xblnr', 1), ('bkpf', 1), ('kahrthyeuiuiw', 1), ('sctqwgmj', 1), ('yambwtfk', 1), ('bohyub', 1), ('jaya', 1), ('conveyed', 1), ('fulfilled', 1), ('klarp', 1), ('geuacyltoe', 1), ('hxgayczeet', 1), ('einwandfrei', 1), ('lagging', 1), ('lag', 1), ('embarrassing', 1), ('apologise', 1), ('rarely', 1), ('mid', 1), ('mornings', 1), ('companyst', 1), ('apc', 1), ('mehrugshy', 1), ('gmrxwqlf', 1), ('vzacdmbj', 1), ('referencing', 1), ('barcodes', 1), ('scannable', 1), ('revisit', 1), ('fastenal', 1), ('claiming', 1), ('datamax', 1), ('notxygdz', 1), ('mjudivse', 1), ('fwchqjor', 1), ('indiana', 1), ('pnhlrfao', 1), ('ivjxlyfz', 1), ('zqbgmfle', 1), ('wrkmieao', 1), ('mmbe', 1), ('managementbe', 1), ('bakertm', 1), ('svlcqmnb', 1), ('qlwerhxt', 1), ('bases', 1), ('eliminate', 1), ('coworker', 1), ('remfg', 1), ('toolted', 1), ('webdhyt', 1), ('outook', 1), ('drlab', 1), ('jerhtyua', 1), ('schdule', 1), ('lhqksbdxac', 1), ('switzerlandim', 1), ('switzerlandik', 1), ('fighting', 1), ('mangers', 1), ('sonstiges', 1), ('taskbar', 1), ('ltcw', 1), ('funktionierende', 1), ('tkhaymqg', 1), ('speaking', 1), ('faint', 1), ('clzwduvj', 1), ('keflinbj', 1), ('quits', 1), ('tonerpatrone', 1), ('yzodcxkn', 1), ('zyewibop', 1), ('freundlicher', 1), ('gastronomie', 1), ('jahr', 1), ('kassiaryu', 1), ('mahapthysk', 1), ('ngyht', 1), ('loginto', 1), ('maghtyion', 1), ('cnjkeko', 1), ('cekomthyr', 1), ('gaop', 1), ('scievjwr', 1), ('cdlsvoif', 1), ('sarhfa', 1), ('jack', 1), ('relly', 1), ('naisdxtk', 1), ('mqzvewsb', 1), ('manjuvghy', 1), ('nederland', 1), ('wim', 1), ('duisenbergplantsoen', 1), ('maastricht', 1), ('postbus', 1), ('schrenfgker', 1), ('heinrifgtch', 1), ('lesen', 1), ('knopfdruck', 1), ('informiert', 1), ('haende', 1), ('hrrn', 1), ('pries', 1), ('iwqfelcu', 1), ('gsubfiml', 1), ('vvdgtyachac', 1), ('cheghthan', 1), ('achghar', 1), ('zh', 1), ('kassia', 1), ('jerydwbn', 1), ('gdylnaue', 1), ('oraarch', 1), ('achghyardr', 1), ('dmexgspl', 1), ('mruzqhac', 1), ('wsczgfal', 1), ('hjfklsdg', 1), ('witam', 1), ('zg', 1), ('aszam', 1), ('opoty', 1), ('nie', 1), ('zrobi', 1), ('synchronizacji', 1), ('brak', 1), ('raport', 1), ('itp', 1), ('pozdrawiam', 1), ('bath', 1), ('hgyvopct', 1), ('dhckfmbq', 1), ('sdguo', 1), ('rujpckto', 1), ('awswering', 1), ('passwordmanager', 1), ('handy', 1), ('internetsignal', 1), ('hartghymutg', 1), ('internetleitung', 1), ('anschluss', 1), ('eingang', 1), ('ganedsght', 1), ('plznsryi', 1), ('ikugwqec', 1), ('servrs', 1), ('serves', 1), ('srinifghvahs', 1), ('degrees', 1), ('poe', 1), ('injector', 1), ('acessar', 1), ('slight', 1), ('editable', 1), ('intuitive', 1), ('challenging', 1), ('vertical', 1), ('strips', 1), ('pic', 1), ('fluke', 1), ('trueview', 1), ('appliance', 1), ('intermittenly', 1), ('tally', 1), ('hdfvbjoe', 1), ('obvrxtya', 1), ('raghjkavhjkdra', 1), ('mentions', 1), ('ltjkirwy', 1), ('acts', 1), ('positions', 1), ('ifblxjmc', 1), ('spqrtkew', 1), ('vpmnusaf', 1), ('avigtshay', 1), ('subdirectory', 1), ('promoting', 1), ('mutual', 1), ('lopgin', 1), ('attribute', 1), ('fctmzhyk', 1), ('cznlfbom', 1), ('manuten', 1), ('cria', 1), ('documentos', 1), ('diversos', 1), ('importa', 1), ('exporta', 1), ('uzpycdho', 1), ('passward', 1), ('arrojhsjd', 1), ('maps', 1), ('johghajknn', 1), ('hipghkinjyt', 1), ('zikuvsat', 1), ('spxycizr', 1), ('paste', 1), ('mjvfxnka', 1), ('hits', 1), ('privilages', 1), ('pdfmailer', 1), ('installl', 1), ('powering', 1), ('einmal', 1), ('nachschauen', 1), ('xwgnvksi', 1), ('dwijxgob', 1), ('zmgsfner', 1), ('hybegvwo', 1), ('facts', 1), ('alphastdgtyal', 1), ('za', 1), ('rcpt', 1), ('wunderlist', 1), ('sehe', 1), ('reisekostenabrechnungen', 1), ('gogtyek', 1), ('ustvaifg', 1), ('hmzfewks', 1), ('mathyuithihyt', 1), ('engracia', 1), ('nad', 1), ('tempdev', 1), ('portugal', 1), ('authenticate', 1), ('soap', 1), ('nfchost', 1), ('stg', 1), ('datastore', 1), ('vmx', 1), ('vahjtusa', 1), ('wenghtyele', 1), ('weghyndlv', 1), ('brembo', 1), ('experiancing', 1), ('currentlyx', 1), ('axhg', 1), ('complain', 1), ('syncronize', 1), ('lzycofut', 1), ('mzbovhpd', 1), ('bobhyb', 1), ('discconect', 1), ('takheghshi', 1), ('afefsano', 1), ('illustrate', 1), ('workng', 1), ('bestand', 1), ('stl', 1), ('ipd', 1), ('wasload', 1), ('kennconnect', 1), ('dxnskvbm', 1), ('xbaswghy', 1), ('duplex', 1), ('matches', 1), ('pratma', 1), ('deusad', 1), ('sgd', 1), ('effects', 1), ('qmkpsbgl', 1), ('zfovlrah', 1), ('infotrmed', 1), ('dwjvfkqe', 1), ('transiit', 1), ('smitctdrhell', 1), ('ims', 1), ('executes', 1), ('verifies', 1), ('notion', 1), ('mdulwthb', 1), ('sldowapb', 1), ('generator', 1), ('closets', 1), ('resumes', 1), ('skirtylset', 1), ('setrup', 1), ('beec', 1), ('yerrav', 1), ('ulezhxfw', 1), ('kslocjtaaruthapandian', 1), ('karhjuyutk', 1), ('sections', 1), ('excuse', 1), ('ethnics', 1), ('ivdntecr', 1), ('cdasfpjb', 1), ('netzwerke', 1), ('synchronisieren', 1), ('beiden', 1), ('identisch', 1), ('stimmen', 1), ('berein', 1), ('vorab', 1), ('mafghyrina', 1), ('ntner', 1), ('tfsehruw', 1), ('dzrgpkyn', 1), ('wirelss', 1), ('multinational', 1), ('hgwofcbx', 1), ('tnowikyv', 1), ('gmrkisxy', 1), ('wgtcylir', 1), ('vijuryat', 1), ('dgurhtya', 1), ('reselect', 1), ('qerror', 1), ('rder', 1), ('qlong', 1), ('eletronico', 1), ('segue', 1), ('speker', 1), ('conferences', 1), ('confernece', 1), ('cker', 1), ('possibilities', 1), ('safghghga', 1), ('gabryltka', 1), ('sabhtyhiko', 1), ('komar', 1), ('truck', 1), ('fitness', 1), ('signage', 1), ('aufgedruckter', 1), ('zeichnungsnummer', 1), ('richtigstellen', 1), ('tfhzidnq', 1), ('wqpfdobe', 1), ('ksvqzmre', 1), ('yqnajdwh', 1), ('heighjtyich', 1), ('entschuldigung', 1), ('falschen', 1), ('gdw', 1), ('paar', 1), ('monate', 1), ('zweites', 1), ('umbenannt', 1), ('konvertiert', 1), ('waren', 1), ('rechts', 1), ('druckstempel', 1), ('eurer', 1), ('gescannten', 1), ('nictafvwlpz', 1), ('opfigqramdntyatisch', 1), ('konnten', 1), ('etwa', 1), ('davon', 1), ('machten', 1), ('hundert', 1), ('scheinen', 1), ('euch', 1), ('reparieren', 1), ('bedarf', 1), ('consisting', 1), ('ordinary', 1), ('stamm', 1), ('vollen', 1), ('preu', 1), ('lilesfhpk', 1), ('eqmuniov', 1), ('ehxkcbgj', 1), ('ahmet', 1), ('materiallager', 1), ('macht', 1), ('anschauen', 1), ('formatierung', 1), ('wiypaqtu', 1), ('lmgqbpvi', 1), ('bwvrncih', 1), ('ibuyfrcq', 1), ('dipl', 1), ('reenable', 1), ('henghl', 1), ('diginet', 1), ('mabelteghj', 1), ('syghmesa', 1), ('vtzhelgs', 1), ('ivewqogm', 1), ('btyvqhjw', 1), ('xbyolhsw', 1), ('aufgerufen', 1), ('auftragsbereitstellung', 1), ('spend', 1), ('bsm', 1), ('workaround', 1), ('mcgudftigre', 1), ('steince', 1), ('elm', 1), ('qohfjpna', 1), ('exphkims', 1), ('raju', 1), ('leslie', 1), ('themfg', 1), ('billghj', 1), ('dhjuyick', 1), ('lipfnxsy', 1), ('rvjlnpef', 1), ('legit', 1), ('krisyuhnyrtkurtyar', 1), ('kghpanhjuwdian', 1), ('ranging', 1), ('ghjvreicj', 1), ('tszvorba', 1), ('wtldpncx', 1), ('jreichard', 1), ('ppstrixner', 1), ('nkjtoxwv', 1), ('wqtlzvxu', 1), ('frei', 1), ('schalten', 1), ('wichtigkeit', 1), ('hoch', 1), ('hohgajnn', 1), ('wollte', 1), ('weiteres', 1), ('sicher', 1), ('korrekten', 1), ('anmeldedaten', 1), ('anbetracht', 1), ('bevorstehenden', 1), ('abrechnung', 1), ('teamleitung', 1), ('klein', 1), ('strixner', 1), ('ahydmrbu', 1), ('fjymgtvo', 1), ('eiomnuba', 1), ('blokker', 1), ('teh', 1), ('lcamiopz', 1), ('rqll', 1), ('ovxwqybe', 1), ('gevzkrlp', 1), ('incase', 1), ('raghfhgh', 1), ('gowhjtya', 1), ('froajhdb', 1), ('ijetmkuc', 1), ('cached', 1), ('simplyfies', 1), ('troubleshooter', 1), ('toolhone', 1), ('realtek', 1), ('earphone', 1), ('conclusion', 1), ('dialogue', 1), ('spit', 1), ('engineers', 1), ('prevents', 1), ('allocating', 1), ('inventrtgoy', 1), ('odd', 1), ('subjected', 1), ('gezpktrq', 1), ('opkqwevj', 1), ('wl', 1), ('deghjick', 1), ('culghjn', 1), ('cizplrvw', 1), ('ymxgteir', 1), ('jstart', 1), ('bearbeitung', 1), ('znqlmjvt', 1), ('uhyokzlt', 1), ('eingeblendet', 1), ('raonoke', 1), ('rapids', 1), ('barcodescanner', 1), ('befindet', 1), ('wmp', 1), ('uadkqcsj', 1), ('xtmjlari', 1), ('amb', 1), ('anuxbyzg', 1), ('bvsqcjkw', 1), ('departmenst', 1), ('leserecht', 1), ('glichen', 1), ('arbeit', 1), ('wcnfvajb', 1), ('kxylsamv', 1), ('shikghtyuha', 1), ('gcxvtbrz', 1), ('oportunities', 1), ('seibel', 1), ('prodn', 1), ('diconnection', 1), ('ipqgrnxk', 1), ('acxedqjm', 1), ('tjtigtyps', 1), ('zeilmann', 1), ('hrend', 1), ('kopiervorganges', 1), ('geknittert', 1), ('facixepyfbga', 1), ('plastic', 1), ('groove', 1), ('controll', 1), ('qwg', 1), ('ralf', 1), ('takeshi', 1), ('asano', 1), ('mathyida', 1), ('shighjvnn', 1), ('effected', 1), ('registering', 1), ('cagrty', 1), ('iehdjwvt', 1), ('rndtlyhv', 1), ('paris', 1), ('rcfwnpbi', 1), ('kvhedyrc', 1), ('inqury', 1), ('theme', 1), ('reopned', 1), ('scren', 1), ('manages', 1), ('bgwneavl', 1), ('owners', 1), ('nrbcqwgj', 1), ('johyue', 1), ('ghjuardt', 1), ('hakim', 1), ('belhadjhamida', 1), ('hakityum', 1), ('plantronics', 1), ('caexmols', 1), ('udpate', 1), ('sarmtlhyanardhanan', 1), ('rqiw', 1), ('rqigfgage', 1), ('edula', 1), ('venkat', 1), ('erihyuk', 1), ('clarifications', 1), ('rodstock', 1), ('jankowski', 1), ('maschinen', 1), ('evh', 1), ('schlie', 1), ('zwei', 1), ('geschrieben', 1), ('behoben', 1), ('origin', 1), ('proforma', 1), ('sentences', 1), ('gekennzeichnete', 1), ('qualit', 1), ('tsmanagement', 1), ('metroligic', 1), ('canner', 1), ('freetext', 1), ('requisitions', 1), ('cars', 1), ('changeable', 1), ('overwritten', 1), ('bestellung', 1), ('katalog', 1), ('pltkqrfd', 1), ('bfohnjmz', 1), ('sende', 1), ('einkaufen', 1), ('cute', 1), ('atcl', 1), ('atclx', 1), ('yi', 1), ('kuttiadi', 1), ('kurs', 1), ('absolvieren', 1), ('bekomme', 1), ('hierher', 1), ('besten', 1), ('nschen', 1), ('eckersdorfer', 1), ('bayern', 1), ('hra', 1), ('byer', 1), ('outloot', 1), ('rscrm', 1), ('qskwgjym', 1), ('bsqdaxhf', 1), ('intrans', 1), ('differential', 1), ('usr', 1), ('connectt', 1), ('incompletion', 1), ('acess', 1), ('lkwspqce', 1), ('knxaipyj', 1), ('allowe', 1), ('battel', 1), ('debhyue', 1), ('fhyuiinch', 1), ('yoltmegh', 1), ('bmadvixs', 1), ('brianna', 1), ('tnghnha', 1), ('ajuiegrson', 1), ('connectors', 1), ('payout', 1), ('modification', 1), ('arises', 1), ('assing', 1), ('shiv', 1), ('devise', 1), ('kkbsm', 1), ('sica', 1), ('pez', 1), ('rufo', 1), ('thhyuokhkp', 1), ('hoi', 1), ('wikfnmlwds', 1), ('wmtiruac', 1), ('inn', 1), ('printout', 1), ('shipament', 1), ('lable', 1), ('ajuyanni', 1), ('dmqjhrso', 1), ('gzbunwie', 1), ('italian', 1), ('shipmet', 1), ('weak', 1), ('barley', 1), ('dual', 1), ('papiertransport', 1), ('druckerpapiers', 1), ('treten', 1), ('regelm', 1), ('igen', 1), ('nden', 1), ('einkerbungen', 1), ('papiereinzug', 1), ('wall', 1), ('duca', 1), ('productmanagement', 1), ('capturing', 1), ('karnataka', 1), ('herewith', 1), ('aspect', 1), ('vvlahstyurr', 1), ('student', 1), ('instatnt', 1), ('fywphaxc', 1), ('jacdqbks', 1), ('instantly', 1), ('dgwrmsja', 1), ('jzlpwuit', 1), ('freimachen', 1), ('proven', 1), ('foun', 1), ('konten', 1), ('referenz', 1), ('twdyzsfr', 1), ('gjedmfvh', 1), ('auftraggeber', 1), ('nm', 1), ('nghyuakm', 1), ('druing', 1), ('fjnmxoya', 1), ('iljgptas', 1), ('unseren', 1), ('schichtf', 1), ('hrern', 1), ('gabryltk', 1), ('byczkowski', 1), ('stankewitz', 1), ('vorbeikommen', 1), ('accespoint', 1), ('wsb', 1), ('datenlogger', 1), ('opus', 1), ('thip', 1), ('raumtemperatur', 1), ('externen', 1), ('wartung', 1), ('eorjylzm', 1), ('vrkofesj', 1), ('subnet', 1), ('tomoe', 1), ('vvdghtteij', 1), ('pruchase', 1), ('defigned', 1), ('maitained', 1), ('urgnet', 1), ('xcirqlup', 1), ('zopbiufn', 1), ('asiapac', 1), ('apacpuchn', 1), ('vmsw', 1), ('handscanner', 1), ('ramdntygy', 1), ('rhozsfty', 1), ('ijyuvind', 1), ('sohytganvi', 1), ('jeyabalan', 1), ('jofvunqs', 1), ('uwigjmzv', 1), ('yxvqdtmk', 1), ('kbicqjrp', 1), ('yjurztgd', 1), ('dacl', 1), ('sifco', 1), ('collated', 1), ('lcoked', 1), ('clientless', 1), ('sarhytukas', 1), ('snaps', 1), ('yicpojmf', 1), ('chian', 1), ('kindftyed', 1), ('dghuane', 1), ('whryuiams', 1), ('gerados', 1), ('convention', 1), ('geengineering', 1), ('tooloductdata', 1), ('demoed', 1), ('uli', 1), ('underscore', 1), ('graph', 1), ('approfghaching', 1), ('rmas', 1), ('afcbrhqw', 1), ('vudghzcb', 1), ('subsitute', 1), ('substitute', 1), ('qsoxltny', 1), ('dzjespml', 1), ('tjnwdauo', 1), ('jkdwbhgs', 1), ('nz', 1), ('jmkcewds', 1), ('qkoipbzn', 1), ('gather', 1), ('reconnected', 1), ('tinmuym', 1), ('ltcl', 1), ('hgmx', 1), ('rarty', 1), ('cds', 1), ('converstion', 1), ('mdosid', 1), ('opp', 1), ('bahdqrcs', 1), ('onbankrding', 1), ('hired', 1), ('disaster', 1), ('directions', 1), ('caught', 1), ('sharee', 1), ('backdate', 1), ('salaried', 1), ('technically', 1), ('dinners', 1), ('passed', 1), ('frustrated', 1), ('timeframdntye', 1), ('marhty', 1), ('vms', 1), ('oewshlmd', 1), ('azjfshry', 1), ('pain', 1), ('meetinmg', 1), ('fqiurzas', 1), ('eidtfbqk', 1), ('vvlfbhtyeisd', 1), ('anylonger', 1), ('guvgytniak', 1), ('bobje', 1), ('storch', 1), ('ffner', 1), ('lfel', 1), ('stationiert', 1), ('betreuen', 1), ('zeigt', 1), ('fwkxbley', 1), ('stndeick', 1), ('sonhygg', 1), ('vmthcrkf', 1), ('iceyusnd', 1), ('pipfhypeu', 1), ('mbs', 1), ('departmentlaufwerk', 1), ('unterordner', 1), ('verschwunden', 1), ('wrk', 1), ('angeh', 1), ('ngten', 1), ('ujbaemlc', 1), ('ilzhrxjo', 1), ('lkrfndev', 1), ('kztlojin', 1), ('resigned', 1), ('valn', 1), ('thi', 1), ('houcdelq', 1), ('wnypackq', 1), ('druckers', 1), ('heiner', 1), ('sponsel', 1), ('lvlw', 1), ('ltcevgap', 1), ('easiest', 1), ('fabxjimdghtyo', 1), ('depfugcy', 1), ('readded', 1), ('lhol', 1), ('vvnookr', 1), ('exceptions', 1), ('jbfmsxik', 1), ('mfzjncva', 1), ('notch', 1), ('snagit', 1), ('lucgnhda', 1), ('carthy', 1), ('gcknzthb', 1), ('dots', 1), ('harrfgyibs', 1), ('messgage', 1), ('manipulator', 1), ('excessive', 1), ('affection', 1), ('proglovia', 1), ('ucs', 1), ('databaseslisted', 1), ('exposition', 1), ('thilo', 1), ('specifics', 1), ('zmkitbsh', 1), ('bxsyaipz', 1), ('bedord', 1), ('ldbsm', 1), ('lizenz', 1), ('rmzlvqjf', 1), ('eaqyxljb', 1), ('dhmfuvgw', 1), ('jralkfcb', 1), ('hufghygh', 1), ('attn', 1), ('auftragspapiere', 1), ('staut', 1), ('ausdruck', 1), ('berirtchts', 1), ('gelbe', 1), ('felder', 1), ('llt', 1), ('berirtcht', 1), ('bertagen', 1), ('extenral', 1), ('flashes', 1), ('sreedhar', 1), ('hris', 1), ('ngerungskabel', 1), ('hxwtidja', 1), ('ixahzmvf', 1), ('cudgevmx', 1), ('waqslrbd', 1), ('metrics', 1), ('kunde', 1), ('erh', 1), ('rechnungen', 1), ('rechnungsempf', 1), ('empfangende', 1), ('sew', 1), ('eurodrive', 1), ('problemlos', 1), ('bezahlt', 1), ('feststellen', 1), ('usser', 1), ('assets', 1), ('additions', 1), ('cwip', 1), ('settled', 1), ('immediateley', 1), ('businesstrip', 1), ('jfsmwpny', 1), ('klxsdqiw', 1), ('alter', 1), ('halbautomaten', 1), ('schaltet', 1), ('hitze', 1), ('dreht', 1), ('receivable', 1), ('bxgwyamr', 1), ('hjbukvcq', 1), ('personalnr', 1), ('zpsr', 1), ('zpsu', 1), ('pavan', 1), ('enterence', 1), ('chbvyjqr', 1), ('dqbwijvy', 1), ('ferbfhyunam', 1), ('ekmw', 1), ('vivbhuek', 1), ('kanjdye', 1), ('cleaner', 1), ('blockers', 1), ('depreciation', 1), ('apacjun', 1), ('otmpauyr', 1), ('ijztksal', 1), ('ybuvlkjq', 1), ('nwcobvpl', 1), ('tjlgzkbp', 1), ('iervwjzg', 1), ('iwtvrhnz', 1), ('rxiumhfk', 1), ('qmglkaru', 1), ('thgheijmer', 1), ('miltgntyuon', 1), ('knighdjhtyt', 1), ('involves', 1), ('significant', 1), ('svbymfiz', 1), ('afqvyuwh', 1), ('cjsukemx', 1), ('lqkcesuf', 1), ('rubyfgty', 1), ('resignation', 1), ('organizational', 1), ('adjtmlzn', 1), ('medial', 1), ('smgnhyleck', 1), ('bhghtyum', 1), ('ytcxjzue', 1), ('guplftok', 1), ('krafghyumec', 1), ('zbijdqpg', 1), ('ehpjkwio', 1), ('dinktyhed', 1), ('allen', 1), ('unterordnern', 1), ('ekql', 1), ('interrogates', 1), ('prtgn', 1), ('gn', 1), ('investigation', 1), ('ebom', 1), ('screw', 1), ('screws', 1), ('joe', 1), ('mafgtnik', 1), ('abreu', 1), ('problleme', 1), ('bejvhsfx', 1), ('dmvsclhp', 1), ('emzlw', 1), ('reghythicsa', 1), ('purvis', 1), ('pughjuvirl', 1), ('appends', 1), ('differences', 1), ('gebraucht', 1), ('observerd', 1), ('bizagi', 1), ('securities', 1), ('xcel', 1), ('lbxgodfu', 1), ('usperhki', 1), ('piolfghim', 1), ('viewed', 1), ('compatibilty', 1), ('dhtxwcng', 1), ('hruckmey', 1), ('pol', 1), ('vvtghjscha', 1), ('assumed', 1), ('kreghtmph', 1), ('kujfgtats', 1), ('fh', 1), ('mzdkgnvs', 1), ('svhkgyqb', 1), ('unresponsive', 1), ('rajy', 1), ('roh', 1), ('enormity', 1), ('humanity', 1), ('mankind', 1), ('insensitive', 1), ('gas', 1), ('agian', 1), ('projection', 1), ('witrh', 1), ('rofghach', 1), ('latin', 1), ('fnqelwpk', 1), ('ahrskvln', 1), ('ftnijxup', 1), ('sbltduco', 1), ('nafghyncy', 1), ('gracias', 1), ('dataloads', 1), ('xsdb', 1), ('zenjimdghtybo', 1), ('xirzfpob', 1), ('touchpad', 1), ('thdjzolw', 1), ('submissions', 1), ('exepsne', 1), ('laurent', 1), ('ctry', 1), ('investigat', 1), ('netzwerkdrucker', 1), ('gast', 1), ('generieren', 1), ('laut', 1), ('techniker', 1), ('hersteller', 1), ('wohl', 1), ('sehen', 1), ('nachverfolgen', 1), ('stoppen', 1), ('versuche', 1), ('abzubrechen', 1), ('steckers', 1), ('kabels', 1), ('ews', 1), ('ddl', 1), ('swapped', 1), ('rita', 1), ('baugtymli', 1), ('deloitte', 1), ('touche', 1), ('lcitrixerpall', 1), ('maximgbilian', 1), ('wilsfrtch', 1), ('carolutyuin', 1), ('votgygel', 1), ('pow', 1), ('ercor', 1), ('chart', 1), ('omfvxjpw', 1), ('htiemzsg', 1), ('qcfmxgid', 1), ('jvxanwre', 1), ('ltu', 1), ('inventrory', 1), ('resale', 1), ('workings', 1), ('bound', 1), ('loaners', 1), ('borrowed', 1), ('hotwlygp', 1), ('afukzhnm', 1), ('lzuwmhpr', 1), ('riuheqsg', 1), ('zulassen', 1), ('dknejifu', 1), ('dljvtebc', 1), ('ugawcoye', 1), ('jcfqgviy', 1), ('tastatutur', 1), ('eewhse', 1), ('jmoqelbc', 1), ('fbzsyjne', 1), ('effeghnk', 1), ('imuwhokc', 1), ('ijdfnayb', 1), ('visual', 1), ('coverage', 1), ('praddgtip', 1), ('kumfgtyar', 1), ('sagar', 1), ('outsource', 1), ('prioritise', 1), ('lese', 1), ('schreibberechtigung', 1), ('kbt', 1), ('auftrgasbearb', 1), ('richte', 1), ('nihktgsh', 1), ('kaghjtra', 1), ('heghjyder', 1), ('guldiwmn', 1), ('aroqwuvz', 1), ('ploease', 1), ('verantwortlichen', 1), ('clara', 1), ('vhebydgs', 1), ('rtmjwyvk', 1), ('qualitycontrol', 1), ('layered', 1), ('audi', 1), ('kmfg', 1), ('lpa', 1), ('countermeasures', 1), ('assisted', 1), ('transports', 1), ('venfgugjhytpal', 1), ('nythug', 1), ('sepbdtou', 1), ('noredzyx', 1), ('nil', 1), ('madhaw', 1), ('rai', 1), ('serch', 1), ('belt', 1), ('recertification', 1), ('sudden', 1), ('belts', 1), ('clswzxoq', 1), ('higqaepr', 1), ('pattghyuy', 1), ('karcgdwswski', 1), ('ykolismx', 1), ('kbysnuim', 1), ('gerusky', 1), ('jafgty', 1), ('verghjuen', 1), ('mukghyuhea', 1), ('scghhnelligkeit', 1), ('internets', 1), ('plnvcwuq', 1), ('ikupsjhb', 1), ('anpassung', 1), ('speedport', 1), ('jclrangd', 1), ('kjlnearz', 1), ('muejkipler', 1), ('hoffe', 1), ('ansprechpartnerin', 1), ('wissen', 1), ('weiter', 1), ('voigt', 1), ('greifbar', 1), ('gerberghty', 1), ('rde', 1), ('vermuten', 1), ('direkte', 1), ('mithilfe', 1), ('bzgl', 1), ('beurteilung', 1), ('allerdings', 1), ('unternehmen', 1), ('warten', 1), ('nanrfakurtyar', 1), ('edmsm', 1), ('schulzgth', 1), ('abhandengekommen', 1), ('holen', 1), ('ihn', 1), ('reinstellen', 1), ('sistem', 1), ('kanghytaim', 1), ('keyence', 1), ('dwgliuyt', 1), ('ieqgdpbm', 1), ('ffnerfunktion', 1), ('zeiterfassungskarten', 1), ('dhl', 1), ('wcupoaty', 1), ('fqnzwphj', 1), ('qgilmtyc', 1), ('gmscovxa', 1), ('reseat', 1), ('lwl', 1), ('guestcompany', 1), ('coivmhwj', 1), ('opwckbrv', 1), ('milling', 1), ('substitution', 1), ('sooner', 1), ('vvgtybyrn', 1), ('unconverted', 1), ('accomplish', 1), ('gigabitet', 1), ('hernet', 1), ('nexus', 1), ('frseoupk', 1), ('feluybrn', 1), ('karghytuthik', 1), ('slfxjznk', 1), ('hmspexor', 1), ('organise', 1), ('abovementioned', 1), ('uwlnpscr', 1), ('lvkujfse', 1), ('nyzxjwud', 1), ('lia', 1), ('fin', 1), ('tbioceqj', 1), ('ukjietwz', 1), ('emailing', 1), ('htm', 1), ('bellusco', 1), ('vigrtgyne', 1), ('ravhdyui', 1), ('dbd', 1), ('jayachandran', 1), ('sat', 1), ('battling', 1), ('hurricane', 1), ('beach', 1), ('mailsi', 1), ('lynda', 1), ('bigtyl', 1), ('bachsmhdyhti', 1), ('zfliqpxm', 1), ('dgfvaqlh', 1), ('milton', 1), ('keynes', 1), ('pqwehmzgagannathan', 1), ('jumping', 1), ('piezas', 1), ('merthayu', 1), ('behsnjty', 1), ('donot', 1), ('bra', 1), ('embertell', 1), ('wgpelvyn', 1), ('ouaepwnr', 1), ('pqnteriv', 1), ('michigan', 1), ('vvterra', 1), ('fmeozwng', 1), ('pfneutkg', 1), ('pltndoab', 1), ('jftyff', 1), ('bgtyrant', 1), ('brahdthyu', 1), ('samag', 1), ('lieferschein', 1), ('vujymcls', 1), ('sgpmyviq', 1), ('lkw', 1), ('abholung', 1), ('ort', 1), ('adwind', 1), ('qxhdcnmj', 1), ('caflvjrn', 1), ('vefghgarr', 1), ('foreach', 1), ('rcp', 1), ('enceinjury', 1), ('esp', 1), ('xef', 1), ('beb', 1), ('ssylias', 1), ('assyli', 1), ('dada', 1), ('dea', 1), ('eaeb', 1), ('bbb', 1), ('bcb', 1), ('bef', 1), ('abca', 1), ('aeb', 1), ('deab', 1), ('cfe', 1), ('cfef', 1), ('fcd', 1), ('aff', 1), ('fad', 1), ('cfec', 1), ('ccf', 1), ('ebd', 1), ('fba', 1), ('dbe', 1), ('bba', 1), ('ede', 1), ('bfda', 1), ('ddb', 1), ('eba', 1), ('edb', 1), ('uiotbkhs', 1), ('grymzxiq', 1), ('hardman', 1), ('connector', 1), ('pnabslgh', 1), ('presse', 1), ('blinkt', 1), ('rechnerf', 1), ('stationen', 1), ('externe', 1), ('gung', 1), ('nobook', 1), ('edspmloy', 1), ('fxnkzaqu', 1), ('toolmails', 1), ('meant', 1), ('efbwiadp', 1), ('paasword', 1), ('activex', 1), ('mhfjudahdyue', 1), ('rfgrhtdy', 1), ('debbie', 1), ('festtelefon', 1), ('erneuern', 1), ('dtp', 1), ('bidengineering', 1), ('coferte', 1), ('rofgtyger', 1), ('liugh', 1), ('heptuizns', 1), ('synchs', 1), ('tot', 1), ('tablearu', 1), ('ufesrwmz', 1), ('egujslwx', 1), ('inf', 1), ('mbr', 1), ('markhtyets', 1), ('mbrreporting', 1), ('topcustomersendmarkhtyets', 1), ('entrance', 1), ('sridhar', 1), ('drviers', 1), ('cgo', 1), ('marry', 1), ('accees', 1), ('parfgtkym', 1), ('leegtysm', 1), ('llvw', 1), ('sounds', 1), ('ultra', 1), ('tial', 1), ('kcu', 1), ('xzwlnbfo', 1), ('plstfydx', 1), ('placement', 1), ('eszl', 1), ('nated', 1), ('examine', 1), ('nat', 1), ('alerted', 1), ('curhetyu', 1), ('openvas', 1), ('performing', 1), ('tarzana', 1), ('summaries', 1), ('occ', 1), ('fx', 1), ('sourcecode', 1), ('visid', 1), ('phpinfo', 1), ('xss', 1), ('chaning', 1), ('tray', 1), ('enabling', 1), ('preferred', 1), ('legitmate', 1), ('finial', 1), ('atjsv', 1), ('wxnetroc', 1), ('yecbmliq', 1), ('rohntyub', 1), ('dfhtyuison', 1), ('zstkagwu', 1), ('jlyrhdcf', 1), ('iehshelpdesk', 1), ('anteagroup', 1), ('hpormqtx', 1), ('roesshnktler', 1), ('rightly', 1), ('grkaqnzu', 1), ('mldekqpi', 1), ('hspgxeit', 1), ('prwiqjto', 1), ('diehfhyumj', 1), ('hdmwolxq', 1), ('xqbevoic', 1), ('bhdikthyu', 1), ('vvsimpj', 1), ('ehidbxam', 1), ('rnupoxeq', 1), ('licwu', 1), ('bmwtqxns', 1), ('fcrqkhix', 1), ('timing', 1), ('alexgnhtjunder', 1), ('answers', 1), ('abode', 1), ('lapping', 1), ('vfyuldps', 1), ('kjgcaphx', 1), ('lonin', 1), ('chatryung', 1), ('aficio', 1), ('lager', 1), ('darda', 1), ('telefonanlage', 1), ('nebenstelle', 1), ('umprogramdntymieren', 1), ('qnigujek', 1), ('kopqcjdh', 1), ('webshop', 1), ('firstly', 1), ('glimpse', 1), ('toolmforrun', 1), ('grouped', 1), ('competrhyrncy', 1), ('vijeghtyundra', 1), ('shwhdbthyuiethadri', 1), ('sanhjtyhru', 1), ('span', 1), ('cuytdmhv', 1), ('yeqdmwvt', 1), ('kimufghtyry', 1), ('kvwrbfet', 1), ('jrhoqdix', 1), ('ashley', 1), ('vic', 1), ('lonn', 1), ('ony', 1), ('todghtyud', 1), ('wodzrcjg', 1), ('crtd', 1), ('tipset', 1), ('jogt', 1), ('harrhntyl', 1), ('xlvuhjea', 1), ('mjugvtnd', 1), ('bookmarkhtys', 1), ('realm', 1), ('chaof', 1), ('qgwypesz', 1), ('fzsdnmrk', 1), ('zemboks', 1), ('soflex', 1), ('ripple', 1), ('braze', 1), ('signin', 1), ('nkqafhod', 1), ('xbvghozp', 1), ('engagement', 1), ('mexkspfc', 1), ('nocpyxaz', 1), ('autoforward', 1), ('susfhtyan', 1), ('malfhklouicki', 1), ('france', 1), ('uxhq', 1), ('wezeb', 1), ('controllers', 1), ('rpt', 1), ('rovfghesntine', 1), ('recruiting', 1), ('overpaying', 1), ('cordrtegd', 1), ('costarra', 1), ('vtdygauw', 1), ('wqxcrzhj', 1), ('costa', 1), ('santodde', 1), ('tlvwusmh', 1), ('dbwuyxoq', 1), ('eder', 1), ('santossdm', 1), ('coutidfrc', 1), ('cesarrogerio', 1), ('coutinho', 1), ('vetqkwpn', 1), ('qajtdobg', 1), ('santosdfd', 1), ('santoes', 1), ('serravdsa', 1), ('iltcxkvw', 1), ('dkwmxcgn', 1), ('eet', 1), ('vrtvpopc', 1), ('vvrnrtacri', 1), ('vvrtgffada', 1), ('vvrurtgsur', 1), ('vvrtyjakaa', 1), ('vvrtymitrd', 1), ('vvdeftmea', 1), ('vvbgrtyeleb', 1), ('vvcodgtjud', 1), ('vvggrthhibg', 1), ('qualtiy', 1), ('dslamtcb', 1), ('ezbmonjr', 1), ('qdapolnv', 1), ('jlcavxgi', 1), ('lentid', 1), ('minha', 1), ('fora', 1), ('mesmo', 1), ('sendo', 1), ('weekday', 1), ('venkbghksh', 1), ('pghjkanijkraj', 1), ('framdntywork', 1), ('istallation', 1), ('programdntymes', 1), ('spindle', 1), ('hxgayczerig', 1), ('xjmpacye', 1), ('qgxrptnf', 1), ('looged', 1), ('stil', 1), ('keghn', 1), ('zanegtyla', 1), ('pfxwuvce', 1), ('hcbmiqdp', 1), ('verbinde', 1), ('meinen', 1), ('netzlaufwerk', 1), ('apptc', 1), ('weg', 1), ('reverted', 1), ('wareneingang', 1), ('ripfghscp', 1), ('tutorial', 1), ('influenced', 1), ('tesscenter', 1), ('forever', 1), ('hjkyqecw', 1), ('ixdsbwoz', 1), ('sysetem', 1), ('xnzfsmue', 1), ('kwsazpeu', 1), ('ytu', 1), ('bfghabu', 1), ('skpye', 1), ('mghllenbecfnfk', 1), ('fhtyulvio', 1), ('abandon', 1), ('ekpl', 1), ('missrouting', 1), ('invoiuce', 1), ('protect', 1), ('zme', 1), ('neoarmgd', 1), ('meodvbxu', 1), ('xxxxx', 1), ('soedjitv', 1), ('wvprteja', 1), ('scherfgpd', 1), ('puxiomgy', 1), ('ndjorwab', 1), ('seated', 1), ('mds', 1), ('ziewxqof', 1), ('dartvis', 1), ('juni', 1), ('zahlen', 1), ('zeitkonten', 1), ('urlaubsst', 1), ('nde', 1), ('fehlen', 1), ('demonstrate', 1), ('intercompnay', 1), ('tower', 1), ('petrhyr', 1), ('hunt', 1), ('prototype', 1), ('annette', 1), ('pvtiqgsh', 1), ('orlzgfsx', 1), ('prtqc', 1), ('eror', 1), ('atached', 1), ('hanghdyle', 1), ('zdnqowag', 1), ('cdtyonhw', 1), ('tqvpohwj', 1), ('tbkywpqz', 1), ('jerghjemiah', 1), ('brock', 1), ('hrscc', 1), ('znq', 1), ('zns', 1), ('nesbfirjeerabhadrappa', 1), ('nesbfirj', 1), ('rjtnlocs', 1), ('fclswxkz', 1), ('tanrgty', 1), ('rqxmaindept', 1), ('scghhnell', 1), ('fgsmwvcp', 1), ('uoxkzwes', 1), ('problemas', 1), ('configura', 1), ('alteramdnty', 1), ('solicitadas', 1), ('dificultando', 1), ('assim', 1), ('cota', 1), ('ferramdntyentas', 1), ('especiais', 1), ('softland', 1), ('initiative', 1), ('cphlme', 1), ('cathytyma', 1), ('urls', 1), ('cpmaidhj', 1), ('yesilc', 1), ('zhengdr', 1), ('aenl', 1), ('vvkatts', 1), ('aspects', 1), ('comsumed', 1), ('marrthyu', 1), ('ekbl', 1), ('gbr', 1), ('ptyxefvk', 1), ('fhazbrwn', 1), ('loqdtrfn', 1), ('apxmsjkc', 1), ('holidays', 1), ('vig', 1), ('mxwibrtg', 1), ('qbsmonwv', 1), ('xxx', 1), ('rfvmeyho', 1), ('qgtxjsdc', 1), ('schmiede', 1), ('gvktqfrj', 1), ('grargtzzt', 1), ('asignment', 1), ('druckerzuordnung', 1), ('disponenten', 1), ('icnjlzas', 1), ('cvphuknj', 1), ('weil', 1), ('bergangsweise', 1), ('momentan', 1), ('zugordnet', 1), ('wdybmizf', 1), ('ekqgpaus', 1), ('verbindet', 1), ('axeclkro', 1), ('snfmerdb', 1), ('anmeldetaten', 1), ('eingeben', 1), ('fenster', 1), ('gelingt', 1), ('nochmal', 1), ('gleiches', 1), ('communicator', 1), ('ana', 1), ('pethrywrs', 1), ('vhihrty', 1), ('reassigning', 1), ('shaungtyr', 1), ('wky', 1), ('lefrte', 1), ('eafrtkin', 1), ('possble', 1), ('bise', 1), ('lance', 1), ('kappel', 1), ('templet', 1), ('tmyeqika', 1), ('galganski', 1), ('vdklzxqg', 1), ('jpaftdul', 1), ('infortype', 1), ('companys', 1), ('ckruf', 1), ('mitteleurop', 1), ('ischer', 1), ('neueinstellung', 1), ('vieles', 1), ('kroetzer', 1), ('backflush', 1), ('phil', 1), ('schoenfeld', 1), ('iotudrxg', 1), ('odpcwvez', 1), ('qscdktvl', 1), ('rihendxu', 1), ('worn', 1), ('luis', 1), ('revilla', 1), ('ltaballallcompanycm', 1), ('dtbycsgf', 1), ('vfdglqnp', 1), ('saztolpx', 1), ('xqgovpik', 1), ('querries', 1), ('strahlen', 1), ('blasting', 1), ('mtg', 1), ('forecasting', 1), ('rescheduled', 1), ('insure', 1), ('erstellung', 1), ('zeitnachweise', 1), ('aufgetreten', 1), ('ferienarbeiter', 1), ('randstad', 1), ('einschlie', 1), ('lich', 1), ('northgate', 1), ('gaining', 1), ('btelgpcx', 1), ('nrlfhbmu', 1), ('serverrfcserver', 1), ('serverindexserver', 1), ('servernameserver', 1), ('xfuqovkd', 1), ('efsdciut', 1), ('seefgrtybum', 1), ('atydjkwl', 1), ('sotmfcga', 1), ('ruchitgrr', 1), ('haug', 1), ('tsk', 1), ('beschreibung', 1), ('gesamtes', 1), ('verkaufsgebiet', 1), ('abgebildet', 1), ('urspr', 1), ('ngliche', 1), ('nachricht', 1), ('manage', 1), ('nbr', 1), ('sched', 1), ('sucking', 1), ('bandwidth', 1), ('pasue', 1), ('properties', 1), ('venkthrysh', 1), ('nmtekfrz', 1), ('tjxzeuqm', 1), ('wac', 1), ('letzten', 1), ('umfangsschleifmaschinen', 1), ('sparen', 1), ('designes', 1), ('scandinavia', 1), ('fortive', 1), ('fya', 1), ('telefons', 1), ('unsichtbar', 1), ('schlecht', 1), ('abzulesen', 1), ('follwing', 1), ('connoected', 1), ('wit', 1), ('pone', 1), ('delievery', 1), ('determination', 1), ('pradtheypxsuqgidj', 1), ('txlpcqsg', 1), ('khty', 1), ('saoltrmy', 1), ('xyuscbkn', 1), ('vvrttraja', 1), ('aetwpiox', 1), ('eijzadco', 1), ('shynhjundar', 1), ('tauogwvl', 1), ('xfvqakdw', 1), ('vvttraja', 1), ('wohtyugang', 1), ('ethryju', 1), ('researched', 1), ('atnh', 1), ('workfflow', 1), ('ek', 1), ('ckfragen', 1), ('lqjoagzt', 1), ('gqueiatx', 1), ('anmeldename', 1), ('crishtyutian', 1), ('pryes', 1), ('htvepyua', 1), ('izgulrcf', 1), ('hronovsky', 1), ('jctgwmyi', 1), ('morgens', 1), ('zeitbuchungen', 1), ('vorhanden', 1), ('ompeztak', 1), ('ilkpqtjh', 1), ('dort', 1), ('anmelde', 1), ('ging', 1), ('trat', 1), ('htburown', 1), ('hpkfjqyr', 1), ('hai', 1), ('srirgrtyam', 1), ('fleisrgtyk', 1), ('distribuators', 1), ('sogo', 1), ('skannen', 1), ('unterlagen', 1), ('einskannen', 1), ('rappel', 1), ('vous', 1), ('avez', 1), ('nouveau', 1), ('ingdirect', 1), ('publik', 1), ('zlgmctws', 1), ('khfjzyto', 1), ('ngprt', 1), ('dcgw', 1), ('hinweise', 1), ('existiert', 1), ('extr', 1), ('megfgthyhana', 1), ('pkzthgea', 1), ('kgvsdmpj', 1), ('observe', 1), ('kirgtyan', 1), ('taranga', 1), ('emerald', 1), ('tgynoqcs', 1), ('uxfyzrma', 1), ('nehtjuavathi', 1), ('patirjy', 1), ('vivian', 1), ('jecigpzw', 1), ('gqpmxwal', 1), ('xxxx', 1), ('exterior', 1), ('nhjpxoct', 1), ('ewngozhx', 1), ('mksysbalv', 1), ('axcl', 1), ('remediate', 1), ('opt', 1), ('erplv', 1), ('drwgs', 1), ('hartstoffe', 1), ('jokgacwd', 1), ('hdfcwust', 1), ('atache', 1), ('wjpncyef', 1), ('tspnaimc', 1), ('directs', 1), ('ljpgedia', 1), ('bzqcwsgf', 1), ('draftsight', 1), ('ukxtqfda', 1), ('qvtaykbg', 1), ('mrczxwje', 1), ('ocasryzq', 1), ('behebung', 1), ('folgendem', 1), ('aktiver', 1), ('hochgeladen', 1), ('analog', 1), ('adt', 1), ('tyco', 1), ('suffer', 1), ('forcing', 1), ('ofwxjriq', 1), ('rwcxkflq', 1), ('ebqdmgpk', 1), ('daoyrtmj', 1), ('dist', 1), ('ramdntya', 1), ('aurangabad', 1), ('maharashtra', 1), ('commited', 1), ('docks', 1), ('vahqkojb', 1), ('trlapeso', 1), ('beneficial', 1), ('sized', 1), ('fonts', 1), ('comparison', 1), ('lighter', 1), ('lpfzasmv', 1), ('cleoprzq', 1), ('guru', 1), ('prasath', 1), ('prarttsagj', 1), ('yqwuhzkv', 1), ('icvgkxnt', 1), ('qbnsrzlv', 1), ('gyqxkbae', 1), ('entsperrung', 1), ('benutzung', 1), ('passwortmanagers', 1), ('verwendet', 1), ('anruf', 1), ('musste', 1), ('angleichen', 1), ('telefonat', 1), ('erneutes', 1), ('leute', 1), ('ganzen', 1), ('verbringen', 1), ('berlegen', 1), ('bringt', 1), ('endlich', 1), ('firma', 1), ('effizienter', 1), ('erneute', 1), ('passworts', 1), ('weiterarbeiten', 1), ('navbrtheen', 1), ('gogtr', 1), ('departement', 1), ('hardness', 1), ('guprgttas', 1), ('maste', 1), ('rage', 1), ('grades', 1), ('aufstellung', 1), ('ordnerzugriff', 1), ('wtxvqngf', 1), ('nxjivlmr', 1), ('laufwerks', 1), ('betriebsrat', 1), ('stehenden', 1), ('unmittelbar', 1), ('zukommen', 1), ('betriebsratsvorsitzender', 1), ('altweiherstra', 1), ('stamping', 1), ('timings', 1), ('pulvermetalogy', 1), ('staeberoth', 1), ('absent', 1), ('felix', 1), ('ddeihrsh', 1), ('sigrtyhdeo', 1), ('dibesh', 1), ('dumps', 1), ('cartridge', 1), ('disabling', 1), ('kopierer', 1), ('ausbildungswerkstatt', 1), ('einzug', 1), ('servicedienst', 1), ('toolhones', 1), ('audible', 1), ('speake', 1), ('datasource', 1), ('yinnrty', 1), ('constraint', 1), ('gdwowner', 1), ('fk', 1), ('sbscrptn', 1), ('tblusers', 1), ('violated', 1), ('gdwp', 1), ('prefetch', 1), ('circuits', 1), ('latpop', 1), ('funktionen', 1), ('meldet', 1), ('falscher', 1), ('serverpfad', 1), ('biblotheken', 1), ('synchronisation', 1), ('lte', 1), ('fjtrnslb', 1), ('ejzkrchq', 1), ('door', 1), ('cti', 1), ('adrhtykins', 1), ('jadrhtykins', 1), ('vbmzgsdk', 1), ('jdmyazti', 1), ('confusing', 1), ('dirttwan', 1), ('toolperfect', 1), ('respected', 1), ('guidge', 1), ('perfect', 1), ('arcade', 1), ('tbbuyhexstandoff', 1), ('mws', 1), ('asxpnlgk', 1), ('mnktdsjq', 1), ('lesbar', 1), ('telefonaten', 1), ('mittendrin', 1), ('unterbrochen', 1), ('earnings', 1), ('zearn', 1), ('nikitha', 1), ('upadhyaya', 1), ('lqnoifve', 1), ('wvhelqxu', 1), ('conv', 1), ('phillpd', 1), ('suchfunktion', 1), ('verzeichnisses', 1), ('openstage', 1), ('besorgen', 1), ('rkyjnbqh', 1), ('kfshormi', 1), ('telefonisch', 1), ('erreichbar', 1), ('jintana', 1), ('eoreport', 1), ('timegraphfilters', 1), ('passwortmanager', 1), ('seitdem', 1), ('weder', 1), ('alten', 1), ('igurwxhv', 1), ('ughynofq', 1), ('outsouring', 1), ('caching', 1), ('accessibility', 1), ('mendmre', 1), ('istead', 1), ('costcenter', 1), ('requisitioners', 1), ('anticipating', 1), ('qwynjdbk', 1), ('eamnvwyh', 1), ('tgpvrbyi', 1), ('ztdxwpcn', 1), ('geehrter', 1), ('souzarft', 1), ('erneuten', 1), ('verbindungsherstellung', 1), ('nein', 1), ('versand', 1), ('logistik', 1), ('pcqobjndadditional', 1), ('probieren', 1), ('bescheid', 1), ('dose', 1), ('amrice', 1), ('cashpro', 1), ('staffs', 1), ('reinstallation', 1), ('lertfty', 1), ('zuothryrt', 1), ('kirtyrghwc', 1), ('ykjrbivs', 1), ('wiggrtgyis', 1), ('rtgyon', 1), ('chairman', 1), ('notebooks', 1), ('dind', 1), ('noteb', 1), ('recruit', 1), ('rcf', 1), ('recruiter', 1), ('cent', 1), ('rejections', 1), ('guardi', 1), ('gentileza', 1), ('utiliza', 1), ('transa', 1), ('pela', 1), ('fetch', 1), ('jcsmxrgl', 1), ('ibhsnket', 1), ('igkqpndy', 1), ('swqndxhl', 1), ('duffym', 1), ('magdalena', 1), ('rovsabyl', 1), ('idpvbjtw', 1), ('sidor', 1), ('hntubjela', 1), ('repoter', 1), ('srujan', 1), ('avsbdhyu', 1), ('sahryu', 1), ('worksheet', 1), ('simulating', 1), ('sinterleitstand', 1), ('beilageproben', 1), ('auswerten', 1), ('beilage', 1), ('tms', 1), ('proben', 1), ('cycles', 1), ('furnaces', 1), ('excell', 1), ('intouch', 1), ('laufwerke', 1), ('agvl', 1), ('gqbt', 1), ('qhvspezr', 1), ('fvluqczd', 1), ('nuerthytzg', 1), ('vou', 1), ('imprimir', 1), ('gia', 1), ('somente', 1), ('orgr', 1), ('salsed', 1), ('revers', 1), ('infrastruture', 1), ('wehlauerstr', 1), ('confrim', 1), ('aspap', 1), ('unplanned', 1), ('vicinity', 1), ('leadership', 1), ('bridge', 1), ('representatives', 1), ('bobs', 1), ('ufebvyzx', 1), ('gzahomlv', 1), ('sthqwdpj', 1), ('lpnigfyq', 1), ('glaube', 1), ('erneut', 1), ('ltiges', 1), ('verwenden', 1), ('paneer', 1), ('cuibfgna', 1), ('cbmqufoa', 1), ('berblick', 1), ('antwort', 1), ('gerade', 1), ('demselben', 1), ('vorgeschlafen', 1), ('distinti', 1), ('saluti', 1), ('receivers', 1), ('nuksytoh', 1), ('whovmtez', 1), ('sbcheyu', 1), ('synchro', 1), ('lumia', 1), ('schichtplanung', 1), ('sundaycommitted', 1), ('ste', 1), ('tpflxnhz', 1), ('bdjiosrp', 1), ('slides', 1), ('xkjuigsc', 1), ('nrzykspt', 1), ('bods', 1), ('messsgeraete', 1), ('alkoana', 1), ('datein', 1), ('gedureckt', 1), ('loin', 1), ('pax', 1), ('suggestion', 1), ('yandy', 1), ('pan', 1), ('procted', 1), ('proctected', 1), ('ope', 1), ('omokam', 1), ('mizumoto', 1), ('wfnbtpkg', 1), ('ixecamwrs', 1), ('marftgytins', 1), ('filtered', 1), ('simulation', 1), ('beacon', 1), ('overwhelmed', 1), ('affiliated', 1), ('situations', 1), ('zttf', 1), ('simulate', 1), ('bookmarkhty', 1), ('mov', 1), ('moviments', 1), ('nota', 1), ('moviment', 1), ('ufzxpadv', 1), ('hnxmotwu', 1), ('structured', 1), ('linkage', 1), ('reflects', 1), ('listing', 1), ('banners', 1), ('ertnhxkf', 1), ('gwjibhxm', 1), ('kicking', 1), ('fioghtna', 1), ('wightygins', 1), ('plaese', 1), ('corrective', 1), ('ywfhcuki', 1), ('dajkmxcl', 1), ('modeling', 1), ('majetkm', 1), ('mkjubdti', 1), ('fbusqrlt', 1), ('rodny', 1), ('fzsxgapt', 1), ('remapped', 1), ('actuall', 1), ('deliverable', 1), ('mcnerny', 1), ('lperi', 1), ('onjzqptl', 1), ('kgxmisbj', 1), ('chkmejsn', 1), ('lvidgknc', 1), ('bqyfwclo', 1), ('osjklifb', 1), ('teufeae', 1), ('upiyobvj', 1), ('lwohuizr', 1), ('kompletten', 1), ('selbst', 1), ('komplett', 1), ('wts', 1), ('crtgyerine', 1), ('vlymsnej', 1), ('whlqxcst', 1), ('ueywbzks', 1), ('gepstmfl', 1), ('zpd', 1), ('digits', 1), ('pesylifc', 1), ('wnyierbu', 1), ('pernr', 1), ('angepasst', 1), ('veranlassen', 1), ('gehabt', 1), ('zuvor', 1), ('neugestartet', 1), ('betriebsmittedatenbank', 1), ('contails', 1), ('contains', 1), ('montitor', 1), ('ftungssteuerung', 1), ('heizraum', 1), ('ssler', 1), ('christgrytian', 1), ('requited', 1), ('gydtvnlw', 1), ('miepcwzf', 1), ('angelique', 1), ('coetzk', 1), ('reinecker', 1), ('wahrscheinlich', 1), ('tonerwechsel', 1), ('keybankr', 1), ('documenttype', 1), ('backend', 1), ('usersettings', 1), ('teleservice', 1), ('angeschlossen', 1), ('bleiben', 1), ('beschprechungsraum', 1), ('accesible', 1), ('manger', 1), ('talagrtymr', 1), ('continiously', 1), ('bfckamsg', 1), ('tzdkbmfe', 1), ('che', 1), ('swisscom', 1), ('naa', 1), ('runter', 1), ('gefahren', 1), ('lorw', 1), ('unresolved', 1), ('occurences', 1), ('yqxlbswt', 1), ('eimhxowu', 1), ('tnhymatj', 1), ('ligsnzur', 1), ('smcxerwk', 1), ('grargtfl', 1), ('tony', 1), ('opetions', 1), ('sjv', 1), ('ejpvuxrg', 1), ('tryaibcx', 1), ('preeco', 1), ('cce', 1), ('kanchi', 1), ('excluding', 1), ('calculations', 1), ('wdkaoneh', 1), ('unqlarpk', 1), ('collective', 1), ('rub', 1), ('pzjelyxg', 1), ('vstyaouc', 1), ('quadra', 1), ('eaymvrzj', 1), ('bumzwtco', 1), ('nthing', 1), ('pressed', 1), ('reallocate', 1), ('ross', 1), ('signout', 1), ('tam', 1), ('aka', 1), ('pikosa', 1), ('xepcsrvh', 1), ('tbsokfyl', 1), ('hers', 1), ('succesful', 1), ('agr', 1), ('tskwevno', 1), ('sjhpoakl', 1), ('westcoast', 1), ('roadking', 1), ('expanding', 1), ('container', 1), ('fehlermeldungen', 1), ('ordnung', 1), ('kontrollieren', 1), ('sen', 1), ('tsrp', 1), ('aedzqlvj', 1), ('mkosyxgh', 1), ('pyrtfdxu', 1), ('nxfkqmoy', 1), ('whaley', 1), ('trhuymvb', 1), ('egpcwrkj', 1), ('utlization', 1), ('orientation', 1), ('stolen', 1), ('spengineering', 1), ('toolometer', 1), ('foundry', 1), ('analyzed', 1), ('posrting', 1), ('josh', 1), ('edc', 1), ('sahl', 1), ('sghtyhlp', 1), ('contactperson', 1), ('exemption', 1), ('clcking', 1), ('confirmationofhpcpo', 1), ('subdirectories', 1), ('showixepyfbga', 1), ('slno', 1), ('mfvkxghn', 1), ('mzjasxqd', 1), ('arbeitszeitplan', 1), ('rnibmcve', 1), ('xukajlvg', 1), ('sollarbeitszeit', 1), ('stunden', 1), ('arbeitstage', 1), ('internetconnection', 1), ('ted', 1), ('preciso', 1), ('entregar', 1), ('declara', 1), ('hitacni', 1), ('teamleiter', 1), ('grergtger', 1), ('dispo', 1), ('avfertigungszeiten', 1), ('arbeitsplaner', 1), ('arbeitssteuerer', 1), ('schneidplatten', 1), ('metaplasanlage', 1), ('xls', 1), ('krutjqwp', 1), ('qomksnhd', 1), ('mictbdhryhle', 1), ('burnhntyham', 1), ('ojrplsmx', 1), ('wslifbzc', 1), ('ksb', 1), ('kurz', 1), ('urvitans', 1), ('laqdwvgo', 1), ('karnos', 1), ('disturb', 1), ('markhtyet', 1), ('packen', 1), ('gaps', 1), ('indicators', 1), ('arised', 1), ('rslvwpnh', 1), ('emkfpqiy', 1), ('nliches', 1), ('adelhmk', 1), ('servermigration', 1), ('scghhnellstm', 1), ('seemor', 1), ('voreingestellte', 1), ('attachements', 1), ('sihtvocw', 1), ('yspnqxgw', 1), ('atuldhy', 1), ('upgrading', 1), ('printerscreen', 1), ('anh', 1), ('ngen', 1), ('speicher', 1), ('ffne', 1), ('pzybmcqd', 1), ('fxtemlyg', 1), ('andthyerh', 1), ('cvdebrc', 1), ('vvmathkag', 1), ('vanthyrdys', 1), ('tewgersy', 1), ('tgryudf', 1), ('recheck', 1), ('azovgeck', 1), ('zuwnxdbt', 1), ('construct', 1), ('reinigungsanlage', 1), ('carcau', 1), ('vlcexqpg', 1), ('vjrtqobx', 1), ('qrnusygw', 1), ('amiebrlf', 1), ('vvthuenka', 1), ('ofuhdesi', 1), ('rhbsawmf', 1), ('austausch', 1), ('kontaktieren', 1), ('eilig', 1), ('abteilung', 1), ('abstech', 1), ('betap', 1), ('betapdachform', 1), ('protected', 1), ('xzn', 1), ('worl', 1), ('purchaising', 1), ('accross', 1), ('comapny', 1), ('dataservices', 1), ('taskmgr', 1), ('getassignments', 1), ('deadlocked', 1), ('deadlock', 1), ('asid', 1), ('wwi', 1), ('monthy', 1), ('fileserver', 1), ('efdsm', 1), ('printserver', 1), ('symmetrix', 1), ('emc', 1), ('numerirtc', 1), ('conenct', 1), ('conection', 1), ('conferenced', 1), ('comcas', 1), ('vrjwyqtf', 1), ('qoxkapfw', 1), ('abd', 1), ('errata', 1), ('indicative', 1), ('quantify', 1), ('flaws', 1), ('liedzaft', 1), ('lvnbzktj', 1), ('vhsw', 1), ('hghtyther', 1), ('pollauridamary', 1), ('cobrgtool', 1), ('owns', 1), ('mktgen', 1), ('alluser', 1), ('analtyicspro', 1), ('mkt', 1), ('aero', 1), ('tzrekwqf', 1), ('homwadbs', 1), ('httpsys', 1), ('rce', 1), ('mapp', 1), ('denial', 1), ('unblocked', 1), ('singular', 1), ('patched', 1), ('hardened', 1), ('kernel', 1), ('ewll', 1), ('rqvl', 1), ('unaware', 1), ('extensive', 1), ('articles', 1), ('contributed', 1), ('rabbit', 1), ('pla', 1), ('earbud', 1), ('leaves', 1), ('patrcja', 1), ('szpilewska', 1), ('sicne', 1), ('surge', 1), ('ticked', 1), ('fernando', 1), ('fillipini', 1), ('fabio', 1), ('owdrqmit', 1), ('nhdzcuji', 1), ('gtdxpofz', 1), ('xnksbrwl', 1), ('vndwmyiz', 1), ('cjwqtzai', 1), ('privilege', 1), ('gpresult', 1), ('loghtml', 1), ('qdbmspxf', 1), ('nqdyiclk', 1), ('knowing', 1), ('cccplant', 1), ('pevokgiu', 1), ('hdywstbl', 1), ('idrizj', 1), ('wrapper', 1), ('moe', 1), ('mahtyurch', 1), ('kutgynka', 1), ('operate', 1), ('japanese', 1), ('lakhsynrhty', 1), ('ksp', 1), ('syatem', 1), ('nigktly', 1), ('waits', 1), ('ziped', 1), ('relazed', 1), ('buissness', 1), ('conact', 1), ('uncaught', 1), ('referenceerror', 1), ('xiframdntye', 1), ('definedsubmitform', 1), ('nahytua', 1), ('loan', 1), ('cover', 1), ('accidental', 1), ('gpts', 1), ('uasername', 1), ('ewseditor', 1), ('polling', 1), ('dsilvfgj', 1), ('hfm', 1), ('remain', 1), ('dummy', 1), ('letgyo', 1), ('leo', 1), ('uycravzn', 1), ('feqlznyg', 1), ('hadfiunr', 1), ('vupglewt', 1), ('anup', 1), ('bigdrtyh', 1), ('gavasane', 1), ('discritpion', 1), ('systemaccess', 1), ('applicaiton', 1), ('approvers', 1), ('frente', 1), ('visando', 1), ('avaliar', 1), ('furto', 1), ('betoneira', 1), ('aparecido', 1), ('trhsyvdur', 1), ('staszk', 1), ('qlzgbjck', 1), ('yzwnvbjt', 1), ('djskrgae', 1), ('dnckipwh', 1), ('watcher', 1), ('wolfthry', 1), ('kasphryer', 1), ('nasftgcijj', 1), ('rspqvzgu', 1), ('vroanwhu', 1), ('silvgtyar', 1), ('aliuytre', 1), ('love', 1), ('lewis', 1), ('brrgtyant', 1), ('handles', 1), ('resp', 1), ('schmidt', 1), ('renyhtuee', 1), ('meayhtger', 1), ('cor', 1), ('nmzfdlar', 1), ('uwdqtrnx', 1), ('uhntgvyj', 1), ('zwwirep', 1), ('bash', 1), ('shell', 1), ('vvspecmfrt', 1), ('rhgteini', 1), ('tbukjcyl', 1), ('lxncwqbj', 1), ('dwafrmth', 1), ('oabwzitv', 1), ('reichenberg', 1), ('philipp', 1), ('messmaschine', 1), ('papierstau', 1), ('cvqnstgu', 1), ('ofnimlwx', 1), ('romftguald', 1), ('companymet', 1), ('encl', 1), ('csmtowqn', 1), ('ulpjtgfo', 1), ('zitec', 1), ('whenn', 1), ('winrar', 1), ('sippprs', 1), ('baut', 1), ('einige', 1), ('wenige', 1), ('rechnern', 1), ('loginasanother', 1), ('layouts', 1), ('fcloseconnection', 1), ('loginasanotheruser', 1), ('empl', 1), ('erschien', 1), ('wusa', 1), ('anwendungsfehler', 1), ('tigung', 1), ('anwendung', 1), ('durchgef', 1), ('qmpyjfbn', 1), ('zlyiwtch', 1), ('zcyueotq', 1), ('ehvpaqnf', 1), ('empwx', 1), ('erneuert', 1), ('wrtyuh', 1), ('fufrtal', 1), ('gslpdhey', 1), ('ksiyurvlir', 1), ('trinzic', 1), ('prtpu', 1), ('hrsync', 1), ('oon', 1), ('tfgtodd', 1), ('panelfgt', 1), ('lwbchnga', 1), ('axpqctfr', 1), ('deposited', 1), ('kylfgte', 1), ('ptczqbdw', 1), ('ybaoluck', 1), ('telecomitalia', 1), ('rome', 1), ('shpnkgir', 1), ('mpsycbxl', 1), ('ldsm', 1), ('mgcivbtx', 1), ('bshmfxya', 1), ('suomfxpj', 1), ('izcwuvgo', 1), ('idocs', 1), ('pilot', 1), ('feedbacks', 1), ('fledge', 1), ('promote', 1), ('presume', 1), ('announced', 1), ('scot', 1), ('trask', 1), ('finalized', 1), ('exclude', 1), ('taneghrty', 1), ('zkea', 1), ('fills', 1), ('fell', 1), ('jha', 1), ('jhapg', 1), ('photo', 1), ('hinges', 1), ('seating', 1), ('lhqsl', 1), ('bdclient', 1), ('gwkdsmfx', 1), ('ntorypsd', 1), ('lhnw', 1), ('ragini', 1), ('wgtyills', 1), ('plvnuxmrils', 1), ('mesg', 1), ('disks', 1), ('drac', 1), ('companyprod', 1), ('usrr', 1), ('npvmwszt', 1), ('gzcpejxv', 1), ('rfvchzmp', 1), ('picjthkd', 1), ('bagtylleg', 1), ('zsluxctw', 1), ('ptirhcwv', 1), ('hpmwliog', 1), ('kqtnfvrl', 1), ('cyndy', 1), ('jose', 1), ('raising', 1), ('tiffrtany', 1), ('tafgtyng', 1), ('calander', 1), ('accede', 1), ('unrecognized', 1), ('swicth', 1), ('colin', 1), ('lpgw', 1), ('rfid', 1), ('precimat', 1), ('helpline', 1), ('obj', 1), ('multidetail', 1), ('nogui', 1), ('mshost', 1), ('jxlekivs', 1), ('fwakmztv', 1), ('lic', 1), ('gebucht', 1), ('annehmen', 1), ('uschow', 1), ('zslugaxq', 1), ('dde', 1), ('ausdrucken', 1), ('springt', 1), ('zwar', 1), ('girnda', 1), ('alm', 1), ('verstrauensstellung', 1), ('zwischen', 1), ('arbeitsstation', 1), ('prim', 1), ('ren', 1), ('rahmen', 1), ('geladen', 1), ('zeichnungsrahmen', 1), ('switching', 1), ('outloo', 1), ('jvxmzteb', 1), ('vsdcnfyr', 1), ('hellp', 1), ('szcbhvwe', 1), ('edpouqjl', 1), ('inivation', 1), ('frgtyetij', 1), ('bvcdpxrt', 1), ('qamyesuv', 1), ('npmzxbek', 1), ('julgttie', 1), ('pix', 1), ('ltrobe', 1), ('scanners', 1), ('scn', 1), ('cyxzfvtj', 1), ('yklmvqxf', 1), ('meldungen', 1), ('beshryuout', 1), ('momentarily', 1), ('dealing', 1), ('bjrtfeyi', 1), ('fuqapwtv', 1), ('latrosince', 1), ('jsut', 1), ('marocm', 1), ('cneter', 1), ('unannounced', 1), ('similarly', 1), ('festnetztelefon', 1), ('geraumer', 1), ('unvollst', 1), ('erkennbar', 1), ('waagerechte', 1), ('linien', 1), ('machen', 1), ('unlesbar', 1), ('lautet', 1), ('demage', 1), ('mechanics', 1), ('visits', 1), ('evaluation', 1), ('xrfcjkdl', 1), ('dtnzgkby', 1), ('vvsfgtyrinv', 1), ('albaney', 1), ('aggergrythator', 1), ('jilgtyq', 1), ('arguments', 1), ('kggsmgetstring', 1), ('soemec', 1), ('anyother', 1), ('vmhfteqo', 1), ('jpsfikow', 1), ('conformaclad', 1), ('sqlagent', 1), ('sqlservr', 1), ('tikona', 1), ('zmmdata', 1), ('vaghjmskee', 1), ('jpy', 1), ('conducting', 1), ('broadscan', 1), ('scrip', 1), ('pethrywrsburg', 1), ('eeec', 1), ('markhtye', 1), ('ting', 1), ('parti', 1), ('als', 1), ('companyme', 1), ('talipg', 1), ('mozill', 1), ('irefox', 1), ('nection', 1), ('ive', 1), ('ontent', 1), ('tex', 1), ('xdef', 1), ('vendorclassification', 1), ('logformat', 1), ('windowseventid', 1), ('logsource', 1), ('ontologystring', 1), ('einzelne', 1), ('tasten', 1), ('horst', 1), ('adress', 1), ('rpo', 1), ('betwenn', 1), ('cesvpmor', 1), ('azgtrbow', 1), ('mis', 1), ('cumbersome', 1), ('rabin', 1), ('middle', 1), ('visualize', 1), ('routings', 1), ('navarcm', 1), ('collemc', 1), ('geronca', 1), ('furlaf', 1), ('pintog', 1), ('restared', 1), ('edite', 1), ('deployed', 1), ('dyxrpmwo', 1), ('hcljzivn', 1), ('informing', 1), ('reschedule', 1), ('kowfthyuale', 1), ('ierfgayt', 1), ('alwjivqg', 1), ('postmaster', 1), ('tadeu', 1), ('wnyeczkb', 1), ('eqpjcukv', 1), ('reddfgymos', 1), ('dokumenten', 1), ('qnzmjxsl', 1), ('logsrwnb', 1), ('ruckruf', 1), ('zkgfcyvx', 1), ('sgxeatyb', 1), ('zssid', 1), ('uers', 1), ('busse', 1), ('dies', 1), ('schicken', 1), ('buero', 1), ('raghu', 1), ('zjcsqtdn', 1), ('jikyworg', 1), ('architecture', 1), ('sokdelfgty', 1), ('dmhpm', 1), ('cindy', 1), ('xia', 1), ('tfw', 1), ('dumovtpj', 1), ('ahgjsvoq', 1), ('abhanden', 1), ('gekommen', 1), ('zpdist', 1), ('distribute', 1), ('ebi', 1), ('debugging', 1), ('interrupts', 1), ('analyzing', 1), ('gebuchte', 1), ('anzahlungsrechnungen', 1), ('beigef', 1), ('wsignin', 1), ('jrilgbqu', 1), ('kbspjrod', 1), ('wpakylnj', 1), ('wdtsyuxg', 1), ('hannas', 1), ('meixni', 1), ('obqridjk', 1), ('ugelctsz', 1), ('babanlal', 1), ('synchronisiert', 1), ('stundenlang', 1), ('dadurch', 1), ('andere', 1), ('dinge', 1), ('geblockt', 1), ('sitze', 1), ('xjvubmlq', 1), ('vyamhjip', 1), ('duwvesim', 1), ('cixqbmfr', 1), ('narefgttndra', 1), ('shigthyuva', 1), ('rvsbtxue', 1), ('cdrwsymj', 1), ('sessions', 1), ('schneeberger', 1), ('schauen', 1), ('held', 1), ('worklfow', 1), ('bell', 1), ('health', 1), ('deduction', 1), ('elected', 1), ('kigthuym', 1), ('whjtyulen', 1), ('whjtlkn', 1), ('shrghyadja', 1), ('ejsxqmia', 1), ('cujvlrfq', 1), ('tif', 1), ('syslog', 1), ('ervin', 1), ('replicating', 1), ('budget', 1), ('dunham', 1), ('bradstreet', 1), ('feeder', 1), ('jghjimdghty', 1), ('bfhjtuiwell', 1), ('tem', 1), ('csi', 1), ('oops', 1), ('wbr', 1), ('kruse', 1), ('frthdyui', 1), ('geklappt', 1), ('lieben', 1), ('urgqkinl', 1), ('zpcokgbj', 1), ('reinaldo', 1), ('albrecht', 1), ('ewag', 1), ('dwnload', 1), ('rmdtqfxa', 1), ('fwpnqdxo', 1), ('bussiness', 1), ('governance', 1), ('seite', 1), ('angezeigt', 1), ('lizensiertes', 1), ('vnsmwqhb', 1), ('ogtpenjd', 1), ('produkte', 1), ('zeigen', 1), ('himself', 1), ('owenghyga', 1), ('ruf', 1), ('unsafe', 1), ('ihr', 1), ('skotthyutc', 1), ('skiped', 1), ('languague', 1), ('pws', 1), ('copyright', 1), ('corporation', 1), ('mdf', 1), ('tge', 1), ('sng', 1), ('attens', 1), ('jpix', 1), ('jp', 1), ('corpcare', 1), ('guests', 1), ('suitable', 1), ('stwpzxbf', 1), ('bjehirkx', 1), ('aenderungsantrag', 1), ('geloescht', 1), ('lizeciertes', 1), ('beckes', 1), ('wgyhktic', 1), ('zmmtaxupd', 1), ('crieria', 1), ('temporarlly', 1), ('irrespective', 1), ('exection', 1), ('unlicensed', 1), ('chaint', 1), ('kc', 1), ('jkmeusfq', 1), ('vjpckzsa', 1), ('danach', 1), ('carousel', 1), ('vnglqiht', 1), ('sebxvtdj', 1), ('proble', 1), ('existed', 1), ('sprache', 1), ('thehub', 1), ('englisch', 1), ('deutsch', 1), ('umstelle', 1), ('finde', 1), ('anklicke', 1), ('sagt', 1), ('aprtgghj', 1), ('messag', 1), ('jcmxerol', 1), ('nbfyczqr', 1), ('rdwpangu', 1), ('lybaxonw', 1), ('rolghtyuando', 1), ('santolgiy', 1), ('carlos', 1), ('idelcia', 1), ('almeida', 1), ('nascimento', 1), ('uvrwikmy', 1), ('yusexirn', 1), ('pasting', 1), ('importing', 1), ('judi', 1), ('elituytt', 1), ('nopr', 1), ('alexandre', 1), ('pinto', 1), ('alexandfrre', 1), ('pintfgtyo', 1), ('vertiayhtu', 1), ('russoddfac', 1), ('nmcxfrij', 1), ('hgaxtqmy', 1), ('pintoddsa', 1), ('blancog', 1), ('gargtcia', 1), ('ybplwrez', 1), ('lqcyehbf', 1), ('bxtqducs', 1), ('zuhoylts', 1), ('cegtcil', 1), ('fmw', 1), ('lewicki', 1), ('restricting', 1), ('offices', 1), ('recognizing', 1), ('rec', 1), ('forcast', 1), ('dale', 1), ('odfimpjg', 1), ('ptfoiake', 1), ('einstein', 1), ('ctuodmai', 1), ('vguwqjtd', 1), ('oppo', 1), ('duel', 1), ('refinery', 1), ('withdraw', 1), ('consumption', 1), ('discounted', 1), ('posten', 1), ('suffering', 1), ('bottleneck', 1), ('disprove', 1), ('iowa', 1), ('minnesotta', 1), ('dakota', 1), ('executncqulao', 1), ('qauighdpss', 1), ('itslpwra', 1), ('vybdkuoa', 1), ('yimwfntl', 1), ('rkdwohas', 1), ('fro', 1), ('tbloeczi', 1), ('gxlmeyph', 1), ('discussion', 1), ('shifted', 1), ('luciano', 1), ('amadeu', 1), ('iaxyjkrz', 1), ('pctnvdrm', 1), ('eulalla', 1), ('silvaes', 1), ('basket', 1), ('enhancement', 1), ('ihmbfeoy', 1), ('exbgcfsk', 1), ('enhance', 1), ('eplan', 1), ('vault', 1), ('juvfghtla', 1), ('kahtuithra', 1), ('corretly', 1), ('azerbaijan', 1), ('svuxjkpg', 1), ('tpurnjvi', 1), ('xovczlad', 1), ('fkicawph', 1), ('storno', 1), ('severeal', 1), ('bzekndcu', 1), ('ivhnpdbu', 1), ('sandstrahlger', 1), ('setuplaptop', 1), ('lautsprecher', 1), ('mikrofon', 1), ('mobilteil', 1), ('kratzen', 1), ('zeitweise', 1), ('allocation', 1), ('propose', 1), ('facilitate', 1), ('itelephony', 1), ('softwarei', 1), ('mngr', 1), ('infrastrcture', 1), ('jivp', 1), ('jivc', 1), ('chandmt', 1), ('sww', 1), ('pssh', 1), ('ypuaejsc', 1), ('yoxrqtsn', 1), ('walfgtkek', 1), ('tagsyrhu', 1), ('spinning', 1), ('vaugtyghtl', 1), ('ausliefern', 1), ('totalteamsales', 1), ('eternal', 1), ('rebalancing', 1), ('dinkifgtrl', 1), ('dinfgrtyukins', 1), ('noname', 1), ('findings', 1), ('surprised', 1), ('fengapac', 1), ('prohgtyarb', 1), ('includes', 1), ('dramdntyatically', 1), ('decreases', 1), ('accuracy', 1), ('rqxaudix', 1), ('gshn', 1), ('zpress', 1), ('sdnemlwy', 1), ('dsbmgonk', 1), ('aufgebaut', 1), ('apl', 1), ('juchaomy', 1), ('gaxrulwo', 1), ('umziehen', 1), ('ehem', 1), ('ventilst', 1), ('kontrolle', 1), ('whalep', 1), ('eozqgims', 1), ('rbmosifh', 1), ('incompatibility', 1), ('knovel', 1), ('ldil', 1), ('funktionert', 1), ('gruesse', 1), ('plantronic', 1), ('beshryuwire', 1), ('thecomputer', 1), ('stempelzeiten', 1), ('stelligen', 1), ('personalnummern', 1), ('leihmitarbeiter', 1), ('andre', 1), ('stempelt', 1), ('seiner', 1), ('karte', 1), ('zeitpunkt', 1), ('stimmt', 1), ('karten', 1), ('stammnummer', 1), ('ersten', 1), ('achtstelligen', 1), ('daran', 1), ('iszaguwe', 1), ('bdfzamjs', 1), ('zuvjqgwa', 1), ('joacrhfz', 1), ('ctrbjusz', 1), ('zahllauf', 1), ('lafgturie', 1), ('sherwtgyu', 1), ('populates', 1), ('orgs', 1), ('romania', 1), ('slovakia', 1), ('sicherheitsupdate', 1), ('verbinung', 1), ('maertosv', 1), ('soarewer', 1), ('alvesdss', 1), ('kgcw', 1), ('ziqmkgcw', 1), ('machasssg', 1), ('puxqyesm', 1), ('xqathyuz', 1), ('gislei', 1), ('machado', 1), ('peredrfifj', 1), ('schidrftas', 1), ('vieiresddr', 1), ('albussdqp', 1), ('larsffar', 1), ('rafaelm', 1), ('lara', 1), ('aiuknwzj', 1), ('nbsjzkqa', 1), ('ribewddwic', 1), ('carmsswot', 1), ('garcisdwr', 1), ('rodrigo', 1), ('fernansdes', 1), ('rodrigofernandes', 1), ('olivesswc', 1), ('gcbrdkzl', 1), ('oamkcufr', 1), ('cassio', 1), ('placisddwd', 1), ('grateful', 1), ('kisp', 1), ('iis', 1), ('ziehe', 1), ('mertut', 1), ('peilerk', 1), ('krdvgzeh', 1), ('yboasemp', 1), ('stoebtrt', 1), ('sxhcapoe', 1), ('kbluefip', 1), ('gsnuhpji', 1), ('qpyfctwl', 1), ('ausgedruckt', 1), ('chkzbeav', 1), ('ykeilmog', 1), ('ylhptzmd', 1), ('owslfzqi', 1), ('wphqnxly', 1), ('htvrbxmd', 1), ('pifyudbo', 1), ('tagsfbny', 1), ('seraching', 1), ('aofnvyzt', 1), ('daria', 1), ('afterit', 1), ('spoken', 1), ('dveuglzp', 1), ('mqetjxwp', 1), ('melbourne', 1), ('rowville', 1), ('notworking', 1), ('qad', 1), ('ciruit', 1), ('thermal', 1), ('mvkpjdfi', 1), ('guexyfpk', 1), ('bsopzxhi', 1), ('irfhcgzq', 1), ('xzjlkfvc', 1), ('agfmpyhr', 1), ('wanrtygm', 1), ('olivgtyera', 1), ('olivgtyemc', 1), ('displaced', 1), ('magda', 1), ('republish', 1), ('permition', 1), ('esias', 1), ('sharepiont', 1), ('cpbzkrel', 1), ('qualys', 1), ('implement', 1), ('hoavdlwc', 1), ('ungksotp', 1), ('kishore', 1), ('nikam', 1), ('shahid', 1), ('nawab', 1), ('qualsys', 1), ('widtqkwzup', 1), ('vtkzuqly', 1), ('tinning', 1), ('badging', 1), ('ykhsqvnu', 1), ('amfgtyartya', 1), ('gofgrthyuetz', 1), ('unsaved', 1), ('tfrbwoua', 1), ('aegpkruc', 1), ('maintaining', 1), ('kagthrl', 1), ('heyinz', 1), ('mcfgtydonn', 1), ('postal', 1), ('elogic', 1), ('lucinda', 1), ('decommission', 1), ('tvmuzqio', 1), ('bhsmdxgz', 1), ('wyjsbzda', 1), ('yfeuhtib', 1), ('sadghryiosh', 1), ('karagtfma', 1), ('commodities', 1), ('unbekannten', 1), ('fredi', 1), ('stury', 1), ('ruemlang', 1), ('spt', 1), ('mrptype', 1), ('extends', 1), ('defaulted', 1), ('matgrp', 1), ('uvoyrbhp', 1), ('qbupdjhw', 1), ('mfp', 1), ('replaces', 1), ('dotnetframdntyework', 1), ('prioritized', 1), ('oppening', 1), ('immidiately', 1), ('rebuilt', 1), ('metavis', 1), ('promts', 1), ('thnsguzj', 1), ('utwijzag', 1), ('yucgfmiq', 1), ('jamgpnqe', 1), ('setun', 1), ('luesebrink', 1), ('uhammet', 1), ('kuluz', 1), ('rohlings', 1), ('njdrcagt', 1), ('shourxyp', 1), ('contv', 1), ('suche', 1), ('giumxwvh', 1), ('lfvwjtin', 1), ('versendeten', 1), ('vorschl', 1), ('ckinfo', 1), ('pdujfybc', 1), ('edww', 1), ('passwordmanage', 1), ('lhqx', 1), ('warn', 1), ('sandfield', 1), ('describes', 1), ('exlude', 1), ('chopamghy', 1), ('milyhyakrp', 1), ('meinerseits', 1), ('cip', 1), ('tfedp', 1), ('tups', 1), ('kargthythik', 1), ('zxdtskpw', 1), ('kzybxher', 1), ('prjaswhe', 1), ('cmm', 1), ('nkademwy', 1), ('ihsepkwz', 1), ('deposit', 1), ('zor', 1), ('agains', 1), ('zf', 1), ('dbsgicet', 1), ('xzopjhlq', 1), ('vvgtyhpej', 1), ('peghyurozich', 1), ('jpsgeghtyui', 1), ('bdf', 1), ('nathyuasha', 1), ('smoltelephony', 1), ('snmp', 1), ('rabkypet', 1), ('zocjdutp', 1), ('fothrmijm', 1), ('ltmoubvy', 1), ('utrimobs', 1), ('gomeshthyru', 1), ('beshryued', 1), ('distorted', 1), ('complex', 1), ('macros', 1), ('sam', 1), ('lughjm', 1), ('nature', 1), ('recommends', 1), ('reliable', 1), ('portable', 1), ('hacker', 1), ('acache', 1), ('jin', 1), ('rong', 1), ('hong', 1), ('kong', 1), ('hkg', 1), ('apacnet', 1), ('hongkong', 1), ('bev', 1), ('loughner', 1), ('ghiklyop', 1), ('nikszpeu', 1), ('buzzing', 1), ('ztnfhiwq', 1), ('njpwxmdi', 1), ('iba', 1), ('ez', 1), ('asuenpyg', 1), ('vzmneycx', 1), ('nologin', 1), ('unexpected', 1), ('falls', 1), ('apparent', 1), ('succeeds', 1), ('sdxjiwlq', 1), ('ynowzqfh', 1), ('instal', 1), ('nwbhomqe', 1), ('ejavblzu', 1), ('eagclhome', 1), ('martif', 1), ('insertapps', 1), ('uterqfld', 1), ('ufmtgndo', 1), ('ldpequhm', 1), ('nqclatbw', 1), ('campbell', 1), ('ceramdntyic', 1), ('specs', 1), ('dana', 1), ('luxembourg', 1), ('enco', 1), ('immex', 1), ('dmc', 1), ('corporacion', 1), ('srl', 1), ('kaufsfthyman', 1), ('teyldpkw', 1), ('kbnfxpsy', 1), ('gehxzayq', 1), ('sua', 1), ('foi', 1), ('recebida', 1), ('por', 1), ('mais', 1), ('dos', 1), ('enc', 1), ('curso', 1), ('cls', 1), ('poss', 1), ('vel', 1), ('encontrar', 1), ('seguintes', 1), ('tente', 1), ('enviar', 1), ('novamente', 1), ('tarde', 1), ('contate', 1), ('rede', 1), ('opera', 1), ('cliente', 1), ('falhou', 1), ('recive', 1), ('westes', 1), ('betriebliche', 1), ('regelungen', 1), ('anlage', 1), ('bonus', 1), ('outloock', 1), ('infelizmente', 1), ('encontrou', 1), ('impedindo', 1), ('funcionar', 1), ('corretamente', 1), ('resultado', 1), ('dever', 1), ('fechado', 1), ('gostaria', 1), ('fiz', 1), ('ssemos', 1), ('bot', 1), ('fechar', 1), ('ajuda', 1), ('grouping', 1), ('expand', 1), ('arrows', 1), ('ldism', 1), ('claapdico', 1), ('pathname', 1), ('techcenter', 1), ('unsure', 1), ('offsite', 1), ('charges', 1), ('words', 1), ('overwrite', 1), ('restores', 1), ('revisar', 1), ('rfv', 1), ('usadtto', 1), ('dfsdpor', 1), ('ffthhiago', 1), ('frsilva', 1), ('usar', 1), ('atualizar', 1), ('grupos', 1), ('executar', 1), ('defrag', 1), ('nova', 1), ('vers', 1), ('pacote', 1), ('hxgayczear', 1), ('acessos', 1), ('itau', 1), ('caixa', 1), ('abertura', 1), ('plaghynilhas', 1), ('seep', 1), ('dctviemg', 1), ('bahbrgy', 1), ('wptbgchj', 1), ('jutpdcqf', 1), ('recruited', 1), ('reside', 1), ('cadb', 1), ('imply', 1), ('utility', 1), ('kyocera', 1), ('rxpjomyf', 1), ('hvolsgqn', 1), ('pnwbkitv', 1), ('phbnwmkl', 1), ('wyighrjl', 1), ('xcwavhyu', 1), ('beosjgxt', 1), ('mdevcqjk', 1), ('eaglt', 1), ('forum', 1), ('roedel', 1), ('realpresence', 1), ('destop', 1), ('kjtqxroc', 1), ('qyndvmlw', 1), ('imcvznow', 1), ('desktops', 1), ('tvmlrwkz', 1), ('rsxftjep', 1), ('nachstehenden', 1), ('schriftverkehr', 1), ('scghhnellstens', 1), ('aryndruh', 1), ('wjbtlxry', 1), ('gdbqzjyw', 1), ('telephonic', 1), ('ctask', 1), ('hghjnlabel', 1), ('dehnfyru', 1), ('uvjpaeli', 1), ('bnphqsxr', 1), ('orshop', 1), ('unnessary', 1), ('pmqansex', 1), ('nvihmbwc', 1), ('gzwasqoc', 1), ('gadisyxr', 1), ('coshopfloor', 1), ('hdbnameserver', 1), ('hdbxsengine', 1), ('hdbpreprocessor', 1), ('hdbcompileserve', 1), ('hdbindexserver', 1), ('zdis', 1), ('hourly', 1), ('desks', 1), ('zjihgovn', 1), ('ihlsmzdn', 1), ('cnhqgzwt', 1), ('ewgihcnz', 1), ('vdjqoeip', 1), ('moxnqszg', 1), ('zgdckste', 1), ('debtgyrur', 1), ('fzwxitmen', 1), ('jwvuasft', 1), ('usaing', 1), ('publication', 1), ('hooked', 1), ('microscope', 1), ('bsod', 1), ('cnljsmat', 1), ('ocxjvdnz', 1), ('opzuciql', 1), ('muedfkhz', 1), ('rabhtuikurtyar', 1), ('tammineni', 1), ('znet', 1), ('oahmgpfy', 1), ('mitctdrhb', 1), ('ricagthyr', 1), ('doflefne', 1), ('noggtyuerp', 1), ('ptgwfymc', 1), ('mchpwvgf', 1), ('gprs', 1), ('cgqjxtkf', 1), ('soewtuvf', 1), ('ginemkl', 1), ('kevguind', 1), ('gineman', 1), ('rolls', 1), ('qeue', 1), ('csvlijud', 1), ('jzhnkclo', 1), ('rrsp', 1), ('authenticated', 1), ('shagfferon', 1), ('bregtnnl', 1), ('awhile', 1), ('lehl', 1), ('ingreso', 1), ('puedo', 1), ('ingresar', 1), ('contrase', 1), ('coatncqulao', 1), ('qauighdpchine', 1), ('bqdlegnp', 1), ('lnphmsco', 1), ('bfrgtonersp', 1), ('gdec', 1), ('mailer', 1), ('wirftejas', 1), ('wiejas', 1), ('defekte', 1), ('sektoren', 1), ('oqxdecus', 1), ('encxjoawjr', 1), ('netwo', 1), ('suraj', 1), ('betshdy', 1), ('zxvjsipd', 1), ('jbzmgyvd', 1), ('spins', 1), ('portuguese', 1), ('nqtjsbad', 1), ('jfxeoudc', 1), ('roanoke', 1), ('esaqztby', 1), ('mhnbqiyc', 1), ('arrangement', 1), ('thinclients', 1), ('serverteam', 1), ('mstnjfai', 1), ('xcobykhl', 1), ('provigjtyswkb', 1), ('dpvaymxrest', 1), ('vikrhtyas', 1), ('kart', 1), ('oprn', 1), ('reprot', 1), ('subscriptions', 1), ('remained', 1), ('maintains', 1), ('doubled', 1), ('investment', 1), ('antrag', 1), ('anbei', 1), ('investmentantrag', 1), ('weit', 1), ('diesem', 1), ('thema', 1), ('anwendungstechniker', 1), ('nimmt', 1), ('bisschen', 1), ('erfasse', 1), ('aqstdryv', 1), ('flbnyqzc', 1), ('zadnryuinudin', 1), ('rer', 1), ('apparat', 1), ('einlasten', 1), ('practicing', 1), ('habits', 1), ('allways', 1), ('untinstall', 1), ('gortyhlia', 1), ('heating', 1), ('kgarnzdo', 1), ('nkiopevt', 1), ('argentina', 1), ('kashfyujqti', 1), ('instan', 1), ('proplems', 1), ('sype', 1), ('analyser', 1), ('designation', 1), ('aztlkeifowndararajan', 1), ('bvlcarfe', 1), ('antigvjx', 1), ('zekluqim', 1), ('gopi', 1), ('elevated', 1), ('orjcgtyz', 1), ('worylufs', 1), ('xuqvaobxuy', 1), ('ntqkuocz', 1), ('jacfgtykson', 1), ('ckflmqoj', 1), ('fojkrlmw', 1), ('aykegsvr', 1), ('kelli', 1), ('bactelephony', 1), ('softwarea', 1), ('rudfgbens', 1), ('rtwjunior', 1), ('juniowsrr', 1), ('glzshbja', 1), ('aoehpltm', 1), ('alexansxcddre', 1), ('olovxcdeira', 1), ('olivesadia', 1), ('qbtvmhau', 1), ('zowemnca', 1), ('robsdgerp', 1), ('writfxsq', 1), ('nwmaxpts', 1), ('wlzqaivr', 1), ('bigleman', 1), ('labor', 1), ('jrigdbox', 1), ('bgyluoqn', 1), ('beahleb', 1), ('gowzv', 1), ('hxgayczes', 1), ('enviroment', 1), ('uwiqchfp', 1), ('hnsukjma', 1), ('savers', 1), ('led', 1), ('myhzrtsi', 1), ('rwnhqiyv', 1), ('urgapyzt', 1), ('vaghyliort', 1), ('wesley', 1), ('tomlin', 1), ('xvmjocfn', 1), ('bqxcdfiz', 1), ('verboncouer', 1), ('verboma', 1), ('yepifgbl', 1), ('chefgtnp', 1), ('mason', 1), ('bengtjamin', 1), ('masonb', 1), ('mys', 1), ('ptmjvysi', 1), ('vkrepcybwa', 1), ('trmhfxoz', 1), ('bxofhryg', 1), ('talryhtir', 1), ('tayloml', 1), ('wiksufty', 1), ('zazrtulds', 1), ('carmer', 1), ('cardfrmeca', 1), ('povich', 1), ('trtgoy', 1), ('povictcfgt', 1), ('magtyrtijc', 1), ('kzeqbica', 1), ('kzcjeiyd', 1), ('santrhyat', 1), ('addressing', 1), ('wrking', 1), ('vlinspectkiosk', 1), ('eccqa', 1), ('apis', 1), ('foundation', 1), ('businesslogic', 1), ('businesslogicexception', 1), ('init', 1), ('silently', 1), ('httpexception', 1), ('paramdntys', 1), ('dataconnectors', 1), ('customersearchinputtype', 1), ('commoninput', 1), ('commoninputs', 1), ('distchannel', 1), ('maxrows', 1), ('soldto', 1), ('attributecode', 1), ('attributetext', 1), ('attributetype', 1), ('customersearch', 1), ('countrylong', 1), ('district', 1), ('houseno', 1), ('statelong', 1), ('zipcode', 1), ('customerno', 1), ('customertype', 1), ('partnerrole', 1), ('webserviceclient', 1), ('demo', 1), ('postpone', 1), ('broad', 1), ('band', 1), ('disassociated', 1), ('wlc', 1), ('timestamp', 1), ('rvdwyapu', 1), ('fubjamlr', 1), ('spoof', 1), ('ucosgrfy', 1), ('aeithcvp', 1), ('rxuobtjg', 1), ('grcmqaxd', 1), ('glnfyoqe', 1), ('fexliuytreu', 1), ('hqbxstoy', 1), ('mdjftxli', 1), ('zarlgjes', 1), ('qfrntose', 1), ('ivnhumzj', 1), ('formatar', 1), ('micro', 1), ('abrechnungs', 1), ('hmovlkyq', 1), ('kinawsdv', 1), ('iechuoxb', 1), ('zcejmwsq', 1), ('lobby', 1), ('pathryu', 1), ('etasthon', 1), ('robdyhr', 1), ('nvamcrpq', 1), ('gkrlmxne', 1), ('pat', 1), ('ludwidjfft', 1), ('mijhmijhmiles', 1), ('leader', 1), ('pobleme', 1), ('langsamer', 1), ('fung', 1), ('logn', 1), ('kbdljsxf', 1), ('kcmqtjgf', 1), ('graphic', 1), ('margins', 1), ('fehlversuche', 1), ('resided', 1), ('johghajknnes', 1), ('wildschuetz', 1), ('sicherheisdatenbank', 1), ('computerkonto', 1), ('arbeitsstationsvertrauensstellung', 1), ('zlxcsqdg', 1), ('ckpojwir', 1), ('wilfert', 1), ('ainl', 1), ('gabryltkla', 1), ('christgryta', 1), ('occationally', 1), ('laoding', 1), ('ung', 1), ('verwendung', 1), ('auftragsausgang', 1), ('zmcp', 1), ('lynerwjgthy', 1), ('lapels', 1), ('platz', 1), ('obvyknzx', 1), ('gzvjtish', 1), ('catgyhilp', 1), ('coatea', 1), ('whatlgp', 1), ('nerreter', 1), ('sandrgru', 1), ('asfgthok', 1), ('topefd', 1), ('uxndyfrs', 1), ('visfgthal', 1), ('vvjodav', 1), ('goptijdtnsya', 1), ('atcbvglq', 1), ('bdvmuszt', 1), ('budhtya', 1), ('phase', 1), ('vas', 1), ('undone', 1), ('ccep', 1), ('rhoades', 1), ('cfzsajbe', 1), ('lyejkdho', 1), ('beenefits', 1), ('pulls', 1), ('tgbtyim', 1), ('dgtalone', 1), ('beyklcmj', 1), ('bgfmrltw', 1), ('stehdgty', 1), ('jfhying', 1), ('decision', 1), ('zcustgrp', 1), ('eagvusbr', 1), ('nguqityl', 1), ('spamming', 1), ('individuals', 1), ('payable', 1), ('hugcadrn', 1), ('ixhlwdgt', 1), ('ralfteimp', 1), ('alto', 1), ('dotnet', 1), ('uninstaller', 1), ('cptl', 1), ('sop', 1), ('pasgryo', 1), ('knethyen', 1), ('grechduy', 1), ('visitble', 1), ('avglmrts', 1), ('vhqmtiua', 1), ('tifpdchb', 1), ('pedxruyf', 1), ('utilities', 1), ('drawers', 1), ('adjustment', 1), ('mehreren', 1), ('verschiedene', 1), ('prgramdntyme', 1)]
Out[0]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fd7e1e81c18>

In [0]:
def get_top_ticketdesc_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
In [0]:
plt.figure(figsize=(10,5))
top_bigrams=get_top_ticketdesc_bigrams(dfTicketngrams['EnglishDescription'])[:20]
x,y=map(list,zip(*top_bigrams))
sns.barplot(x=y,y=x)
Out[0]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fd7e35ab898>

In [0]:
def get_top_ticketdesc_trigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
In [0]:
plt.figure(figsize=(10,5))
top_trigrams=get_top_ticketdesc_trigrams(dfTicketngrams['EnglishDescription'])[:20]
x,y=map(list,zip(*top_trigrams))
sns.barplot(x=y,y=x)
Out[0]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fd7e3716f28>

Create Corpus
In [0]:
def listToString(s):
  str1 = " "
  for ele in s:
    str1 = str1 + " " + ele
  return str1
In [0]:
def create_corpus(df):
    corpus=[]
    for desc in tqdm(df['LemmaString'].astype(str)):
        words=[word.lower() for word in word_tokenize(desc) if((word.isalpha()==1))] # & (word not in stop_words))]
        corpus.append(words)
    return corpus
In [0]:
dfTicketngrams['LemmaString'] = dfTicketngrams['LemmaWords'].apply(lambda x: listToString(x))

corpus=create_corpus(dfTicketngrams)
100%|██████████| 8500/8500 [00:00<00:00, 9873.53it/s]
Model Building
Approach 1 ( Classification Approach )
Classification of 75 groups

Train and test Split
In [0]:
df1= dfTicketngrams[['LemmaString','Assignment group']]
df1.head()
Out[0]:
LemmaString	Assignment group
0	login verify detail employee manager check advise caller confirm resolve	GRP_0
1	outlook hmjdrvpb komuaywn team meeting skype appear calendar somebody advise correct kind	GRP_0
2	log vpn eylqgodm ybqkwiam not	GRP_0
3	hr	GRP_0
4	skype	GRP_0
In [0]:
X_train = df1.loc[:6000, 'LemmaString'].values
y_train = df1.loc[:6000, 'Assignment group'].values
X_test = df1.loc[6001:, 'LemmaString'].values
y_test = df1.loc[6001:, 'Assignment group'].values
In [0]:
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)
(6001, 9959) (2499, 9959)
Model
Naive Bayes
In [0]:
clf = MultinomialNB().fit(train_vectors, y_train)
In [0]:
predicted = clf.predict(test_vectors)
print(accuracy_score(y_test,predicted))
0.49459783913565425
In [0]:
print(predicted[1])
GRP_0
In [0]:
cm = confusion_matrix(y_test, predicted) 
cm
Out[0]:
array([[1131,    0,    0, ...,    0,    3,    0],
       [   4,    0,    0, ...,    0,    4,    0],
       [  26,    0,    0, ...,    0,    0,    0],
       ...,
       [   1,    0,    0, ...,    0,    0,    0],
       [ 112,    0,    0, ...,    0,   81,    0],
       [  71,    0,    0, ...,    0,    0,    0]])
SVM Model
In [0]:
SVM = svm.SVC(C=1.0, kernel='linear', gamma='auto')
In [0]:
SVM.fit(train_vectors ,y_train)
Out[0]:
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
In [0]:
predictions = SVM.predict(test_vectors)
print("SVM Accuracy Score -> ",accuracy_score(predictions, y_test)*100)
SVM Accuracy Score ->  59.343737494997995
In [0]:
Lcm = confusion_matrix(y_test, predictions) 
print(Lcm)
[[1096    0    0 ...    0    3    0]
 [   2    0    0 ...    0    0    0]
 [  16    0    8 ...    0    0    0]
 ...
 [   1    0    0 ...    0    0    0]
 [  85    0    0 ...    0   97    0]
 [  64    0    0 ...    0    1    4]]
In [0]:
print(classification_report(y_test, predictions))
              precision    recall  f1-score   support

       GRP_0       0.57      0.97      0.72      1134
       GRP_1       0.00      0.00      0.00         9
      GRP_10       0.89      0.31      0.46        26
      GRP_11       0.00      0.00      0.00        13
      GRP_12       0.62      0.39      0.48        94
      GRP_13       0.40      0.39      0.40        41
      GRP_14       0.67      0.37      0.47        38
      GRP_15       0.00      0.00      0.00        12
      GRP_16       0.00      0.00      0.00        18
      GRP_17       0.90      1.00      0.95        19
      GRP_18       0.54      0.28      0.37        25
      GRP_19       0.55      0.09      0.15        68
       GRP_2       0.58      0.44      0.50        64
      GRP_20       0.00      0.00      0.00        13
      GRP_21       0.00      0.00      0.00         8
      GRP_22       0.00      0.00      0.00        12
      GRP_23       0.50      0.29      0.36         7
      GRP_24       0.78      0.82      0.80        84
      GRP_25       0.33      0.04      0.07        24
      GRP_26       0.00      0.00      0.00        10
      GRP_27       0.00      0.00      0.00        10
      GRP_28       0.00      0.00      0.00        12
      GRP_29       0.80      0.26      0.39        31
       GRP_3       0.30      0.11      0.16        56
      GRP_30       1.00      0.07      0.13        14
      GRP_31       0.00      0.00      0.00        18
      GRP_32       0.00      0.00      0.00         3
      GRP_33       0.71      0.14      0.24        35
      GRP_34       0.75      0.18      0.29        17
      GRP_36       0.00      0.00      0.00         1
      GRP_37       0.00      0.00      0.00         3
      GRP_38       0.00      0.00      0.00         1
      GRP_39       0.00      0.00      0.00         4
       GRP_4       1.00      0.11      0.20        37
      GRP_40       0.00      0.00      0.00        10
      GRP_41       1.00      0.22      0.36         9
      GRP_42       0.00      0.00      0.00         8
      GRP_43       0.00      0.00      0.00         2
      GRP_44       1.00      0.25      0.40         4
      GRP_45       0.00      0.00      0.00        14
      GRP_46       0.00      0.00      0.00         2
      GRP_47       0.00      0.00      0.00        18
      GRP_48       0.00      0.00      0.00         7
      GRP_49       0.00      0.00      0.00         3
       GRP_5       0.50      0.54      0.52        26
      GRP_50       0.00      0.00      0.00         7
      GRP_51       0.00      0.00      0.00         3
      GRP_52       0.00      0.00      0.00         5
      GRP_53       0.00      0.00      0.00         1
      GRP_55       0.00      0.00      0.00         2
      GRP_56       0.00      0.00      0.00         1
      GRP_57       0.00      0.00      0.00         1
      GRP_58       0.00      0.00      0.00         1
      GRP_59       0.00      0.00      0.00         1
       GRP_6       1.00      0.25      0.40        75
      GRP_60       0.00      0.00      0.00        18
      GRP_62       0.00      0.00      0.00        14
      GRP_63       0.00      0.00      0.00         1
      GRP_65       0.00      0.00      0.00         7
      GRP_66       0.00      0.00      0.00         3
      GRP_68       0.00      0.00      0.00         1
      GRP_69       0.00      0.00      0.00         1
       GRP_7       0.52      0.76      0.62        21
      GRP_70       0.00      0.00      0.00         1
      GRP_71       0.00      0.00      0.00         2
      GRP_72       0.00      0.00      0.00         2
      GRP_73       0.00      0.00      0.00         1
       GRP_8       0.73      0.50      0.59       195
       GRP_9       0.80      0.06      0.11        71

    accuracy                           0.59      2499
   macro avg       0.25      0.13      0.15      2499
weighted avg       0.56      0.59      0.51      2499

Grid Search CV
In [0]:
params_grid = [{'kernel': ['linear','rbf'], 'C': [10.0,1.0,0.1,0.01]}]

svm_model_pc = GridSearchCV(svm.SVC(probability=True, decision_function_shape='ovr'), params_grid,cv=5,  verbose=1)

svm_model_pc.fit(train_vectors, y_train)

svm_model_pc.score

# View the accuracy score
print('Best score for training data:', svm_model_pc.best_score_,"\n") 

# View the best parameters for the model found using grid search
print('Best C:',svm_model_pc.best_estimator_.C,"\n") 
print('Best Kernel:',svm_model_pc.best_estimator_.kernel,"\n")

final_model_pc = svm_model_pc.best_estimator_

y_pred = final_model_pc.predict(test_vectors)
Fitting 5 folds for each of 8 candidates, totalling 40 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  40 out of  40 | elapsed: 26.7min finished
Best score for training data: 0.6025661948376353 

Best C: 1.0 

Best Kernel: linear 

In [0]:
print("Training score for GridSearch SVM: %f" % final_model_pc.score(train_vectors, y_train))
print("Testing score for GridSearch SVM: %f" % final_model_pc.score(test_vectors, y_test))
print("\n")
Training score for GridSearch SVM: 0.768539
Testing score for GridSearch SVM: 0.593437


In [0]:
Lcm = confusion_matrix(y_test, y_pred) 
print(Lcm)
[[1096    0    0 ...    0    3    0]
 [   2    0    0 ...    0    0    0]
 [  16    0    8 ...    0    0    0]
 ...
 [   1    0    0 ...    0    0    0]
 [  85    0    0 ...    0   97    0]
 [  64    0    0 ...    0    1    4]]
In [0]:

Deep Learning
In [0]:
df1.head()
Out[0]:
LemmaString	Assignment group
0	login verify detail employee manager check advise caller confirm resolve	GRP_0
1	outlook hmjdrvpb komuaywn team meeting skype appear calendar somebody advise correct kind	GRP_0
2	log vpn eylqgodm ybqkwiam not	GRP_0
3	hr	GRP_0
4	skype	GRP_0
In [0]:
## TOTAL NUMBER OF WORDS USED IN EACH HEADLINE
df1['nb_words'] = df1.LemmaString.apply(lambda x: len(x.split()))

## TOTAL NUMBER OF UNIQUE WORDS USED IN EACH HEADLINE
df1['nb_unique_words'] = df1.LemmaString.apply(lambda x: len(set(x.split())))

## TOTAL NUMBER OF CHARACTERS USED IN EACH HEADLINE
df1['nb_char'] = df1.LemmaString.apply(lambda x: len(x))
In [0]:
df1.LemmaString
Out[0]:
0         login verify detail employee manager check advise caller confirm resolve                 
1         outlook hmjdrvpb komuaywn team meeting skype appear calendar somebody advise correct kind
2         log vpn eylqgodm ybqkwiam not                                                            
3         hr                                                                                       
4         skype                                                                                    
         ...                                                                                       
8495      email come mail avglmrts vhqmtiua receive advise                                         
8496      telephony software                                                                       
8497      window tifpdchb pedxruyf                                                                 
8498      machine funcionando utility finish drawer adjustment setting network                     
8499      mehreren lassen sich verschiedene prgramdntyme nicht ffnen bereich cnc                   
Name: LemmaString, Length: 8500, dtype: object
In [0]:
maxlen = df1['nb_words'].max()
print(maxlen)
395
In [0]:
max_features = 10000
embedding_size = 50
In [0]:
tokenizer_obj=Tokenizer(num_words=max_features)
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)
In [0]:
#tokenizer = Tokenizer(num_words=max_features)
#tokenizer_obj.fit_on_texts(corpus)
In [0]:
X = tokenizer_obj.texts_to_sequences(corpus)
X = pad_sequences(X, maxlen = maxlen)

df1['Assignment group'].astype(str)
label_encoder = preprocessing.LabelEncoder() 
# Encode labels in column 'species'. 
y = df1['Assignment group']= label_encoder.fit_transform(df1['Assignment group']) 
y = to_categorical(y, num_classes= 74)

print("Number of Samples:", len(X))
print("Number of Labels: ", len(y))
Number of Samples: 8500
Number of Labels:  8500
Get the Vocabulary size
In [0]:
word_index = tokenizer_obj.word_index
num_words = len(word_index)+1
print(num_words)
12743
Word Embedding
In [0]:
embeddings = {}
for o in open(project_path+'/glove.6B.50d.txt','r'):
    word = o.split(" ")[0]
    # print(word)
    embd = o.split(" ")[1:]
    embd = np.asarray(embd, dtype='float32')
    # print(embd)
    embeddings[word] = embd
Create a weight matrix for words in training docs
In [0]:
embedding_matrix = np.zeros((num_words, embedding_size))

for word, i in tokenizer_obj.word_index.items():
  if i >= max_features: continue
  embedding_vector = embeddings.get(word)
  if embedding_vector is not None:
     embedding_matrix[i] = embedding_vector

len(embeddings.values())
Out[0]:
400000
Create and Compile your Model
LSTM
In [0]:
model = Sequential()
model.add(Embedding(num_words, embedding_size, weights = [embedding_matrix]))
model.add(Bidirectional(LSTM(128, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(74, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:66: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:541: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4432: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:197: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:203: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:148: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3733: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/optimizers.py:793: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3576: The name tf.log is deprecated. Please use tf.math.log instead.

In [0]:
model.summary()
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 50)          637150    
_________________________________________________________________
bidirectional_1 (Bidirection (None, None, 256)         183296    
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 256)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 74)                19018     
=================================================================
Total params: 839,464
Trainable params: 839,464
Non-trainable params: 0
_________________________________________________________________
Fit your model
In [0]:
from keras.callbacks import EarlyStopping
history = model.fit(X,y,batch_size=100,epochs=50,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.

Train on 6800 samples, validate on 1700 samples
Epoch 1/50
6800/6800 [==============================] - 67s 10ms/step - loss: 2.5900 - acc: 0.4703 - val_loss: 2.4858 - val_acc: 0.4612
Epoch 2/50
6800/6800 [==============================] - 65s 10ms/step - loss: 2.2167 - acc: 0.4979 - val_loss: 2.2757 - val_acc: 0.4882
Epoch 3/50
6800/6800 [==============================] - 69s 10ms/step - loss: 2.0093 - acc: 0.5312 - val_loss: 2.1101 - val_acc: 0.5012
Epoch 4/50
6800/6800 [==============================] - 67s 10ms/step - loss: 1.8476 - acc: 0.5578 - val_loss: 2.0127 - val_acc: 0.5112
Epoch 5/50
6800/6800 [==============================] - 66s 10ms/step - loss: 1.7172 - acc: 0.5856 - val_loss: 1.9117 - val_acc: 0.5529
Epoch 6/50
6800/6800 [==============================] - 66s 10ms/step - loss: 1.5995 - acc: 0.6028 - val_loss: 1.8476 - val_acc: 0.5594
Epoch 7/50
6800/6800 [==============================] - 69s 10ms/step - loss: 1.4986 - acc: 0.6224 - val_loss: 1.8353 - val_acc: 0.5665
Epoch 8/50
6800/6800 [==============================] - 67s 10ms/step - loss: 1.4099 - acc: 0.6431 - val_loss: 1.7538 - val_acc: 0.5771
Epoch 9/50
6800/6800 [==============================] - 67s 10ms/step - loss: 1.3492 - acc: 0.6519 - val_loss: 1.7420 - val_acc: 0.5800
Epoch 10/50
6800/6800 [==============================] - 67s 10ms/step - loss: 1.2695 - acc: 0.6662 - val_loss: 1.7276 - val_acc: 0.5765
Epoch 11/50
6800/6800 [==============================] - 70s 10ms/step - loss: 1.1838 - acc: 0.6897 - val_loss: 1.6992 - val_acc: 0.5929
Epoch 12/50
6800/6800 [==============================] - 67s 10ms/step - loss: 1.1110 - acc: 0.7053 - val_loss: 1.7556 - val_acc: 0.5582
Plot Graph
In [0]:
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
fig.suptitle("Performance: LSTM Model")
ax1.plot(history.history['acc'])
ax1.plot(history.history['val_acc'])
cut = np.where(history.history['val_acc'] == np.max(history.history['val_acc']))[0][0]
ax1.axvline(x=cut, color='k', linestyle='--')
ax1.set_title("Model Accuracy")
ax1.legend(['train', 'test'])

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]
ax2.axvline(x=cut, color='k', linestyle='--')
ax2.set_title("Model Loss")
ax2.legend(['train', 'test'])
plt.show()

GRU
In [0]:
model = Sequential()
model.add(Embedding(num_words, embedding_size, weights = [embedding_matrix]))
model.add(Bidirectional(GRU(128, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(74, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
In [0]:
model.summary()
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, None, 50)          637150    
_________________________________________________________________
bidirectional_2 (Bidirection (None, None, 256)         137472    
_________________________________________________________________
global_max_pooling1d_2 (Glob (None, 256)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 74)                19018     
=================================================================
Total params: 793,640
Trainable params: 793,640
Non-trainable params: 0
_________________________________________________________________
Fit your model
In [0]:
from keras.callbacks import EarlyStopping
history = model.fit(X,y,batch_size=100,epochs=50,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
Train on 6800 samples, validate on 1700 samples
Epoch 1/50
6800/6800 [==============================] - 55s 8ms/step - loss: 2.5693 - acc: 0.4762 - val_loss: 2.3573 - val_acc: 0.4676
Epoch 2/50
6800/6800 [==============================] - 53s 8ms/step - loss: 2.1007 - acc: 0.5141 - val_loss: 2.1627 - val_acc: 0.4824
Epoch 3/50
6800/6800 [==============================] - 53s 8ms/step - loss: 1.9015 - acc: 0.5560 - val_loss: 2.0424 - val_acc: 0.5124
Epoch 4/50
6800/6800 [==============================] - 54s 8ms/step - loss: 1.7506 - acc: 0.5832 - val_loss: 1.9270 - val_acc: 0.5512
Epoch 5/50
6800/6800 [==============================] - 57s 8ms/step - loss: 1.6320 - acc: 0.5984 - val_loss: 1.8494 - val_acc: 0.5635
Epoch 6/50
6800/6800 [==============================] - 55s 8ms/step - loss: 1.5165 - acc: 0.6181 - val_loss: 1.7951 - val_acc: 0.5576
Epoch 7/50
6800/6800 [==============================] - 53s 8ms/step - loss: 1.4182 - acc: 0.6350 - val_loss: 1.7519 - val_acc: 0.5688
Epoch 8/50
6800/6800 [==============================] - 53s 8ms/step - loss: 1.3250 - acc: 0.6613 - val_loss: 1.7236 - val_acc: 0.5818
Epoch 9/50
6800/6800 [==============================] - 53s 8ms/step - loss: 1.2435 - acc: 0.6737 - val_loss: 1.6808 - val_acc: 0.5812
Epoch 10/50
6800/6800 [==============================] - 54s 8ms/step - loss: 1.1671 - acc: 0.6899 - val_loss: 1.6720 - val_acc: 0.5835
Epoch 11/50
6800/6800 [==============================] - 54s 8ms/step - loss: 1.0973 - acc: 0.7099 - val_loss: 1.6559 - val_acc: 0.5871
Epoch 12/50
6800/6800 [==============================] - 54s 8ms/step - loss: 1.0285 - acc: 0.7290 - val_loss: 1.6391 - val_acc: 0.5959
Epoch 13/50
6800/6800 [==============================] - 55s 8ms/step - loss: 0.9590 - acc: 0.7451 - val_loss: 1.6380 - val_acc: 0.5941
Epoch 14/50
6800/6800 [==============================] - 54s 8ms/step - loss: 0.9034 - acc: 0.7587 - val_loss: 1.6707 - val_acc: 0.5912
Plot Graph
In [0]:
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
fig.suptitle("Performance: GRU Model")
ax1.plot(history.history['acc'])
ax1.plot(history.history['val_acc'])
cut = np.where(history.history['val_acc'] == np.max(history.history['val_acc']))[0][0]
ax1.axvline(x=cut, color='k', linestyle='--')
ax1.set_title("Model Accuracy")
ax1.legend(['train', 'test'])

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]
ax2.axvline(x=cut, color='k', linestyle='--')
ax2.set_title("Model Loss")
ax2.legend(['train', 'test'])
plt.show()

Summary of Approach 1:
Currently, we have used Classical Machine Learning Model such as Naive Bayes, SVM, GridSearchCV on SVM model, LSTM and GRU models for said text classification.
As can be seen, the accuracy along with precision and recall for different classes is very low.
The reason for this low accuracy is mentioned below:

     Highly Imbalanced Dataset
     Proper Hyperparameter tuning may be required
     Proper Model selection may be required


We may try below steps to fine tune our model during our next Milestone.

  1. Fine tuning existing model with hyper parameters.
  2. Handling imbalance of classes if accuracy with step 1 is not achieved.
  3. Using a different approach altogether. Since the base of the problem talks about L1/L2/L3 groups, our first approach would be to divide the tickets to a larger group of L1/L2/L3 and then break it down into individual groups as given in the dataset.
Approach 2- Hyperparameter tuning on Deep Learning LSTM Model
As we can see, we are not seeing good results from above approach. We will perform Hyparameter tuning on our models.

In [0]:
batch_size = 32
epochs = 25
3 A. Stacked LSTM with 64 Neurons
In [0]:
# Define model for stacked LSTM with 64 units

def stacked_lstm64(X, y, num_class):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 24)

  main_input = Input(shape=(maxlen,))

  #em = Embedding(max_features, 100, input_length=maxlen) (main_input)
  em = Embedding(len(wordvectors.syn0), 100, weights = [wordvectors.vectors]) (main_input)

  lstm_out1 = LSTM(64, return_sequences = True)(em)
  lstm_out2 = LSTM(64)(lstm_out1)

  x = Dropout(0.4)(lstm_out2)

  main_output = Dense(num_class, activation = 'softmax')(x)

  model = Model(inputs = main_input, outputs = main_output)

  # compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  print('Model summary for stacked LSTM model with 64 units: \n', model.summary())

  #Fit the model
  model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, class_weight = 'auto') 

  # Print model for train and test data
  print('Trainig set accuracy: ', model.evaluate(X_train, y_train))
  print('Test set accuracy: ', model.evaluate(X_test, y_test))

  # Prediction on Test data
  y_pred = model.predict(X_test)
  y_pred_rd = np.round(y_pred)

  #Print Classification matrix
  print(classification_report(y_test, y_pred_rd))

  return y_pred_rd
In [0]:
y_pred64_rd = stacked_lstm64(X_train, y_train, num_class) # Run the stacked LSTM model with 64 units and print model summary along with classification report for the predicted target value
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:66: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:541: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4432: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:197: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:203: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:148: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3733: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/optimizers.py:793: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3576: The name tf.log is deprecated. Please use tf.math.log instead.

Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 150)               0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 150, 100)          2700      
_________________________________________________________________
lstm_1 (LSTM)                (None, 150, 64)           42240     
_________________________________________________________________
lstm_2 (LSTM)                (None, 64)                33024     
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 74)                4810      
=================================================================
Total params: 82,774
Trainable params: 82,774
Non-trainable params: 0
_________________________________________________________________
Model summary for stacked LSTM model with 64 units: 
 None
WARNING:tensorflow:From /tensorflow-1.15.0/python3.6/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.

Train on 4203 samples, validate on 1051 samples
Epoch 1/25
4203/4203 [==============================] - 60s 14ms/step - loss: 2.8946 - acc: 0.4268 - val_loss: 2.4927 - val_acc: 0.4567
Epoch 2/25
4203/4203 [==============================] - 57s 14ms/step - loss: 2.4935 - acc: 0.4875 - val_loss: 2.3165 - val_acc: 0.4976
Epoch 3/25
4203/4203 [==============================] - 57s 14ms/step - loss: 2.3459 - acc: 0.5034 - val_loss: 2.2658 - val_acc: 0.5147
Epoch 4/25
4203/4203 [==============================] - 57s 14ms/step - loss: 2.2688 - acc: 0.5075 - val_loss: 2.1731 - val_acc: 0.5157
Epoch 5/25
4203/4203 [==============================] - 57s 14ms/step - loss: 2.2162 - acc: 0.5106 - val_loss: 2.1422 - val_acc: 0.5119
Epoch 6/25
4203/4203 [==============================] - 57s 13ms/step - loss: 2.1758 - acc: 0.5175 - val_loss: 2.0972 - val_acc: 0.5195
Epoch 7/25
4203/4203 [==============================] - 57s 14ms/step - loss: 2.1203 - acc: 0.5161 - val_loss: 2.0733 - val_acc: 0.5195
Epoch 8/25
4203/4203 [==============================] - 58s 14ms/step - loss: 2.0981 - acc: 0.5206 - val_loss: 2.0468 - val_acc: 0.5195
Epoch 9/25
4203/4203 [==============================] - 57s 14ms/step - loss: 2.0663 - acc: 0.5246 - val_loss: 2.0188 - val_acc: 0.5262
Epoch 10/25
4203/4203 [==============================] - 57s 13ms/step - loss: 2.0389 - acc: 0.5291 - val_loss: 2.0000 - val_acc: 0.5271
Epoch 11/25
4203/4203 [==============================] - 57s 13ms/step - loss: 2.0154 - acc: 0.5291 - val_loss: 1.9889 - val_acc: 0.5271
Epoch 12/25
4203/4203 [==============================] - 57s 13ms/step - loss: 1.9957 - acc: 0.5325 - val_loss: 1.9745 - val_acc: 0.5281
Epoch 13/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.9876 - acc: 0.5349 - val_loss: 1.9664 - val_acc: 0.5300
Epoch 14/25
4203/4203 [==============================] - 57s 13ms/step - loss: 1.9634 - acc: 0.5320 - val_loss: 1.9676 - val_acc: 0.5338
Epoch 15/25
4203/4203 [==============================] - 57s 13ms/step - loss: 1.9499 - acc: 0.5353 - val_loss: 1.9526 - val_acc: 0.5328
Epoch 16/25
4203/4203 [==============================] - 56s 13ms/step - loss: 1.9231 - acc: 0.5382 - val_loss: 1.9434 - val_acc: 0.5357
Epoch 17/25
4203/4203 [==============================] - 56s 13ms/step - loss: 1.9042 - acc: 0.5429 - val_loss: 1.9430 - val_acc: 0.5423
Epoch 18/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.8991 - acc: 0.5418 - val_loss: 1.9258 - val_acc: 0.5442
Epoch 19/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.9009 - acc: 0.5375 - val_loss: 1.9318 - val_acc: 0.5338
Epoch 20/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.8837 - acc: 0.5413 - val_loss: 1.9435 - val_acc: 0.5328
Epoch 21/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.8741 - acc: 0.5399 - val_loss: 1.9275 - val_acc: 0.5347
Epoch 22/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.8526 - acc: 0.5472 - val_loss: 1.9212 - val_acc: 0.5404
Epoch 23/25
4203/4203 [==============================] - 56s 13ms/step - loss: 1.8483 - acc: 0.5432 - val_loss: 1.9242 - val_acc: 0.5376
Epoch 24/25
4203/4203 [==============================] - 57s 13ms/step - loss: 1.8274 - acc: 0.5468 - val_loss: 1.9466 - val_acc: 0.5366
Epoch 25/25
4203/4203 [==============================] - 56s 13ms/step - loss: 1.8302 - acc: 0.5458 - val_loss: 1.9169 - val_acc: 0.5357
5254/5254 [==============================] - 24s 4ms/step
Trainig set accuracy:  [1.7657753161073502, 0.551199086410354]
2253/2253 [==============================] - 10s 5ms/step
Test set accuracy:  [1.8517003019174576, 0.5481580116989028]
              precision    recall  f1-score   support

           0       0.80      0.76      0.78       989
           1       0.00      0.00      0.00         7
           2       0.00      0.00      0.00        43
           3       0.00      0.00      0.00         9
           4       0.57      0.27      0.37        84
           5       0.00      0.00      0.00        39
           6       0.00      0.00      0.00        33
           7       0.00      0.00      0.00         9
           8       0.00      0.00      0.00        31
           9       1.00      1.00      1.00        21
          10       0.00      0.00      0.00        25
          11       0.00      0.00      0.00        59
          12       0.45      0.07      0.13        67
          13       0.00      0.00      0.00        12
          14       0.00      0.00      0.00         7
          15       0.00      0.00      0.00        12
          16       0.00      0.00      0.00         8
          17       0.00      0.00      0.00        47
          18       0.00      0.00      0.00        28
          19       0.00      0.00      0.00        16
          20       0.00      0.00      0.00         5
          21       0.00      0.00      0.00        15
          22       0.00      0.00      0.00        25
          23       0.00      0.00      0.00        61
          24       0.00      0.00      0.00         3
          25       0.00      0.00      0.00         7
          26       0.00      0.00      0.00         0
          27       0.00      0.00      0.00        22
          28       0.00      0.00      0.00        20
          29       0.00      0.00      0.00         0
          30       0.00      0.00      0.00         4
          31       0.00      0.00      0.00         5
          32       0.00      0.00      0.00         1
          33       0.00      0.00      0.00         6
          34       0.00      0.00      0.00        31
          35       0.00      0.00      0.00         9
          36       0.00      0.00      0.00         9
          37       0.00      0.00      0.00        11
          38       0.00      0.00      0.00         2
          39       0.00      0.00      0.00         3
          40       0.00      0.00      0.00        14
          41       0.00      0.00      0.00         2
          42       0.00      0.00      0.00         9
          43       0.00      0.00      0.00         3
          44       0.00      0.00      0.00         1
          45       0.00      0.00      0.00        38
          46       0.00      0.00      0.00         3
          47       0.00      0.00      0.00         3
          48       0.00      0.00      0.00         1
          49       0.00      0.00      0.00         3
          50       0.00      0.00      0.00         0
          51       0.00      0.00      0.00         3
          52       0.00      0.00      0.00         1
          53       0.00      0.00      0.00         1
          54       0.00      0.00      0.00         0
          55       0.00      0.00      0.00         2
          56       0.00      0.00      0.00        64
          57       0.00      0.00      0.00         4
          58       0.00      0.00      0.00         0
          59       0.00      0.00      0.00         0
          60       0.00      0.00      0.00         1
          61       0.00      0.00      0.00         0
          62       0.00      0.00      0.00         2
          63       0.00      0.00      0.00         0
          64       0.00      0.00      0.00         0
          65       0.00      0.00      0.00         2
          66       0.00      0.00      0.00         0
          67       0.00      0.00      0.00        16
          68       0.00      0.00      0.00         1
          69       0.00      0.00      0.00         2
          70       0.00      0.00      0.00         0
          71       0.00      0.00      0.00         0
          72       0.74      0.44      0.55       220
          73       0.00      0.00      0.00        72

   micro avg       0.79      0.40      0.53      2253
   macro avg       0.05      0.03      0.04      2253
weighted avg       0.47      0.40      0.42      2253
 samples avg       0.40      0.40      0.40      2253

3 B. Stacked LSTM with 128 Neurons
In [0]:
# Define model for stacked LSTM with 128 units

def stacked_lstm(X, y, num_class):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 24)
  main_input = Input(shape=(maxlen,))

  em = Embedding(max_features, 100, input_length=maxlen) (main_input)

  lstm_out1 = LSTM(128, return_sequences = True)(em)
  lstm_out2 = LSTM(128)(lstm_out1)

  x = Dropout(0.4)(lstm_out2)

  main_output = Dense(num_class, activation = 'softmax')(x)

  model = Model(inputs = main_input, outputs = main_output)

  # compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  print('Model summary for stacked LSTM model: \n', model.summary())

  #Fit the model
  model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, class_weight = 'auto') #callbacks=[EarlyStopping(verbose=True, patience=5, monitor='acc')],

  # Print accuracy of the model on train and test data
  print('Trainig set accuracy: ', model.evaluate(X_train, y_train))
  print('Test set accuracy: ', model.evaluate(X_test, y_test))

  # Prediction on Test data
  y_pred = model.predict(X_test)
  y_pred_rd = np.round(y_pred)

  #Print Classification matrix
  print(classification_report(y_test, y_pred_rd))

  return y_pred_rd
In [0]:
y_pred_rd = stacked_lstm(X_train, y_train, num_class) # Run the stacked LSTM model with 128 units and print model summary along with classification report for the predicted target value
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 150)               0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 150, 100)          1000000   
_________________________________________________________________
lstm_3 (LSTM)                (None, 150, 128)          117248    
_________________________________________________________________
lstm_4 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 74)                9546      
=================================================================
Total params: 1,258,378
Trainable params: 1,258,378
Non-trainable params: 0
_________________________________________________________________
Model summary for stacked LSTM model: 
 None
Train on 4203 samples, validate on 1051 samples
Epoch 1/25
4203/4203 [==============================] - 58s 14ms/step - loss: 2.6849 - acc: 0.4566 - val_loss: 2.1728 - val_acc: 0.5119
Epoch 2/25
4203/4203 [==============================] - 57s 14ms/step - loss: 2.0941 - acc: 0.5196 - val_loss: 2.0554 - val_acc: 0.5319
Epoch 3/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.9031 - acc: 0.5346 - val_loss: 2.0410 - val_acc: 0.5281
Epoch 4/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.7591 - acc: 0.5570 - val_loss: 2.0522 - val_acc: 0.5271
Epoch 5/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.5972 - acc: 0.5917 - val_loss: 2.0461 - val_acc: 0.5195
Epoch 6/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.4557 - acc: 0.6129 - val_loss: 2.1450 - val_acc: 0.5128
Epoch 7/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.3326 - acc: 0.6398 - val_loss: 2.1514 - val_acc: 0.4995
Epoch 8/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.2148 - acc: 0.6695 - val_loss: 2.2363 - val_acc: 0.5167
Epoch 9/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.1031 - acc: 0.6962 - val_loss: 2.2462 - val_acc: 0.5100
Epoch 10/25
4203/4203 [==============================] - 57s 14ms/step - loss: 1.0252 - acc: 0.7147 - val_loss: 2.3164 - val_acc: 0.5081
Epoch 11/25
4203/4203 [==============================] - 57s 13ms/step - loss: 0.9306 - acc: 0.7330 - val_loss: 2.3439 - val_acc: 0.4938
Epoch 12/25
4203/4203 [==============================] - 57s 14ms/step - loss: 0.8485 - acc: 0.7468 - val_loss: 2.4203 - val_acc: 0.5186
Epoch 13/25
4203/4203 [==============================] - 57s 14ms/step - loss: 0.7863 - acc: 0.7737 - val_loss: 2.4972 - val_acc: 0.5214
Epoch 14/25
4203/4203 [==============================] - 57s 14ms/step - loss: 0.6865 - acc: 0.8039 - val_loss: 2.5722 - val_acc: 0.4910
Epoch 15/25
4203/4203 [==============================] - 57s 14ms/step - loss: 0.6540 - acc: 0.8147 - val_loss: 2.5493 - val_acc: 0.5138
Epoch 16/25
4203/4203 [==============================] - 57s 14ms/step - loss: 0.5717 - acc: 0.8311 - val_loss: 2.5923 - val_acc: 0.5186
Epoch 17/25
4203/4203 [==============================] - 57s 13ms/step - loss: 0.5489 - acc: 0.8482 - val_loss: 2.5743 - val_acc: 0.5138
Epoch 18/25
4203/4203 [==============================] - 57s 13ms/step - loss: 0.5018 - acc: 0.8556 - val_loss: 2.7037 - val_acc: 0.5128
Epoch 19/25
4203/4203 [==============================] - 57s 14ms/step - loss: 0.4740 - acc: 0.8660 - val_loss: 2.7597 - val_acc: 0.5119
Epoch 20/25
4203/4203 [==============================] - 58s 14ms/step - loss: 0.4175 - acc: 0.8872 - val_loss: 2.7726 - val_acc: 0.5433
Epoch 21/25
4203/4203 [==============================] - 57s 14ms/step - loss: 0.4011 - acc: 0.8870 - val_loss: 2.8120 - val_acc: 0.5271
Epoch 22/25
4203/4203 [==============================] - 58s 14ms/step - loss: 0.3857 - acc: 0.8832 - val_loss: 2.7831 - val_acc: 0.5366
Epoch 23/25
4203/4203 [==============================] - 59s 14ms/step - loss: 0.3714 - acc: 0.8951 - val_loss: 2.9314 - val_acc: 0.5109
Epoch 24/25
4203/4203 [==============================] - 58s 14ms/step - loss: 0.3553 - acc: 0.8979 - val_loss: 2.9214 - val_acc: 0.5290
Epoch 25/25
4203/4203 [==============================] - 58s 14ms/step - loss: 0.3196 - acc: 0.9074 - val_loss: 2.9322 - val_acc: 0.5319
5254/5254 [==============================] - 25s 5ms/step
Trainig set accuracy:  [0.8044471308719445, 0.8431671107500553]
2253/2253 [==============================] - 11s 5ms/step
Test set accuracy:  [2.823585946984255, 0.5428317800078242]
              precision    recall  f1-score   support

           0       0.78      0.78      0.78       989
           1       0.00      0.00      0.00         7
           2       0.38      0.49      0.43        43
           3       0.00      0.00      0.00         9
           4       0.64      0.43      0.51        84
           5       0.77      0.26      0.38        39
           6       0.26      0.27      0.27        33
           7       0.00      0.00      0.00         9
           8       0.00      0.00      0.00        31
           9       0.95      0.86      0.90        21
          10       0.56      0.20      0.29        25
          11       0.12      0.10      0.11        59
          12       0.37      0.19      0.25        67
          13       0.09      0.08      0.09        12
          14       0.06      0.14      0.08         7
          15       0.50      0.08      0.14        12
          16       0.50      0.25      0.33         8
          17       0.79      0.72      0.76        47
          18       0.15      0.14      0.15        28
          19       0.00      0.00      0.00        16
          20       0.00      0.00      0.00         5
          21       0.00      0.00      0.00        15
          22       0.33      0.08      0.13        25
          23       0.23      0.23      0.23        61
          24       0.00      0.00      0.00         3
          25       0.00      0.00      0.00         7
          26       0.00      0.00      0.00         0
          27       0.22      0.09      0.13        22
          28       0.00      0.00      0.00        20
          29       0.00      0.00      0.00         0
          30       0.00      0.00      0.00         4
          31       0.00      0.00      0.00         5
          32       0.00      0.00      0.00         1
          33       0.00      0.00      0.00         6
          34       0.16      0.16      0.16        31
          35       0.00      0.00      0.00         9
          36       0.08      0.11      0.09         9
          37       0.00      0.00      0.00        11
          38       0.00      0.00      0.00         2
          39       0.00      0.00      0.00         3
          40       0.11      0.14      0.12        14
          41       0.00      0.00      0.00         2
          42       0.00      0.00      0.00         9
          43       0.00      0.00      0.00         3
          44       0.00      0.00      0.00         1
          45       0.76      0.34      0.47        38
          46       0.00      0.00      0.00         3
          47       0.00      0.00      0.00         3
          48       0.00      0.00      0.00         1
          49       0.00      0.00      0.00         3
          50       0.00      0.00      0.00         0
          51       0.00      0.00      0.00         3
          52       0.00      0.00      0.00         1
          53       0.00      0.00      0.00         1
          54       0.00      0.00      0.00         0
          55       0.00      0.00      0.00         2
          56       0.67      0.28      0.40        64
          57       0.50      0.25      0.33         4
          58       0.00      0.00      0.00         0
          59       0.00      0.00      0.00         0
          60       0.00      0.00      0.00         1
          61       0.00      0.00      0.00         0
          62       0.00      0.00      0.00         2
          63       0.00      0.00      0.00         0
          64       0.00      0.00      0.00         0
          65       0.00      0.00      0.00         2
          66       0.00      0.00      0.00         0
          67       0.25      0.25      0.25        16
          68       0.00      0.00      0.00         1
          69       0.00      0.00      0.00         2
          70       0.00      0.00      0.00         0
          71       0.00      0.00      0.00         0
          72       0.79      0.54      0.64       220
          73       0.25      0.08      0.12        72

   micro avg       0.62      0.50      0.55      2253
   macro avg       0.15      0.10      0.12      2253
weighted avg       0.58      0.50      0.52      2253
 samples avg       0.50      0.50      0.50      2253

3 C. Stacked LSTM with 128 Neurons and L2 regularization
In [0]:
# Define model for stacked LSTM with 128 units and L2 regularization 

def stacked_lstm_l2(X, y, num_class):
    main_input_l2 = Input(shape=(maxlen,))

    em_l2 = Embedding(max_features, 100, input_length=maxlen) (main_input_l2)


    lstm_out1_l2 = LSTM(128, return_sequences = True)(em_l2)
    lstm_out2_l2 = LSTM(128)(lstm_out1_l2)

    x_l2 = Dropout(0.4)(lstm_out2_l2)

    main_output_l2 = Dense(num_class, activation = 'softmax', kernel_regularizer=l2(0.01))(x_l2)

    model_l2 = Model(inputs = main_input_l2, outputs = main_output_l2)

    # compile the model
    model_l2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('Model summary for stacked LSTM model with L2 regularization: \n', model_l2.summary())

    #Fit the model
    model_l2.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, class_weight = 'auto') #callbacks=[EarlyStopping(verbose=True, patience=5, monitor='acc')],

    print('Trainig set accuracy: ', model_l2.evaluate(X_train, y_train))
    print('Test set accuracy: ', model_l2.evaluate(X_test, y_test))

    # Prediction on Test data
    y_pred_l2 = model_l2.predict(X_test)
    y_pred_l2_rd = np.round(y_pred_l2)

    #Print Classification matrix
    print(classification_report(y_test, y_pred_l2_rd))

    return y_pred_l2_rd
In [0]:
y_pred_l2_rd = stacked_lstm_l2(X_train, y_train, num_class) # Run the stacked LSTM model with 128 units and L2 regularization and print model summary along with classification report for the predicted target value
Model: "model_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         (None, 150)               0         
_________________________________________________________________
embedding_3 (Embedding)      (None, 150, 100)          1000000   
_________________________________________________________________
lstm_5 (LSTM)                (None, 150, 128)          117248    
_________________________________________________________________
lstm_6 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 74)                9546      
=================================================================
Total params: 1,258,378
Trainable params: 1,258,378
Non-trainable params: 0
_________________________________________________________________
Model summary for stacked LSTM model with L2 regularization: 
 None
Train on 4203 samples, validate on 1051 samples
Epoch 1/25
4203/4203 [==============================] - 61s 14ms/step - loss: 3.3772 - acc: 0.4440 - val_loss: 2.6398 - val_acc: 0.5138
Epoch 2/25
4203/4203 [==============================] - 58s 14ms/step - loss: 2.4140 - acc: 0.5161 - val_loss: 2.3003 - val_acc: 0.5224
Epoch 3/25
4203/4203 [==============================] - 58s 14ms/step - loss: 2.0962 - acc: 0.5330 - val_loss: 2.2207 - val_acc: 0.5347
Epoch 4/25
4203/4203 [==============================] - 59s 14ms/step - loss: 1.9360 - acc: 0.5477 - val_loss: 2.2048 - val_acc: 0.5090
Epoch 5/25
4203/4203 [==============================] - 59s 14ms/step - loss: 1.8112 - acc: 0.5667 - val_loss: 2.2793 - val_acc: 0.4976
Epoch 6/25
4203/4203 [==============================] - 59s 14ms/step - loss: 1.7315 - acc: 0.5808 - val_loss: 2.3361 - val_acc: 0.5100
Epoch 7/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.6401 - acc: 0.6015 - val_loss: 2.3263 - val_acc: 0.5138
Epoch 8/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.5518 - acc: 0.6246 - val_loss: 2.2687 - val_acc: 0.5052
Epoch 9/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.4752 - acc: 0.6462 - val_loss: 2.2989 - val_acc: 0.5043
Epoch 10/25
4203/4203 [==============================] - 59s 14ms/step - loss: 1.3921 - acc: 0.6576 - val_loss: 2.3670 - val_acc: 0.4795
Epoch 11/25
4203/4203 [==============================] - 59s 14ms/step - loss: 1.3379 - acc: 0.6783 - val_loss: 2.3764 - val_acc: 0.4805
Epoch 12/25
4203/4203 [==============================] - 59s 14ms/step - loss: 1.2842 - acc: 0.6950 - val_loss: 2.3977 - val_acc: 0.5052
Epoch 13/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.2286 - acc: 0.7126 - val_loss: 2.4113 - val_acc: 0.4919
Epoch 14/25
4203/4203 [==============================] - 59s 14ms/step - loss: 1.1870 - acc: 0.7278 - val_loss: 2.4678 - val_acc: 0.5214
Epoch 15/25
4203/4203 [==============================] - 59s 14ms/step - loss: 1.1673 - acc: 0.7340 - val_loss: 2.4992 - val_acc: 0.4824
Epoch 16/25
4203/4203 [==============================] - 59s 14ms/step - loss: 1.1343 - acc: 0.7457 - val_loss: 2.4345 - val_acc: 0.5109
Epoch 17/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.0717 - acc: 0.7637 - val_loss: 2.4670 - val_acc: 0.5062
Epoch 18/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.0260 - acc: 0.7813 - val_loss: 2.5110 - val_acc: 0.5071
Epoch 19/25
4203/4203 [==============================] - 58s 14ms/step - loss: 1.0007 - acc: 0.7951 - val_loss: 2.4993 - val_acc: 0.5024
Epoch 20/25
4203/4203 [==============================] - 58s 14ms/step - loss: 0.9733 - acc: 0.7990 - val_loss: 2.5216 - val_acc: 0.5024
Epoch 21/25
4203/4203 [==============================] - 59s 14ms/step - loss: 0.9452 - acc: 0.8092 - val_loss: 2.4592 - val_acc: 0.5052
Epoch 22/25
4203/4203 [==============================] - 58s 14ms/step - loss: 0.9185 - acc: 0.8208 - val_loss: 2.5229 - val_acc: 0.4910
Epoch 23/25
4203/4203 [==============================] - 58s 14ms/step - loss: 0.8886 - acc: 0.8325 - val_loss: 2.5281 - val_acc: 0.5119
Epoch 24/25
4203/4203 [==============================] - 58s 14ms/step - loss: 0.8684 - acc: 0.8425 - val_loss: 2.5234 - val_acc: 0.5052
Epoch 25/25
4203/4203 [==============================] - 58s 14ms/step - loss: 0.8634 - acc: 0.8413 - val_loss: 2.5334 - val_acc: 0.5081
5254/5254 [==============================] - 25s 5ms/step
Trainig set accuracy:  [1.1452475181313853, 0.7841644461135879]
2253/2253 [==============================] - 11s 5ms/step
Test set accuracy:  [2.4294953092277924, 0.5268530849345885]
              precision    recall  f1-score   support

           0       0.81      0.75      0.77       989
           1       0.00      0.00      0.00         7
           2       0.59      0.30      0.40        43
           3       0.00      0.00      0.00         9
           4       0.67      0.31      0.42        84
           5       0.23      0.08      0.12        39
           6       0.25      0.18      0.21        33
           7       0.00      0.00      0.00         9
           8       0.25      0.03      0.06        31
           9       0.95      0.86      0.90        21
          10       0.38      0.24      0.29        25
          11       0.11      0.08      0.10        59
          12       0.37      0.28      0.32        67
          13       0.00      0.00      0.00        12
          14       0.00      0.00      0.00         7
          15       0.00      0.00      0.00        12
          16       0.00      0.00      0.00         8
          17       0.88      0.60      0.71        47
          18       0.00      0.00      0.00        28
          19       0.00      0.00      0.00        16
          20       0.00      0.00      0.00         5
          21       0.00      0.00      0.00        15
          22       0.13      0.08      0.10        25
          23       0.27      0.15      0.19        61
          24       0.00      0.00      0.00         3
          25       0.00      0.00      0.00         7
          26       0.00      0.00      0.00         0
          27       0.17      0.05      0.07        22
          28       0.00      0.00      0.00        20
          29       0.00      0.00      0.00         0
          30       0.00      0.00      0.00         4
          31       0.00      0.00      0.00         5
          32       0.00      0.00      0.00         1
          33       0.00      0.00      0.00         6
          34       0.27      0.13      0.17        31
          35       0.00      0.00      0.00         9
          36       0.00      0.00      0.00         9
          37       0.00      0.00      0.00        11
          38       0.00      0.00      0.00         2
          39       0.00      0.00      0.00         3
          40       0.00      0.00      0.00        14
          41       0.00      0.00      0.00         2
          42       0.00      0.00      0.00         9
          43       0.00      0.00      0.00         3
          44       0.00      0.00      0.00         1
          45       0.79      0.39      0.53        38
          46       0.00      0.00      0.00         3
          47       0.00      0.00      0.00         3
          48       0.00      0.00      0.00         1
          49       0.00      0.00      0.00         3
          50       0.00      0.00      0.00         0
          51       0.00      0.00      0.00         3
          52       0.00      0.00      0.00         1
          53       0.00      0.00      0.00         1
          54       0.00      0.00      0.00         0
          55       0.00      0.00      0.00         2
          56       0.72      0.28      0.40        64
          57       0.00      0.00      0.00         4
          58       0.00      0.00      0.00         0
          59       0.00      0.00      0.00         0
          60       0.00      0.00      0.00         1
          61       0.00      0.00      0.00         0
          62       0.00      0.00      0.00         2
          63       0.00      0.00      0.00         0
          64       0.00      0.00      0.00         0
          65       0.00      0.00      0.00         2
          66       0.00      0.00      0.00         0
          67       0.75      0.19      0.30        16
          68       0.00      0.00      0.00         1
          69       0.00      0.00      0.00         2
          70       0.00      0.00      0.00         0
          71       0.00      0.00      0.00         0
          72       0.82      0.55      0.66       220
          73       0.71      0.07      0.13        72

   micro avg       0.70      0.46      0.56      2253
   macro avg       0.14      0.08      0.09      2253
weighted avg       0.60      0.46      0.51      2253
 samples avg       0.46      0.46      0.46      2253

In [0]:

ROC curve:
In [0]:
y_test_inv = np.argmax(y_test, axis=-1) # Inverse transform of y_test for use in ROC curve
In [0]:
y_pred64_inv = np.argmax(y_pred64_rd, axis = -1) # Inverse transform of y_pred of stacked lstm model with 64 units for use in ROC curve
In [0]:
y64_fpr, y64_tpr,_x = roc_curve(y_test_inv, y_pred64_inv, pos_label=1)
In [0]:
y_pred_inv = np.argmax(y_pred_rd, axis = -1) # Inverse transform of y_pred of stacked lstm model with 128 units for use in ROC curve
In [0]:
y_fpr, y_tpr, _ = roc_curve(y_test_inv, y_pred_inv, pos_label=1)
In [0]:
y_pred_l2_inv = np.argmax(y_pred_l2_rd, axis = -1) # Inverse transform of y_pred of stacked lstm model with 128 units and L2 regularization for use in ROC curve
In [0]:
y_l2_fpr, y_l2_tpr, _l2 = roc_curve(y_test_inv, y_pred_l2_inv, pos_label=1)
In [0]:
plt.plot(y_fpr, y_tpr, linestyle='--', label='LSTM_0.4_drpout and 128 units')
plt.plot(y_l2_fpr, y_l2_tpr, linestyle='solid', label='LSTM_0.4_dropout_L2')
plt.plot(y64_fpr, y64_tpr, linestyle='dashdot', label='LSTM with 64 units')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

Learnings from the approach
• We observed that by means of tweaking the various Hyperparameters in LSTM Model, maximum validation accuracy we reached is around 60 percent.

• Changing LSTM neurons from 64 to 128 has resulted in decrease of validation accuracy but has shown significant increase in train accuracy, which clearly the case of overfitting.

• We tried increasing the Dropout, which did not have impact on improving the validation accuracy.

• To reduce the overfitting, we tried to include L2 regularization parameter in the model, which as expected decreased the accuracy of the model as model is not generalizing well on unseen data.

• Though we were able to achieve an accuracy of 85 percent on train data, we are not getting above 60% on the validation set.
We can confidently say the main reasons could be the highly imbalanced data.

• There are around more than 35 groups which holds less than 2% (1 incident per group) of the total distribution where GRP_0 has hold nearly 47% of the distribution. We need to handle this distribution in order to get better results. We have seen how our RNN models are classifying the data. Now we are curious to see how classical machine learning models treat this kind of data. Models like SVM and Naïve Bayes prove to be quite robust in classification problems. As our next approach we will build these models on our preprocessed data.

Approach 3-Clustering method
In 3rd Approach we will merge the groups using Clustering method and explore some more insights on our dataset.
Please Note: This is not Modelling Approach rather it is an extension to our EDA.

We have understood that our data is highly imbalanced where one groups is holding mode than 46% of the distribution and on the other hand many groups only have single ticket entry. To understand our dataset better we are building clusters to identify groups having similar patterns and if need be can be grouped together to make data less sparse.

In [0]:
def prep_data4Modelling(data):
  
  print("Converting all text columns datatype to String type:")
  data['Short description']=data['Short description'].apply(str)
  data['Description'] = data['Description'].apply(str)
  data['Caller'] = data['Caller'].apply(str)
  
  print("Removing Caller Names from the text:")
  caller_list = data["Caller"].str.split(" ", n = 1, expand = True)  
  caller_fname = caller_list[0]
  caller_lname = caller_list[1]  
  caller_fname_list = caller_fname.to_list()
  caller_lname_list = caller_lname.to_list()
  
  data.Description = data.Description.apply(lambda x: ' '.join([word for word in x.split() if word not in caller_fname_list]))
  data.Description = data.Description.apply(lambda x: ' '.join([word for word in x.split() if word not in caller_lname_list]))
  
  print("Merging Description and short Description column:")
  #Let's combine all 3 independent attribute to 1
  data['Complete_Description'] = data['Short description'].str.cat(data['Description'],sep=" ")
  print("Removing unnecessary spaces:")
  data.Complete_Description = data.Complete_Description.apply(lambda x: x.strip())
  
  data['Complete_Description'] = (data['Complete_Description'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))
  
  print("Removing the rest of the columns from dataframe:")
  data.drop(['Short description','Description','Caller'], axis=1, inplace=True)

  print("Performing Language Detection:")
  translator = Translator()
  languages = googletrans.LANGUAGES

  DetectorFactory.seed = 0

  data['Language'] = data['Complete_Description'].apply(lambda x: detect(x))
  data['Language'] = data['Language'].apply(lambda x: languages[x].upper())
  
  print("Data Preprocessing Starts Here:")
  data = textPreprocessing(data,'Complete_Description')
  print("Data Preprocessing Ends Now:")

  #print(data.head())

  #print("Validation if all texts converted are english or not:")
  #data['NewLanguage'] = data['EnglishDescription'].apply(lambda x: detect(x))
  #print("Keeping only those texts which are translated to English by the Googletrans:")
  #data = data[data['NewLanguage']=='en']

  print("Tokenizing the data:")
  data['Tokens'] = data['EnglishDescription'].apply(lambda x: tokenizeText(x))
  
  print("Cleaning the tokens by removing stop words:")
  cleanTokens = []
  for col_desc in range(len(data.Tokens)):
    str_token = data.Tokens[col_desc]
    cleanTokens.append([w for w in str_token if w not in stop_words] )
  data['Tokens'] = cleanTokens
  
  print("Lemmatizing the tokenized words:")
  data['LemmaWords'] = data['Tokens'].apply(lambda x: lemmatizeText(x))
  print("Removing Duplicate words from the lemmatized words:")
  data['LemmaWords'] = data['LemmaWords'].apply(lambda x: RemovDupWordTokens(x))
  print("Converting the lemmatized tokens to String:")
  data['LemmaString'] = data['LemmaWords'].apply(lambda x: listToString(x))

  print("Now our all Pre-Processing steps are completed and final dataframe is created.")
  return data.head(5)
In [0]:
df = pd.read_excel('/content/drive/My Drive/GL AIML/Capstone/Input Data Synthetic (created but not used in our project).xlsx')
In [0]:
prep_data4Modelling(df)
Converting all text columns datatype to String type:
Removing Caller Names from the text:
Merging Description and short Description column:
Removing unnecessary spaces:
Removing the rest of the columns from dataframe:
Performing Language Detection:
Data Preprocessing Starts Here:
updating all cases to lower cases:
Translating Non English to English:
removing data using regular expression List:
removing stopwords:
removing top 20 Most common words:
removing punctuations:
Data Preprocessing Ends Now:
Tokenizing the data:
Cleaning the tokens by removing stop words:
Lemmatizing the tokenized words:
Removing Duplicate words from the lemmatized words:
Converting the lemmatized tokens to String:
Now our all Pre-Processing steps are completed and final dataframe is created.
Out[0]:
Assignment group	Complete_Description	Language	EnglishDescription	Tokens	LemmaWords	LemmaString
0	GRP_0	login issue -verified user details.(employee# ...	ENGLISH	login verified details employee manager checke...	[login, verified, details, employee, manager, ...	[login, verified, details, employee, manager, ...	login verified details employee manager chec...
1	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail...	ENGLISH	outlook hmjdrvpb komuaywn team meetings skype ...	[outlook, hmjdrvpb, komuaywn, team, meetings, ...	[outlook, hmjdrvpb, komuaywn, team, meetings, ...	outlook hmjdrvpb komuaywn team meetings skyp...
2	GRP_0	cant log in to vpn received from: eylqgodm.ybq...	ENGLISH	cant log vpn eylqgodm ybqkwiam cannot	[log, vpn, eylqgodm, ybqkwiam, not]	[log, vpn, eylqgodm, ybqkwiam, not]	log vpn eylqgodm ybqkwiam not
3	GRP_0	unable to access hr_tool page	ENGLISH	hr	[hr]	[hr]	hr
4	GRP_0	skype error	NORWEGIAN	skype	[skype]	[skype]	skype
In [0]:
final_df = df[['LemmaString','Assignment group']].copy()
In [0]:
final_df.head()
Out[0]:
LemmaString	Assignment group
0	login verified details employee manager chec...	GRP_0
1	outlook hmjdrvpb komuaywn team meetings skyp...	GRP_0
2	log vpn eylqgodm ybqkwiam not	GRP_0
3	hr	GRP_0
4	skype	GRP_0
In [0]:
df1 = final_df.copy()

df1['New group'] = LabelEncoder().fit_transform(df1['Assignment group'])
df1.head()
Out[0]:
LemmaString	Assignment group	New group
0	login verified details employee manager chec...	GRP_0	0
1	outlook hmjdrvpb komuaywn team meetings skyp...	GRP_0	0
2	log vpn eylqgodm ybqkwiam not	GRP_0	0
3	hr	GRP_0	0
4	skype	GRP_0	0
In [0]:
df2 = df1[['New group','LemmaString']].copy()
In [0]:
df2.head()
Out[0]:
New group	LemmaString
0	0	login verified details employee manager chec...
1	0	outlook hmjdrvpb komuaywn team meetings skyp...
2	0	log vpn eylqgodm ybqkwiam not
3	0	hr
4	0	skype
In [0]:
X = df2['LemmaString'].values

vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)
print(X_vectors.shape)
(8500, 14279)
In [0]:
from sklearn.metrics import silhouette_score, davies_bouldin_score

km_scores= []
km_silhouette = []
vmeasure_score =[]
db_score = []
for i in range(2,15):
    km = KMeans(n_clusters=i, random_state=0).fit(X_vectors)
    preds = km.predict(X_vectors)
    
    print("Score for number of cluster(s) {}: {}".format(i,km.score(X_vectors)))
    km_scores.append(-km.score(X_vectors))
    
    silhouette = silhouette_score(X_vectors,preds)
    km_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(X_vectors.toarray(),preds)
    db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    
    print("-"*100)
Score for number of cluster(s) 2: -7699.029694459533
Silhouette score for number of cluster(s) 2: 0.03824453198554169
Davies Bouldin score for number of cluster(s) 2: 3.2041028035086048
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 3: -7594.33101813118
Silhouette score for number of cluster(s) 3: 0.043912385955778656
Davies Bouldin score for number of cluster(s) 3: 3.0345585502552552
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 4: -7492.214319989217
Silhouette score for number of cluster(s) 4: 0.04979278227528991
Davies Bouldin score for number of cluster(s) 4: 2.5510831688599294
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 5: -7394.319103964547
Silhouette score for number of cluster(s) 5: 0.05747786214787084
Davies Bouldin score for number of cluster(s) 5: 2.6996758526846016
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 6: -7320.053072227638
Silhouette score for number of cluster(s) 6: 0.06123565669584459
Davies Bouldin score for number of cluster(s) 6: 3.1149470901328864
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 7: -7299.695389090368
Silhouette score for number of cluster(s) 7: 0.06373975833938139
Davies Bouldin score for number of cluster(s) 7: 3.3771656392302245
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 8: -7156.950785682733
Silhouette score for number of cluster(s) 8: 0.075695574166678
Davies Bouldin score for number of cluster(s) 8: 2.270863604278846
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 9: -7181.085424920812
Silhouette score for number of cluster(s) 9: 0.07020072083801077
Davies Bouldin score for number of cluster(s) 9: 2.8020389420503076
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 10: -7146.606652404974
Silhouette score for number of cluster(s) 10: 0.06670049090724077
Davies Bouldin score for number of cluster(s) 10: 5.366361259615799
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 11: -7040.426992892624
Silhouette score for number of cluster(s) 11: 0.08072405338196556
Davies Bouldin score for number of cluster(s) 11: 2.590282067045791
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 12: -7015.544525553675
Silhouette score for number of cluster(s) 12: 0.0814118274619837
Davies Bouldin score for number of cluster(s) 12: 3.6976390847983214
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 13: -6999.713018607186
Silhouette score for number of cluster(s) 13: 0.08189916914181834
Davies Bouldin score for number of cluster(s) 13: 3.801489036166272
----------------------------------------------------------------------------------------------------
Score for number of cluster(s) 14: -6979.210291979621
Silhouette score for number of cluster(s) 14: 0.08244411656400594
Davies Bouldin score for number of cluster(s) 14: 3.9278470077274434
----------------------------------------------------------------------------------------------------
In [0]:
plt.figure(figsize=(7,4))
plt.title("The elbow method for determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,15)],y=km_scores,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("K-means score",fontsize=15)
plt.xticks([i for i in range(2,15)],fontsize=14)
plt.yticks(fontsize=15)
plt.show()

In [0]:
plt.figure(figsize=(7,4))
plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,15)],y=km_silhouette,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Silhouette score",fontsize=15)
plt.xticks([i for i in range(2,15)],fontsize=14)
plt.yticks(fontsize=15)
plt.show()

In [0]:
plt.figure(figsize=(7,4))
plt.scatter(x=[i for i in range(2,15)],y=db_score,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Davies-Bouldin score")
plt.show()

In [0]:
km_8 = KMeans(n_clusters=8, random_state=0).fit(X_vectors)
preds_8 = km_8.predict(X_vectors)
In [0]:
labels = km_8.labels_
In [0]:
df2['clusters'] = labels
In [0]:
df2.head()
Out[0]:
New group	LemmaString	clusters
0	0	login verified details employee manager chec...	4
1	0	outlook hmjdrvpb komuaywn team meetings skyp...	1
2	0	log vpn eylqgodm ybqkwiam not	1
3	0	hr	1
4	0	skype	1
In [0]:
df.head()
Out[0]:
Assignment group	Complete_Description	Language	EnglishDescription	Tokens	LemmaWords	LemmaString
0	GRP_0	login issue -verified user details.(employee# ...	ENGLISH	login verified details employee manager checke...	[login, verified, details, employee, manager, ...	[login, verified, details, employee, manager, ...	login verified details employee manager chec...
1	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail...	ENGLISH	outlook hmjdrvpb komuaywn team meetings skype ...	[outlook, hmjdrvpb, komuaywn, team, meetings, ...	[outlook, hmjdrvpb, komuaywn, team, meetings, ...	outlook hmjdrvpb komuaywn team meetings skyp...
2	GRP_0	cant log in to vpn received from: eylqgodm.ybq...	ENGLISH	cant log vpn eylqgodm ybqkwiam cannot	[log, vpn, eylqgodm, ybqkwiam, not]	[log, vpn, eylqgodm, ybqkwiam, not]	log vpn eylqgodm ybqkwiam not
3	GRP_0	unable to access hr_tool page	ENGLISH	hr	[hr]	[hr]	hr
4	GRP_0	skype error	NORWEGIAN	skype	[skype]	[skype]	skype
In [0]:
df2['Assignment_group'] = df['Assignment group']
In [0]:
df2.head()
Out[0]:
New group	LemmaString	clusters	Assignment_group
0	0	login verified details employee manager chec...	4	GRP_0
1	0	outlook hmjdrvpb komuaywn team meetings skyp...	1	GRP_0
2	0	log vpn eylqgodm ybqkwiam not	1	GRP_0
3	0	hr	1	GRP_0
4	0	skype	1	GRP_0
In [0]:
df2[df2['Assignment_group']=='GRP_0']['clusters'].unique()
Out[0]:
array([4, 1, 5, 3, 6, 2])
In [0]:
df2[df2['Assignment_group']=='GRP_1']['clusters'].unique()
Out[0]:
array([1, 0])
In [0]:
#Word Cloud
text = ' '.join(map(str, df2[df2['clusters']==0].LemmaString))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

In [0]:
#Word Cloud
text = ' '.join(map(str, df2[df2['clusters']==1].LemmaString))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

In [0]:
#Word Cloud
text = ' '.join(map(str, df2[df2['clusters']==2].LemmaString))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

In [0]:
#Word Cloud
text = ' '.join(map(str, df2[df2['clusters']==3].LemmaString))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

In [0]:
#Word Cloud
text = ' '.join(map(str, df2[df2['clusters']==4].LemmaString))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

In [0]:
#Word Cloud
text = ' '.join(map(str, df2[df2['clusters']==5].LemmaString))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

In [0]:
#Word Cloud
text = ' '.join(map(str, df2[df2['clusters']==6].LemmaString))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

In [0]:
#Word Cloud
text = ' '.join(map(str, df2[df2['clusters']==7].LemmaString))

wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(18,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

Observations:
We could see very interesting insights from these 8 clusters. The issues in these 8 clusters seems to be as below:
0: Job Abends, backup issue
1: Outlook, message, mail, vpn issue
2: Internet Explorer, Browser issues
3: Windows login, locking issues
4: Employee login, teamviewer, password related issues
5: inplant, printer, windows update related issues
6: windows software request, locking issues
7: Scheduled maintenance, equipment related issue

Exporting the Extracted Cluster Label Data and Complete Proprocessed data to Excel File for future use

In [0]:
from pandas import ExcelWriter

writer1 = ExcelWriter('Complete_data2.xlsx')
df.to_excel(writer1,'Sheet1', index=False, )
writer1.save()

writer2 = ExcelWriter('Cluster_labels2.xlsx')
df2.to_excel(writer2,'Sheet1', index=False, )
writer2.save()
In [0]:
df2.head()
Out[0]:
New group	LemmaString	clusters	Assignment_group
0	0	login verified details employee manager chec...	4	GRP_0
1	0	outlook hmjdrvpb komuaywn team meetings skyp...	1	GRP_0
2	0	log vpn eylqgodm ybqkwiam not	1	GRP_0
3	0	hr	1	GRP_0
4	0	skype	1	GRP_0
Model Building
In [0]:
max_features = 10000
embedding_size = 50
In [0]:
def create_corpus(df):
    corpus=[]
    for desc in tqdm(df['LemmaString'].astype(str)):
        words=[word.lower() for word in word_tokenize(desc) if((word.isalpha()==1))] # & (word not in stop_words))]
        corpus.append(words)
    return corpus
In [0]:
def create_inputvector(data,max_features,embedding_size,target):
  corpus = create_corpus(data)

  nb_words = []
  for i in range(len(data['LemmaString'])):
    nb_words.append(len(data['LemmaString'].iloc[i].split()))

  data['nb_words'] = nb_words
  print(data.columns)
  maxlen = max(data['nb_words'])

  tokenizer_obj=Tokenizer(num_words=max_features)
  tokenizer_obj.fit_on_texts(corpus)
  sequences=tokenizer_obj.texts_to_sequences(corpus)

  word_index = tokenizer_obj.word_index
  num_words = len(word_index)+1

  X = tokenizer_obj.texts_to_sequences(corpus)
  X = pad_sequences(X, maxlen = maxlen)
  y1 = data[target]
  y = to_categorical(y1, num_classes=8)
  return X,y,y1,maxlen,num_words
In [0]:
X,y,y1,maxlen,num_words=create_inputvector(data=df2,max_features=max_features,embedding_size=embedding_size,target='clusters')
100%|██████████| 8500/8500 [00:01<00:00, 8394.22it/s]
Index(['New group', 'LemmaString', 'clusters', 'Assignment_group', 'nb_words'], dtype='object')
In [0]:
batch_size = 100
epochs = 50
In [0]:
def LSTM_model(maxlen,batch_size,epochs):
  main_input = Input(shape=(maxlen,))
  em = Embedding(max_features, 100, input_length=maxlen) (main_input)

  lstm_out1 = LSTM(128, return_sequences = True)(em)
  lstm_out2 = LSTM(128)(lstm_out1)

  x = Dropout(0.2)(lstm_out2)

  main_output = Dense(8, activation = 'softmax')(x)

  model = Model(inputs = main_input, outputs = main_output)
  # compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  checkpoint = ModelCheckpoint("model-{loss:.2f}.h5", monitor="loss", verbose=1, save_best_only=True,
                               save_weights_only=True, mode="min", period=1)
  stop = EarlyStopping(monitor="loss", patience=5, mode="min")
  reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="min")
  print(model.summary())
  return model,checkpoint,stop,reduce_lr
In [0]:
def plot_graph(history):
  fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
  fig.suptitle("Performance: GRU Model")
  ax1.plot(history.history['acc'])
  ax1.plot(history.history['val_acc'])
  cut = np.where(history.history['val_acc'] == np.max(history.history['val_acc']))[0][0]
  ax1.axvline(x=cut, color='k', linestyle='--')
  ax1.set_title("Model Accuracy")
  ax1.legend(['train', 'test'])

  ax2.plot(history.history['loss'])
  ax2.plot(history.history['val_loss'])
  cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]
  ax2.axvline(x=cut, color='k', linestyle='--')
  ax2.set_title("Model Loss")
  ax2.legend(['train', 'test'])
Model 1: Vanilla LSTM
In [0]:
model1,checkpoint1,stop1,reduce_lr1 = LSTM_model(maxlen=maxlen,batch_size=batch_size,epochs=epochs)
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:66: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:541: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4432: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:148: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3733: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/optimizers.py:793: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3576: The name tf.log is deprecated. Please use tf.math.log instead.

Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 415)               0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 415, 100)          1000000   
_________________________________________________________________
lstm_1 (LSTM)                (None, 415, 128)          117248    
_________________________________________________________________
lstm_2 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 1032      
=================================================================
Total params: 1,249,864
Trainable params: 1,249,864
Non-trainable params: 0
_________________________________________________________________
None
In [0]:
history1 = model1.fit(X, y, validation_split= 0.2, batch_size=batch_size, epochs=50, callbacks=[stop1], class_weight = 'auto')
WARNING:tensorflow:From /tensorflow-1.15.0/python3.6/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3005: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

Train on 6800 samples, validate on 1700 samples
Epoch 1/50
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:197: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

6800/6800 [==============================] - 167s 25ms/step - loss: 0.6808 - acc: 0.8684 - val_loss: 0.9454 - val_acc: 0.8388
Epoch 2/50
6800/6800 [==============================] - 164s 24ms/step - loss: 0.3590 - acc: 0.9143 - val_loss: 0.7745 - val_acc: 0.8447
Epoch 3/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.1682 - acc: 0.9418 - val_loss: 0.7686 - val_acc: 0.8941
Epoch 4/50
6800/6800 [==============================] - 162s 24ms/step - loss: 0.0783 - acc: 0.9765 - val_loss: 0.7969 - val_acc: 0.8988
Epoch 5/50
6800/6800 [==============================] - 164s 24ms/step - loss: 0.0425 - acc: 0.9907 - val_loss: 0.6751 - val_acc: 0.9047
Epoch 6/50
6800/6800 [==============================] - 166s 24ms/step - loss: 0.0252 - acc: 0.9951 - val_loss: 0.7842 - val_acc: 0.9059
Epoch 7/50
6800/6800 [==============================] - 166s 24ms/step - loss: 0.0165 - acc: 0.9966 - val_loss: 0.6002 - val_acc: 0.9035
Epoch 8/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0112 - acc: 0.9979 - val_loss: 0.7224 - val_acc: 0.9071
Epoch 9/50
6800/6800 [==============================] - 167s 25ms/step - loss: 0.0101 - acc: 0.9981 - val_loss: 0.5930 - val_acc: 0.9035
Epoch 10/50
6800/6800 [==============================] - 168s 25ms/step - loss: 0.0089 - acc: 0.9985 - val_loss: 0.6263 - val_acc: 0.9041
Epoch 11/50
6800/6800 [==============================] - 166s 24ms/step - loss: 0.0068 - acc: 0.9981 - val_loss: 0.5756 - val_acc: 0.9047
Epoch 12/50
6800/6800 [==============================] - 166s 24ms/step - loss: 0.0064 - acc: 0.9987 - val_loss: 0.5075 - val_acc: 0.9035
Epoch 13/50
6800/6800 [==============================] - 167s 25ms/step - loss: 0.0053 - acc: 0.9988 - val_loss: 0.4897 - val_acc: 0.9035
Epoch 14/50
6800/6800 [==============================] - 167s 25ms/step - loss: 0.0044 - acc: 0.9993 - val_loss: 0.5763 - val_acc: 0.9035
Epoch 15/50
6800/6800 [==============================] - 167s 25ms/step - loss: 0.0044 - acc: 0.9991 - val_loss: 0.4775 - val_acc: 0.8982
Epoch 16/50
6800/6800 [==============================] - 168s 25ms/step - loss: 0.0040 - acc: 0.9991 - val_loss: 0.3787 - val_acc: 0.9006
Epoch 17/50
6800/6800 [==============================] - 169s 25ms/step - loss: 0.0033 - acc: 0.9994 - val_loss: 0.4721 - val_acc: 0.9035
Epoch 18/50
6800/6800 [==============================] - 171s 25ms/step - loss: 0.0031 - acc: 0.9993 - val_loss: 0.4022 - val_acc: 0.9047
Epoch 19/50
6800/6800 [==============================] - 168s 25ms/step - loss: 0.0028 - acc: 0.9993 - val_loss: 0.4284 - val_acc: 0.9029
Epoch 20/50
6800/6800 [==============================] - 168s 25ms/step - loss: 0.0025 - acc: 0.9996 - val_loss: 0.4706 - val_acc: 0.9024
Epoch 21/50
6800/6800 [==============================] - 168s 25ms/step - loss: 0.0023 - acc: 0.9997 - val_loss: 0.4300 - val_acc: 0.9018
Epoch 22/50
6800/6800 [==============================] - 167s 24ms/step - loss: 0.0021 - acc: 0.9997 - val_loss: 0.4234 - val_acc: 0.9029
Epoch 23/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0018 - acc: 0.9997 - val_loss: 0.4321 - val_acc: 0.9035
Epoch 24/50
6800/6800 [==============================] - 163s 24ms/step - loss: 0.0024 - acc: 0.9996 - val_loss: 0.3407 - val_acc: 0.9024
Epoch 25/50
6800/6800 [==============================] - 160s 24ms/step - loss: 0.0058 - acc: 0.9982 - val_loss: 0.3787 - val_acc: 0.8888
Epoch 26/50
6800/6800 [==============================] - 159s 23ms/step - loss: 0.0125 - acc: 0.9969 - val_loss: 0.4399 - val_acc: 0.9059
Epoch 27/50
6800/6800 [==============================] - 161s 24ms/step - loss: 0.0030 - acc: 0.9996 - val_loss: 0.3610 - val_acc: 0.9071
Epoch 28/50
6800/6800 [==============================] - 162s 24ms/step - loss: 0.0025 - acc: 0.9996 - val_loss: 0.4361 - val_acc: 0.9059
In [0]:
plot_graph(history1)

Model 2: LSTM with Stratified K Fold
In [0]:
skf = StratifiedKFold(n_splits=5,random_state=24, shuffle=True)
models, measures = [], []
for index, (train_indices, val_indices) in enumerate(skf.split(X, y1)):
   ## Extract Kfold
  X_train_sfk, X_test_sfk = X[train_indices], X[val_indices]
  y_train_sfk, y_test_sfk = y1[train_indices], y1[val_indices]
In [0]:
y_train_sfk = to_categorical(y_train_sfk, num_classes= 8)
In [0]:
y_test_sfk = to_categorical(y_test_sfk,num_classes=8)
In [0]:
model2,checkpoint2,stop2,reduce_lr2 = LSTM_model(maxlen=maxlen,batch_size=batch_size,epochs=epochs)
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 415)               0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 415, 100)          1000000   
_________________________________________________________________
lstm_3 (LSTM)                (None, 415, 128)          117248    
_________________________________________________________________
lstm_4 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 1032      
=================================================================
Total params: 1,249,864
Trainable params: 1,249,864
Non-trainable params: 0
_________________________________________________________________
None
In [0]:
history2 = model2.fit(X_train_sfk, y_train_sfk, validation_data=(X_test_sfk,y_test_sfk), batch_size=batch_size, epochs=50, callbacks=[stop2,reduce_lr2]) #, class_weight = 'auto')
Train on 6800 samples, validate on 1700 samples
Epoch 1/50
6800/6800 [==============================] - 162s 24ms/step - loss: 0.7580 - acc: 0.8581 - val_loss: 0.4856 - val_acc: 0.8971
Epoch 2/50
6800/6800 [==============================] - 161s 24ms/step - loss: 0.3923 - acc: 0.8997 - val_loss: 0.2640 - val_acc: 0.9194
Epoch 3/50
6800/6800 [==============================] - 160s 23ms/step - loss: 0.1987 - acc: 0.9321 - val_loss: 0.2012 - val_acc: 0.9341
Epoch 4/50
6800/6800 [==============================] - 157s 23ms/step - loss: 0.2779 - acc: 0.9101 - val_loss: 0.2108 - val_acc: 0.9194
Epoch 5/50
6800/6800 [==============================] - 159s 23ms/step - loss: 0.1526 - acc: 0.9500 - val_loss: 0.1513 - val_acc: 0.9465
Epoch 6/50
6800/6800 [==============================] - 157s 23ms/step - loss: 0.1027 - acc: 0.9697 - val_loss: 0.1192 - val_acc: 0.9671
Epoch 7/50
6800/6800 [==============================] - 159s 23ms/step - loss: 0.0767 - acc: 0.9826 - val_loss: 0.1056 - val_acc: 0.9706
Epoch 8/50
6800/6800 [==============================] - 158s 23ms/step - loss: 0.0568 - acc: 0.9869 - val_loss: 0.0883 - val_acc: 0.9729
Epoch 9/50
6800/6800 [==============================] - 160s 24ms/step - loss: 0.0386 - acc: 0.9928 - val_loss: 0.0815 - val_acc: 0.9794
Epoch 10/50
6800/6800 [==============================] - 160s 24ms/step - loss: 0.0280 - acc: 0.9959 - val_loss: 0.0842 - val_acc: 0.9788
Epoch 11/50
6800/6800 [==============================] - 163s 24ms/step - loss: 0.0212 - acc: 0.9971 - val_loss: 0.0790 - val_acc: 0.9824
Epoch 12/50
6800/6800 [==============================] - 164s 24ms/step - loss: 0.0164 - acc: 0.9979 - val_loss: 0.0793 - val_acc: 0.9824
Epoch 13/50
6800/6800 [==============================] - 166s 24ms/step - loss: 0.0133 - acc: 0.9981 - val_loss: 0.0743 - val_acc: 0.9841
Epoch 14/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0115 - acc: 0.9982 - val_loss: 0.0742 - val_acc: 0.9847
Epoch 15/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0106 - acc: 0.9987 - val_loss: 0.0745 - val_acc: 0.9835
Epoch 16/50
6800/6800 [==============================] - 167s 25ms/step - loss: 0.0087 - acc: 0.9987 - val_loss: 0.0921 - val_acc: 0.9806
Epoch 17/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0079 - acc: 0.9990 - val_loss: 0.0840 - val_acc: 0.9818
Epoch 18/50
6800/6800 [==============================] - 166s 24ms/step - loss: 0.0072 - acc: 0.9988 - val_loss: 0.0768 - val_acc: 0.9835
Epoch 19/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0064 - acc: 0.9990 - val_loss: 0.0744 - val_acc: 0.9841
Epoch 20/50
6800/6800 [==============================] - 164s 24ms/step - loss: 0.0054 - acc: 0.9991 - val_loss: 0.0768 - val_acc: 0.9841
Epoch 21/50
6800/6800 [==============================] - 167s 25ms/step - loss: 0.0051 - acc: 0.9991 - val_loss: 0.0822 - val_acc: 0.9835
Epoch 22/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0049 - acc: 0.9993 - val_loss: 0.0815 - val_acc: 0.9835
Epoch 23/50
6800/6800 [==============================] - 168s 25ms/step - loss: 0.0047 - acc: 0.9994 - val_loss: 0.0830 - val_acc: 0.9841
Epoch 24/50
6800/6800 [==============================] - 170s 25ms/step - loss: 0.0050 - acc: 0.9993 - val_loss: 0.0955 - val_acc: 0.9824
Epoch 25/50
6800/6800 [==============================] - 171s 25ms/step - loss: 0.0045 - acc: 0.9994 - val_loss: 0.0837 - val_acc: 0.9829
Epoch 26/50
6800/6800 [==============================] - 169s 25ms/step - loss: 0.0039 - acc: 0.9993 - val_loss: 0.0865 - val_acc: 0.9847
Epoch 27/50
6800/6800 [==============================] - 168s 25ms/step - loss: 0.0037 - acc: 0.9994 - val_loss: 0.0832 - val_acc: 0.9835
Epoch 28/50
6800/6800 [==============================] - 169s 25ms/step - loss: 0.0035 - acc: 0.9994 - val_loss: 0.0878 - val_acc: 0.9841
Epoch 29/50
6800/6800 [==============================] - 169s 25ms/step - loss: 0.0035 - acc: 0.9994 - val_loss: 0.0839 - val_acc: 0.9841
Epoch 30/50
6800/6800 [==============================] - 168s 25ms/step - loss: 0.0035 - acc: 0.9994 - val_loss: 0.0909 - val_acc: 0.9841
Epoch 31/50
6800/6800 [==============================] - 170s 25ms/step - loss: 0.0036 - acc: 0.9996 - val_loss: 0.0803 - val_acc: 0.9841
Epoch 32/50
6800/6800 [==============================] - 171s 25ms/step - loss: 0.0034 - acc: 0.9994 - val_loss: 0.0864 - val_acc: 0.9835
Epoch 33/50
6800/6800 [==============================] - 171s 25ms/step - loss: 0.0032 - acc: 0.9994 - val_loss: 0.0906 - val_acc: 0.9841
Epoch 34/50
6800/6800 [==============================] - 172s 25ms/step - loss: 0.0032 - acc: 0.9996 - val_loss: 0.0866 - val_acc: 0.9847
Epoch 35/50
6800/6800 [==============================] - 170s 25ms/step - loss: 0.0029 - acc: 0.9996 - val_loss: 0.0826 - val_acc: 0.9841
Epoch 36/50
6800/6800 [==============================] - 170s 25ms/step - loss: 0.0034 - acc: 0.9996 - val_loss: 0.0899 - val_acc: 0.9835
Epoch 37/50
6800/6800 [==============================] - 169s 25ms/step - loss: 0.0030 - acc: 0.9996 - val_loss: 0.0881 - val_acc: 0.9835
Epoch 38/50
6800/6800 [==============================] - 166s 24ms/step - loss: 0.0022 - acc: 0.9996 - val_loss: 0.0887 - val_acc: 0.9829
Epoch 39/50
6800/6800 [==============================] - 166s 24ms/step - loss: 0.0025 - acc: 0.9996 - val_loss: 0.0959 - val_acc: 0.9841
Epoch 40/50
6800/6800 [==============================] - 166s 24ms/step - loss: 0.0028 - acc: 0.9996 - val_loss: 0.0884 - val_acc: 0.9829
Epoch 41/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0028 - acc: 0.9996 - val_loss: 0.0896 - val_acc: 0.9829
Epoch 42/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0026 - acc: 0.9996 - val_loss: 0.0856 - val_acc: 0.9824
Epoch 43/50
6800/6800 [==============================] - 165s 24ms/step - loss: 0.0031 - acc: 0.9996 - val_loss: 0.0943 - val_acc: 0.9841

Epoch 00043: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
In [0]:
plot_graph(history2)

Summary:
Even though the Accuracy came good for all 8 clusters. When we tried to train the model on individual clusters to predict ground truth labels(Grp_0, Grp_1.. Grp_73 etc), accuracy came very low.
As we had similar class imbalance issue in few clusters.
So, we plan to oversample our minority classes so as to correct the imbalance problem.

Approach 4 - L12 and L3 Approach
L12 and L3 Approach

Revisiting our problem description, it describes the current situation as:
“Currently the incidents are created by various stakeholders (Business Users, IT Users and Monitoring Tools) within IT Service Management Tool and are assigned to Service Desk teams (L1 / L2 teams). This team will review the incidents for right ticket categorization, priorities and then carry out initial diagnosis to see if they can resolve. Around 54% of the incidents are resolved by L1 / L2 teams. Incase L1 / L2 is unable to resolve, they will then escalate / assign the tickets to Functional teams from Applications and Infrastructure (L3 teams)”

One possible inference we could draw by relating the statement to the dataset is that few of labeled groups which holds the maximum of the data distribution (54%) belong to L1/L2 teams and rest belong to the functional teams i.e. L3 teams.
This also justifies the data distribution we have as GRP_0 alone holds 46% of the distribution and there are only 8 groups which holds count of 150 or more tickets so they might represent L1/L2 teams and many groups have only 1 or 2 entries belong to L3 teams.
Approach:

As our next approach we will be building a two-stage classification model. At stage 1, ticket will be classified as one to the two groups i.e. L12 and L3. At the later stage, depending on the classified group, its actual groups will be predicted from one of 74 given group label.

image.png

Description of the model

Design of the solution is such that the model first learns how to classify the data into two sub groups and further in the next step it classifies it into final respective target subset groups. The steps as follows:
1) Assigning L12 and L3 groups: First we are adding new column named “L12Grp” to our preprocessed data which will hold binary value of whether the data belongs to L12 group (value 0) or L3(value 1) according to the defined threshold.
Threshold: As a threshold, we have kept 150 as the count. Groups having entries more than 150 belong to L12 dataset (L1/L2 teams) and rest belong to the L3 dataset (L3 Teams).

image.png

2) MODEL STAGE 1: For our stage 1 model we will be training the data keeping “L12Grp” as the target variable and description as input variable. We will be trying two models at this stage one using classical machine learning model Random Forest and one Deep learning RNN model of LSTM.

This model will do binary classification and label the data into 0 (L12) or 1(L3).

3) MODEL STAGE 2: For our stage 2 model, first we will be filtering our dataset into two subsets based on the “L12Grp”. This will result in two separate datasets for the L12 and L3 sub groups and we will build two separate models for each.
This model will do multiclass classification from the subset of groups they have.

In [0]:
L12 = ['GRP_0', 'GRP_8', 'GRP_9', 'GRP_2', 'GRP_12', 'GRP_19', 'GRP_3', 'GRP_6']

L3 = [ 'GRP_1', 'GRP_4', 'GRP_5', 'GRP_7', 'GRP_10', 'GRP_11',  'GRP_13', 'GRP_14', 'GRP_15', 'GRP_16', 'GRP_17', 'GRP_18', 
       'GRP_20', 'GRP_21', 'GRP_22',  'GRP_23', 'GRP_24',  'GRP_25', 'GRP_26', 'GRP_27', 'GRP_28', 'GRP_29', 'GRP_30', 'GRP_31',
       'GRP_33', 'GRP_34', 'GRP_35', 'GRP_36', 'GRP_37', 'GRP_38', 'GRP_39', 'GRP_40', 'GRP_41', 'GRP_42', 'GRP_43', 'GRP_44',
       'GRP_45', 'GRP_46', 'GRP_47', 'GRP_48', 'GRP_49', 'GRP_50',  'GRP_51', 'GRP_52', 'GRP_53', 'GRP_54', 'GRP_55', 'GRP_56',
       'GRP_57', 'GRP_58', 'GRP_59', 'GRP_60', 'GRP_61', 'GRP_32',  'GRP_62', 'GRP_63', 'GRP_64', 'GRP_65',
       'GRP_66', 'GRP_67', 'GRP_68', 'GRP_69', 'GRP_70', 'GRP_71', 'GRP_72', 'GRP_73']
In [0]:
def print_scores(y_test,y_pred):
    print("Accuracy score: \n", accuracy_score(y_test,y_pred))
    print('Test-set confusion matrix:\n', confusion_matrix(y_test,y_pred))
    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm, annot=True, fmt='g')
    print("Classification report:" "\n", classification_report(y_test,y_pred))
Defining the variables

In [0]:
vocabulary = dict()
inverse_vocabulary = ['<unk>']
embedding_dim = 300
embeddings = 1 * np.random.randn(30000, embedding_dim)
   
colsToTrainOn =['LemmaString']
Functions for various purposes
Encode the two groups L12/L3 in 0 and 1

In [0]:
def SetGrp(text):
  ''' this will set the L12 to 0
  and L3 to 1'''
  
  if text in L12:
      return 0
  if text in L3:
      return 1
In [0]:
def text_to_word_list(text):
  ''' Pre process and convert texts to a list of words '''
  text = str(text)
  text = text.lower().replace('.',' ').replace(',',' ').replace('-',' ').replace('  ', ' ')
  text = text.split()

  return text
In [0]:
def readFile():    
  ''' This function will read the file that is saved after the EDA completion
  Set Level Groups to 0 or 1 '''

  df = pd.read_excel('EDA_Cleaned_PreProcessed.xlsx')
  
  df = df[['LemmaString','AssignmentGroup']]
  
  df['LemmaString'] = df['LemmaString'].apply(lambda x : text_to_word_list(x))
  
  df['LemmaStringRetained'] = df['LemmaString']
  df['L12Grp'] = 0
  df['L12Grp'] = df['AssignmentGroup'].apply(lambda x : SetGrp(x))
  
  return df
In [0]:
def PrepDataForPrediction(df,vocabulary,inverse_vocabulary):
  ''' Will create the vocabulary and inverse_vocabulary list to be used in embedding 
  and later to be referred when we will call the same in our predict section '''
  
  for dataset in [df]:
    for index, row in dataset.iterrows():

      # Iterate through the text of description column of the row
      for question in colsToTrainOn:
        q2n = []  
        for word in row[question]:
          if word not in vocabulary:
              vocabulary[word] = len(inverse_vocabulary)
              q2n.append(len(inverse_vocabulary))
              inverse_vocabulary.append(word)
          else:
              q2n.append(vocabulary[word])
            # Replace description as word to description as number representation
          dataset.at[index, question]= q2n
  return df, vocabulary,inverse_vocabulary
In [0]:
def PrepDataForModel_FirstLevelGrouping(df,max_seq_length):
  """First prepare the X and Y by getting the combined description column in the X and Level 1 grouping in Y
  Once this is done then split the data using the train test split and test size randomly given
  After that pad the X_train and X_validation sequences
  Return X_train, Y_train, X_validation,Y_validation"""
  
  X = df[colsToTrainOn[0]]
  Y = np_utils.to_categorical(df['L12Grp'])
    
  X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=30)
  X_train = pad_sequences(X_train, maxlen=maxlen,truncating='post',padding='post',value=0)
  X_validation = pad_sequences(X_validation, maxlen=maxlen,truncating='post',padding='post',value=0)
  return X_train, Y_train, X_validation,Y_validation
In [0]:
def PrepDataForModel_SecondLevelGrouping(df,max_seq_length):
  """This function will first create the two data frames one with groups defined in L12 and other in L3. 
  The objective of this is to find the actual Assignment Group.
  Will create two train test split sets. One that will have for L12 groups and the other for the L3 groups
  """

  partDfZero = df.loc[df['L12Grp'] == 0]
  partDfOne = df.loc[df['L12Grp'] == 1]

  le1 = LabelEncoder()
  le2 = LabelEncoder()
  
  class_weightsZero = class_weight.compute_class_weight('balanced', np.unique(df['AssignmentGroup']), df['AssignmentGroup'])
  class_weightsOne = class_weight.compute_class_weight('balanced', np.unique(df['AssignmentGroup']), df['AssignmentGroup'])


  partDfZero['AssignmentGroup'] = le1.fit_transform(partDfZero['AssignmentGroup'])
  partDfOne['AssignmentGroup'] = le2.fit_transform(partDfOne['AssignmentGroup'])

  X_Zero = partDfZero[colsToTrainOn[0]]
  Y_Zero = np_utils.to_categorical(partDfZero['AssignmentGroup'])

  X_One = partDfOne[colsToTrainOn[0]]
  Y_One = np_utils.to_categorical(partDfOne['AssignmentGroup'])
      
  X_train_Zero, X_validation_Zero, Y_train_Zero, Y_validation_Zero = train_test_split(X_Zero, Y_Zero, test_size=0.3)
  
  X_train_One, X_validation_One, Y_train_One, Y_validation_One = train_test_split(X_One, Y_One, test_size=0.3)

  X_train_Zero = pad_sequences(X_train_Zero, maxlen=maxlen,truncating='post',padding='post',value=0)
  X_validation_Zero = pad_sequences(X_validation_Zero, maxlen=maxlen,truncating='post',padding='post',value=0)

  X_train_One = pad_sequences(X_train_One, maxlen=maxlen,truncating='post',padding='post',value=0)
  X_validation_One = pad_sequences(X_validation_One, maxlen=maxlen,truncating='post',padding='post',value=0)


  return X_train_Zero, Y_train_Zero, X_validation_Zero,Y_validation_Zero,X_train_One,X_validation_One,Y_train_One,Y_validation_One,partDfZero,partDfOne, class_weightsZero, class_weightsOne
Create the embedding matrix. Here Word2Vec is used because of its being trained over the Wikipedia and has better coverage of all words.

In [0]:
def BuildEmbeddingMatrix(word2vec):
  ''' prepare the ebedding vector matrix using the word2vec '''
  
  count = 0    
  for word, index in vocabulary.items():
    if word in word2vec.vocab:
        count+=1
        embeddings[index] = word2vec.word_vec(word)
    else:
        embeddings[index] = 0
This method will help to dump the emebddings, the vocabulary and inverse_vocabulary created for further use in predict model in different python file

In [0]:
def PickleCustomObjects(embeddings,vocabulary,inverse_vocabulary):
  ''' this function will help dump the embeddings, vocabulary and inverse_vocabulary
  which will be referred in predict model python file when we will predict the new incoming data '''

  with open('embeddings.pickle', 'wb') as f:
      pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
      
  with open('vocabulary.pickle', 'wb') as f:
      pickle.dump(vocabulary, f, pickle.HIGHEST_PROTOCOL)
      
  with open('inverse_vocabulary.pickle', 'wb') as f:
      pickle.dump(inverse_vocabulary, f, pickle.HIGHEST_PROTOCOL)
In [0]:
#read files with some processing
df = readFile()

print(df['L12Grp'].value_counts())
0    5273
1    2234
Name: L12Grp, dtype: int64
In [0]:
#display top n records 
df.head(10)
Out[0]:
LemmaString	AssignmentGroup	LemmaStringRetained	L12Grp
0	[login, issue, verified, user, details, employ...	GRP_0	[login, issue, verified, user, details, employ...	0
1	[outlook, hmjdrvpb, komuaywn, team, meetings, ...	GRP_0	[outlook, hmjdrvpb, komuaywn, team, meetings, ...	0
2	[log, vpn, eylqgodm, ybqkwiam, i, not]	GRP_0	[log, vpn, eylqgodm, ybqkwiam, i, not]	0
3	[unable, access, hr, tool]	GRP_0	[unable, access, hr, tool]	0
4	[skype, error]	GRP_0	[skype, error]	0
5	[unable, log, engineering, tool, skype]	GRP_0	[unable, log, engineering, tool, skype]	0
6	[event, critical, hostname, value, mountpoint,...	GRP_1	[event, critical, hostname, value, mountpoint,...	1
7	[no, employment, status, employee, enter, user]	GRP_0	[no, employment, status, employee, enter, user]	0
8	[unable, disable, add, ins, outlook]	GRP_0	[unable, disable, add, ins, outlook]	0
9	[update, inplant]	GRP_0	[update, inplant]	0
In [0]:

Model Building
In [0]:
maxlen = min(df['LemmaString'].map(lambda x:len(x)).max(), 150)
df, vocabulary,inverse_vocabulary = PrepDataForPrediction(df,vocabulary,inverse_vocabulary)
In [0]:
#Load word2vec
with open('/content/drive/My Drive/Greatlakes/Capstone_CB/Approach2/word2vec.pickle', 'rb') as f:
    word2vec = pickle.load(f)
In [0]:
BuildEmbeddingMatrix(word2vec)
First MODEL
In [0]:
X_train, Y_train, X_validation,Y_validation = PrepDataForModel_FirstLevelGrouping(df, maxlen)
In [0]:
model_FirstLevelGrouping=Sequential()

embedding=Embedding(len(embeddings),300,weights=[embeddings], input_length=maxlen,trainable=False)

model_FirstLevelGrouping.add(embedding)

model_FirstLevelGrouping.add(Bidirectional(LSTM(maxlen, return_sequences=True, recurrent_dropout=0.3)))
model_FirstLevelGrouping.add(Bidirectional(LSTM(maxlen, return_sequences=False)))

#model_FirstLevelGrouping.add(LSTM(maxlen, recurrent_dropout=0.2))
model_FirstLevelGrouping.add(Dense(50, activation='relu'))
model_FirstLevelGrouping.add(Dropout(0.4) )
model_FirstLevelGrouping.add(Dense(30, activation='relu'))
model_FirstLevelGrouping.add(Dropout(0.4) )

#we have two groups to target either L12(0) or L3(1)
model_FirstLevelGrouping.add(Dense(2, activation='softmax'))
model_FirstLevelGrouping.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model_FirstLevelGrouping.summary()
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 150, 300)          9000000   
_________________________________________________________________
bidirectional_3 (Bidirection (None, 150, 300)          541200    
_________________________________________________________________
bidirectional_4 (Bidirection (None, 300)               541200    
_________________________________________________________________
dense_4 (Dense)              (None, 50)                15050     
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 30)                1530      
_________________________________________________________________
dropout_4 (Dropout)          (None, 30)                0         
_________________________________________________________________
dense_6 (Dense)              (None, 2)                 62        
=================================================================
Total params: 10,099,042
Trainable params: 1,099,042
Non-trainable params: 9,000,000
_________________________________________________________________
In [0]:
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

history=model_FirstLevelGrouping.fit(X_train,Y_train,batch_size=32,epochs=10, class_weight = 'auto',
                                     validation_data=(X_validation,Y_validation),
                                     callbacks=[early],
                                     verbose=1)

scores = model_FirstLevelGrouping.evaluate(X_validation, Y_validation, verbose=1)

y = model_FirstLevelGrouping.predict(X_train)
Train on 7477 samples, validate on 30 samples
Epoch 1/10
7477/7477 [==============================] - 199s 27ms/step - loss: 0.5639 - acc: 0.7185 - val_loss: 0.5870 - val_acc: 0.6333
Epoch 2/10
7477/7477 [==============================] - 193s 26ms/step - loss: 0.5024 - acc: 0.7535 - val_loss: 0.5777 - val_acc: 0.7667
Epoch 3/10
7477/7477 [==============================] - 197s 26ms/step - loss: 0.4604 - acc: 0.7899 - val_loss: 0.5904 - val_acc: 0.7333
Epoch 4/10
7477/7477 [==============================] - 198s 26ms/step - loss: 0.4246 - acc: 0.8129 - val_loss: 0.7611 - val_acc: 0.7333
Epoch 5/10
7477/7477 [==============================] - 198s 26ms/step - loss: 0.3999 - acc: 0.8247 - val_loss: 0.6424 - val_acc: 0.7667
Epoch 6/10
7477/7477 [==============================] - 196s 26ms/step - loss: 0.3777 - acc: 0.8374 - val_loss: 0.7329 - val_acc: 0.7333
Epoch 7/10
7477/7477 [==============================] - 194s 26ms/step - loss: 0.3480 - acc: 0.8541 - val_loss: 0.6139 - val_acc: 0.6667
30/30 [==============================] - 0s 10ms/step
In [0]:
predictedClass = np.argmax(y,axis=1).tolist() 
actualClass = np.argmax(Y_train,axis=1).tolist()
   
tempDf = pd.DataFrame()
tempDf['ActualValue'] = pd.Series(actualClass)
tempDf['PredictedClass'] = pd.Series(predictedClass)

model_FirstLevelGrouping.save("model_FirstLevelGrouping.h5")
In [0]:
print_scores(tempDf['ActualValue'], tempDf['PredictedClass'])
Accuracy score: 
 0.8748161027149927
Test-set confusion matrix:
 [[5019  236]
 [ 700 1522]]
Classification report:
               precision    recall  f1-score   support

           0       0.88      0.96      0.91      5255
           1       0.87      0.68      0.76      2222

    accuracy                           0.87      7477
   macro avg       0.87      0.82      0.84      7477
weighted avg       0.87      0.87      0.87      7477


In [0]:
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])


In [0]:

SECOND MODEL BEGINS HERE FOR GROUP L12 (0) - All assignment groups that falls under L12 defined above
In [0]:
# get the train test data for the actual assignment groups
X_train_Zero, Y_train_Zero, X_validation_Zero, Y_validation_Zero,X_train_One, X_validation_One, Y_train_One, Y_validation_One, partDfZero, partDfOne, classweightZero, classweightOne  = PrepDataForModel_SecondLevelGrouping(df, maxlen)
In [0]:
# SECOND MODEL BEGINS HERE FOR GROUP 1 - All assignment groups that falls under L12 defined above

model_L12_AssignmentGroups = Sequential()
embedding=Embedding(len(embeddings),300,weights=[embeddings], input_length=maxlen,trainable=False)

model_L12_AssignmentGroups.add(embedding)

model_L12_AssignmentGroups.add(Bidirectional(LSTM(maxlen, return_sequences=True, recurrent_dropout=0.5)))
model_L12_AssignmentGroups.add(Bidirectional(LSTM(maxlen, return_sequences=False)))

#model_L12_AssignmentGroups.add(LSTM(maxlen))
#model_L12_AssignmentGroups.add(Dense(100, activation='relu'))
model_L12_AssignmentGroups.add(Dropout(0.5) )
#model_L12_AssignmentGroups.add(Dense(80, activation='relu'))


model_L12_AssignmentGroups.add(Dense(len(L12), activation='softmax'))
model_L12_AssignmentGroups.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
In [0]:
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

history=model_L12_AssignmentGroups.fit(X_train_Zero,Y_train_Zero,batch_size=32,epochs=10, class_weight = classweightZero,
                                                   validation_data=(X_validation_Zero,Y_validation_Zero),
                                                   callbacks=[early], 
                                                   verbose=1)

scores = model_L12_AssignmentGroups.evaluate(X_validation_Zero, Y_validation_Zero, verbose=1)
model_L12_AssignmentGroups.save("model_L12_AssignmentGroups.h5")
Train on 3691 samples, validate on 1582 samples
Epoch 1/10
3691/3691 [==============================] - 116s 31ms/step - loss: 0.1913 - acc: 0.9339 - val_loss: 0.1670 - val_acc: 0.9398
Epoch 2/10
3691/3691 [==============================] - 112s 30ms/step - loss: 0.1465 - acc: 0.9471 - val_loss: 0.1475 - val_acc: 0.9457
Epoch 3/10
3691/3691 [==============================] - 111s 30ms/step - loss: 0.1275 - acc: 0.9510 - val_loss: 0.1365 - val_acc: 0.9490
Epoch 4/10
3691/3691 [==============================] - 112s 30ms/step - loss: 0.1205 - acc: 0.9530 - val_loss: 0.1307 - val_acc: 0.9497
Epoch 5/10
3691/3691 [==============================] - 109s 30ms/step - loss: 0.1121 - acc: 0.9560 - val_loss: 0.1324 - val_acc: 0.9486
Epoch 6/10
3691/3691 [==============================] - 108s 29ms/step - loss: 0.1065 - acc: 0.9584 - val_loss: 0.1395 - val_acc: 0.9486
Epoch 7/10
3691/3691 [==============================] - 108s 29ms/step - loss: 0.1009 - acc: 0.9600 - val_loss: 0.1315 - val_acc: 0.9499
Epoch 8/10
3691/3691 [==============================] - 108s 29ms/step - loss: 0.0953 - acc: 0.9617 - val_loss: 0.1408 - val_acc: 0.9493
Epoch 9/10
3691/3691 [==============================] - 109s 29ms/step - loss: 0.0931 - acc: 0.9609 - val_loss: 0.1334 - val_acc: 0.9502
1582/1582 [==============================] - 15s 9ms/step
In [0]:
y = model_L12_AssignmentGroups.predict(X_train_Zero)
predictedClass = np.argmax(y,axis=1).tolist()
actualClass = np.argmax(Y_train_Zero,axis=1).tolist()
    
tempDf_L12_AssignmentGroups = pd.DataFrame()
tempDf_L12_AssignmentGroups['ActualValue'] = pd.Series(actualClass)
tempDf_L12_AssignmentGroups['PredictedClass'] = pd.Series(predictedClass)
print_scores(tempDf_L12_AssignmentGroups['ActualValue'], tempDf_L12_AssignmentGroups['PredictedClass'])
Accuracy score: 
 0.8721213763207802
Test-set confusion matrix:
 [[2313   15    4    9    4    0    2    1]
 [  12  136    0    1    0    0   10    1]
 [  71    5   61    1    6    0    0    1]
 [  42    4    0  123    1    0    0    0]
 [  62    3   15    3   46    0    0    0]
 [   0    0    0    0    0   64   42    9]
 [   3    3    0    0    0    3  424   15]
 [  10    1    0    0    0    4  109   52]]
Classification report:
               precision    recall  f1-score   support

           0       0.92      0.99      0.95      2348
           1       0.81      0.85      0.83       160
           2       0.76      0.42      0.54       145
           3       0.90      0.72      0.80       170
           4       0.81      0.36      0.49       129
           5       0.90      0.56      0.69       115
           6       0.72      0.95      0.82       448
           7       0.66      0.30      0.41       176

    accuracy                           0.87      3691
   macro avg       0.81      0.64      0.69      3691
weighted avg       0.87      0.87      0.86      3691


In [0]:
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])


SECOND MODEL BEGINS HERE FOR GROUP L3 (1) - All assignment groups that falls under L3 defined above
In [0]:
# SECOND MODEL BEGINS HERE FOR GROUP 2  - All assignment groups that falls under L3 defined above

model_L3_AssignmentGroups = Sequential()
embedding = Embedding(len(embeddings),300,weights=[embeddings], input_length=maxlen, trainable=False)

model_L3_AssignmentGroups.add(embedding)

#model_L3_AssignmentGroups.add(LSTM(maxlen), recurrent_dropout=0.25)


model_L3_AssignmentGroups.add(Bidirectional(LSTM(maxlen, return_sequences=True, recurrent_dropout=0.3)))
model_L3_AssignmentGroups.add(Bidirectional(LSTM(maxlen, return_sequences=False)))

model_L3_AssignmentGroups.add(Dense(100, activation='relu'))
model_L3_AssignmentGroups.add(Dropout(0.25) )

#L3 subgroups
model_L3_AssignmentGroups.add(Dense(len(L3), activation='softmax'))
model_L3_AssignmentGroups.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
In [0]:
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)


history=model_L3_AssignmentGroups.fit(X_train_One,Y_train_One, batch_size=10,epochs=10, class_weight = classweightOne,
                                      validation_data=(X_validation_One,Y_validation_One),
                                      callbacks=[early], 
                                      verbose=1)

scores = model_L3_AssignmentGroups.evaluate(X_validation_One, Y_validation_One, verbose=1)

model_L3_AssignmentGroups.save("model_L3_AssignmentGroups.h5")
Train on 1563 samples, validate on 671 samples
Epoch 1/10
1563/1563 [==============================] - 153s 98ms/step - loss: 0.0664 - acc: 0.9850 - val_loss: 0.0557 - val_acc: 0.9861
Epoch 2/10
1563/1563 [==============================] - 152s 97ms/step - loss: 0.0539 - acc: 0.9862 - val_loss: 0.0493 - val_acc: 0.9870
Epoch 3/10
1563/1563 [==============================] - 156s 100ms/step - loss: 0.0477 - acc: 0.9871 - val_loss: 0.0459 - val_acc: 0.9870
Epoch 4/10
1563/1563 [==============================] - 154s 98ms/step - loss: 0.0428 - acc: 0.9878 - val_loss: 0.0436 - val_acc: 0.9876
Epoch 5/10
1563/1563 [==============================] - 152s 97ms/step - loss: 0.0386 - acc: 0.9884 - val_loss: 0.0429 - val_acc: 0.9874
Epoch 6/10
1563/1563 [==============================] - 149s 96ms/step - loss: 0.0369 - acc: 0.9887 - val_loss: 0.0425 - val_acc: 0.9883
Epoch 7/10
1563/1563 [==============================] - 148s 95ms/step - loss: 0.0320 - acc: 0.9897 - val_loss: 0.0414 - val_acc: 0.9886
Epoch 8/10
1563/1563 [==============================] - 147s 94ms/step - loss: 0.0295 - acc: 0.9905 - val_loss: 0.0439 - val_acc: 0.9881
Epoch 9/10
1563/1563 [==============================] - 147s 94ms/step - loss: 0.0270 - acc: 0.9909 - val_loss: 0.0431 - val_acc: 0.9882
Epoch 10/10
1563/1563 [==============================] - 147s 94ms/step - loss: 0.0248 - acc: 0.9917 - val_loss: 0.0426 - val_acc: 0.9887
671/671 [==============================] - 6s 9ms/step
In [0]:
y = model_L3_AssignmentGroups.predict(X_train_One)
predictedClass = np.argmax(y,axis=1).tolist()
actualClass = np.argmax(Y_train_One,axis=1).tolist()
    
tempDf_L3_AssignmentGroups = pd.DataFrame()
tempDf_L3_AssignmentGroups['ActualValue'] = pd.Series(actualClass)
tempDf_L3_AssignmentGroups['PredictedClass'] = pd.Series(predictedClass)
In [0]:
print_scores(tempDf_L3_AssignmentGroups['ActualValue'], tempDf_L3_AssignmentGroups['PredictedClass'])
Accuracy score: 
 0.7594369801663468
Test-set confusion matrix:
 [[15  0  0 ...  0  0  0]
 [ 0 83  0 ...  0  0  0]
 [ 0  0 13 ...  0  0  0]
 ...
 [ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]]
Classification report:
               precision    recall  f1-score   support

           0       0.54      0.71      0.61        21
           1       0.65      0.86      0.74        96
           2       0.81      0.65      0.72        20
           3       0.69      0.92      0.79        96
           4       0.94      0.85      0.89        85
           5       0.86      0.60      0.71        30
           6       1.00      0.85      0.92        52
           7       1.00      1.00      1.00        52
           8       0.72      0.88      0.79        58
           9       0.75      0.11      0.19        27
          10       0.81      0.68      0.74        19
          11       0.50      0.90      0.64        21
          12       0.83      0.94      0.88        16
          13       0.91      0.98      0.95       116
          14       0.85      0.84      0.85        82
          15       0.69      0.89      0.78        37
          16       0.56      0.69      0.62        13
          17       0.86      0.59      0.70        32
          18       0.65      0.86      0.74        63
          19       0.00      0.00      0.00         3
          20       0.50      0.55      0.52        20
          21       0.00      0.00      0.00         3
          22       0.66      0.84      0.74        70
          23       0.66      0.74      0.70        39
          24       0.00      0.00      0.00         1
          25       0.67      0.25      0.36         8
          26       0.75      0.21      0.33        14
          27       0.00      0.00      0.00         3
          28       0.80      0.36      0.50        11
          29       0.97      0.96      0.97        73
          30       0.80      0.67      0.73        30
          31       0.96      0.90      0.93        29
          32       0.54      0.33      0.41        21
          33       0.00      0.00      0.00         3
          34       0.00      0.00      0.00         9
          35       0.67      0.21      0.32        28
          36       0.00      0.00      0.00         5
          37       0.45      0.47      0.46        19
          38       0.00      0.00      0.00         2
          39       0.00      0.00      0.00         3
          40       0.85      0.80      0.83        92
          41       0.78      0.70      0.74        10
          42       1.00      0.71      0.83         7
          43       1.00      0.22      0.36         9
          44       1.00      0.14      0.25         7
          46       0.75      0.50      0.60         6
          47       0.00      0.00      0.00         2
          48       0.00      0.00      0.00         1
          49       0.00      0.00      0.00         3
          50       0.00      0.00      0.00         2
          51       0.88      0.44      0.58        16
          52       0.00      0.00      0.00         1
          53       0.47      0.90      0.62        10
          54       0.00      0.00      0.00         1
          56       0.17      0.22      0.19         9
          57       0.00      0.00      0.00         1
          59       0.00      0.00      0.00         2
          60       0.00      0.00      0.00         1
          61       0.69      0.94      0.80        50
          62       0.00      0.00      0.00         1
          63       0.00      0.00      0.00         1
          64       0.00      0.00      0.00         1

    accuracy                           0.76      1563
   macro avg       0.49      0.43      0.44      1563
weighted avg       0.76      0.76      0.74      1563


In [0]:
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])


In [0]:
PickleCustomObjects(embeddings,vocabulary,inverse_vocabulary)
Random Forest Method
In [0]:
df = pd.read_excel('EDA_Cleaned_PreProcessed.xlsx')
dfModelBuilding = df[['LemmaString', 'AssignmentGroup']]
dfModelBuilding.head()
Out[0]:
LemmaString	AssignmentGroup
0	login issue verified user details employee m...	GRP_0
1	outlook hmjdrvpb komuaywn team meetings skyp...	GRP_0
2	log vpn eylqgodm ybqkwiam i not	GRP_0
3	unable access hr tool	GRP_0
4	skype error	GRP_0
In [0]:
dfModelBuilding = dfModelBuilding.dropna()
dfModelBuilding = dfModelBuilding.reset_index(drop=True)

cols = ['LemmaString','AssignmentGroup']
dfModelBuilding[cols] = dfModelBuilding[cols].fillna('no data')
In [0]:
dfModelBuilding.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7504 entries, 0 to 7503
Data columns (total 2 columns):
LemmaString        7504 non-null object
AssignmentGroup    7504 non-null object
dtypes: object(2)
memory usage: 117.4+ KB
In [0]:
dfModelBuilding['L12Grp'] = dfModelBuilding['AssignmentGroup'].apply(lambda x : SetGrp(x))
In [0]:
dfModelBuilding.head()
Out[0]:
LemmaString	AssignmentGroup	Level1Grp	L12Grp
0	login issue verified user details employee m...	GRP_0	0	0
1	outlook hmjdrvpb komuaywn team meetings skyp...	GRP_0	0	0
2	log vpn eylqgodm ybqkwiam i not	GRP_0	0	0
3	unable access hr tool	GRP_0	0	0
4	skype error	GRP_0	0	0
In [0]:
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
  lb = preprocessing.LabelBinarizer()
  lb.fit(y_test)
  y_test = lb.transform(y_test)
  y_pred = lb.transform(y_pred)
  return roc_auc_score(y_test, y_pred, average=average)
In [0]:
dfModelBuilding_1 = dfModelBuilding[['LemmaString','AssignmentGroup','L12Grp']]
dfModelBuilding_1.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7504 entries, 0 to 7503
Data columns (total 3 columns):
LemmaString        7504 non-null object
AssignmentGroup    7504 non-null object
L12Grp             7504 non-null int64
dtypes: int64(1), object(2)
memory usage: 176.0+ KB
Random Forest - Train Random Forest for L12 and L3 groups
In [0]:
data_x = dfModelBuilding_1['LemmaString'].values
data_y = dfModelBuilding_1['L12Grp'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in skf.split(data_x, data_y):
    x_train, x_test = data_x[train_index], data_x[test_index]
    y_train, y_test = data_y[train_index], data_y[test_index]

    clf=pipeline.Pipeline([
        ('tfidf_vectorizer', feature_extraction.text.TfidfVectorizer(lowercase=True)),
        ('rf_classifier', ensemble.RandomForestClassifier(n_estimators=500,verbose=1,n_jobs=-1))
    ])

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
    y_pred_prob = clf.predict_proba(dfModelBuilding_1['LemmaString'])
    print_scores(y_test, y_pred, y_pred_prob)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    2.4s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:   10.4s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:   23.4s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:   26.3s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.5s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
0.8067954696868754
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.7s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    1.5s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    1.7s finished
Accuracy score: 
 0.8067954696868754
Test-set confusion matrix:
 [[980  75]
 [215 231]]
Classification report:
               precision    recall  f1-score   support

           0       0.82      0.93      0.87      1055
           1       0.75      0.52      0.61       446

    accuracy                           0.81      1501
   macro avg       0.79      0.72      0.74      1501
weighted avg       0.80      0.81      0.79      1501

[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    2.4s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:   10.3s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:   23.4s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:   26.2s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.5s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
0.8081279147235176
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.7s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    1.5s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    1.7s finished
Accuracy score: 
 0.8081279147235176
Test-set confusion matrix:
 [[992  63]
 [225 221]]
Classification report:
               precision    recall  f1-score   support

           0       0.82      0.94      0.87      1055
           1       0.78      0.50      0.61       446

    accuracy                           0.81      1501
   macro avg       0.80      0.72      0.74      1501
weighted avg       0.80      0.81      0.79      1501

[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    2.5s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:   10.3s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:   23.2s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:   26.0s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.5s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
0.8147901399067289
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.7s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    1.5s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    1.6s finished
Accuracy score: 
 0.8147901399067289
Test-set confusion matrix:
 [[996  59]
 [219 227]]
Classification report:
               precision    recall  f1-score   support

           0       0.82      0.94      0.88      1055
           1       0.79      0.51      0.62       446

    accuracy                           0.81      1501
   macro avg       0.81      0.73      0.75      1501
weighted avg       0.81      0.81      0.80      1501

[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    2.5s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:   10.3s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:   23.5s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:   26.3s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.5s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
0.8134576948700866
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.7s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    1.5s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    1.6s finished
Accuracy score: 
 0.8134576948700866
Test-set confusion matrix:
 [[1006   48]
 [ 232  215]]
Classification report:
               precision    recall  f1-score   support

           0       0.81      0.95      0.88      1054
           1       0.82      0.48      0.61       447

    accuracy                           0.81      1501
   macro avg       0.82      0.72      0.74      1501
weighted avg       0.81      0.81      0.80      1501

[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    2.4s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:   10.5s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:   23.7s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:   26.5s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.5s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
0.7993333333333333
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.6s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    1.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    1.6s finished
Accuracy score: 
 0.7993333333333333
Test-set confusion matrix:
 [[974  80]
 [221 225]]
Classification report:
               precision    recall  f1-score   support

           0       0.82      0.92      0.87      1054
           1       0.74      0.50      0.60       446

    accuracy                           0.80      1500
   macro avg       0.78      0.71      0.73      1500
weighted avg       0.79      0.80      0.79      1500


In [0]:
#Seperate Class 0 and Class 1 into different data frames

partDfZero = dfModelBuilding.loc[dfModelBuilding['L12Grp'] == 0]
partDfOne = dfModelBuilding.loc[dfModelBuilding['L12Grp'] == 1]
In [0]:
le1 = LabelEncoder()
le2 = LabelEncoder()

partDfZero['AssignmentGroup'] = le1.fit_transform(partDfZero['AssignmentGroup'])
partDfOne['AssignmentGroup'] = le2.fit_transform(partDfOne['AssignmentGroup'])
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  after removing the cwd from sys.path.
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  """
Use Stratified K Fold and Random Forest on to train/predict L12 group sub-groups
In [0]:
data_x2 = partDfZero['LemmaString'].values
data_y2 = partDfZero['AssignmentGroup'].values

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
for train_index, test_index in skf.split(data_x2, data_y2):
    x_train2, x_test2 = data_x2[train_index], data_x2[test_index]
    y_train2, y_test2 = data_y2[train_index], data_y2[test_index]

    clf2=pipeline.Pipeline([
        ('tfidf_vectorizer', feature_extraction.text.TfidfVectorizer(lowercase=True)),
        ('rf_classifier', ensemble.RandomForestClassifier(n_estimators=500,verbose=1,n_jobs=-1))
    ])
    clf2.fit(x_train2, y_train2)
    y_pred2 = clf2.predict(x_test2)
    y_pred_prob2 = clf2.predict_proba(x_test2)
    score2 = accuracy_score(y_test2, y_pred2)
    print(score2)
    print_scores(y_test2, y_pred2, y_pred_prob2)
    print("ROC_AUC_Score: ", multiclass_roc_auc_score(y_test2, y_pred2))
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    0.9s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:    3.7s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:    8.3s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:    9.2s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.4s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.4s finished
0.8026166097838453
Accuracy score: 
 0.8026166097838453
Test-set confusion matrix:
 [[1088    6    1    1    5    1    0    0]
 [  27   52    0    1    0    0    3    0]
 [  62    1    3    0    1    0    0    0]
 [  52    0    0   26    1    0    0    0]
 [  50    0    3    2   10    0    0    0]
 [   9    0    0    0    0   19   22    9]
 [   8    7    0    0    0    1  196    8]
 [  19    1    0    0    0    0   46   17]]
Classification report:
               precision    recall  f1-score   support

           0       0.83      0.99      0.90      1102
           1       0.78      0.63      0.69        83
           2       0.43      0.04      0.08        67
           3       0.87      0.33      0.48        79
           4       0.59      0.15      0.24        65
           5       0.90      0.32      0.48        59
           6       0.73      0.89      0.80       220
           7       0.50      0.20      0.29        83

    accuracy                           0.80      1758
   macro avg       0.70      0.44      0.50      1758
weighted avg       0.78      0.80      0.76      1758

ROC_AUC_Score:  0.6961209832162467
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    0.9s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:    3.7s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:    8.4s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:    9.4s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.4s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.4s finished
0.8048919226393629
Accuracy score: 
 0.8048919226393629
Test-set confusion matrix:
 [[1088   10    1    0    1    2    0    0]
 [  23   51    0    2    0    0    7    0]
 [  61    0    3    0    2    0    1    0]
 [  50    0    0   28    1    0    0    0]
 [  51    0    1    0   12    0    0    0]
 [   4    0    0    0    0   27   21    8]
 [  11    7    0    0    0    1  190   10]
 [  21    0    0    0    0    0   47   16]]
Classification report:
               precision    recall  f1-score   support

           0       0.83      0.99      0.90      1102
           1       0.75      0.61      0.68        83
           2       0.60      0.04      0.08        67
           3       0.93      0.35      0.51        79
           4       0.75      0.19      0.30        64
           5       0.90      0.45      0.60        60
           6       0.71      0.87      0.78       219
           7       0.47      0.19      0.27        84

    accuracy                           0.80      1758
   macro avg       0.74      0.46      0.52      1758
weighted avg       0.79      0.80      0.77      1758

ROC_AUC_Score:  0.705177461041369
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    0.8s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:    3.6s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:    8.1s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:    9.1s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.4s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.4s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.4s finished
0.8002276607854297
Accuracy score: 
 0.8002276607854297
Test-set confusion matrix:
 [[1090    4    0    1    3    0    2    1]
 [  33   45    0    1    0    0    4    0]
 [  57    0    4    1    4    0    2    0]
 [  55    0    0   23    0    0    0    0]
 [  54    0    0    2    9    0    0    0]
 [   8    0    0    0    0   23   25    4]
 [   8    2    0    0    0    2  198    9]
 [  25    0    0    0    0    0   44   14]]
Classification report:
               precision    recall  f1-score   support

           0       0.82      0.99      0.90      1101
           1       0.88      0.54      0.67        83
           2       1.00      0.06      0.11        68
           3       0.82      0.29      0.43        78
           4       0.56      0.14      0.22        65
           5       0.92      0.38      0.54        60
           6       0.72      0.90      0.80       219
           7       0.50      0.17      0.25        83

    accuracy                           0.80      1757
   macro avg       0.78      0.44      0.49      1757
weighted avg       0.80      0.80      0.76      1757

ROC_AUC_Score:  0.6902682814140875

Random Forest and Stratified K Fold to train/predict L3 subgroups
In [0]:
data_x3 = partDfOne['LemmaString'].values
data_y3 = partDfOne['AssignmentGroup'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in skf.split(data_x3, data_y3):
    x_train3, x_test3 = data_x3[train_index], data_x3[test_index]
    y_train3, y_test3 = data_y3[train_index], data_y3[test_index]

    clf3=pipeline.Pipeline([
        ('tfidf_vectorizer', feature_extraction.text.TfidfVectorizer(lowercase=True)),
        ('rf_classifier', ensemble.RandomForestClassifier(n_estimators=500,verbose=1,n_jobs=-1))
    ])
    clf3.fit(x_train3, y_train3)
    y_pred3 = clf3.predict(x_test3)
    y_pred_prob3 = clf3.predict_proba(x_test3)
    score3 = accuracy_score(y_test3, y_pred3)
    print(score3)
    print_scores(y_test3, y_pred3, y_pred_prob3)
    print("ROC_AUC_Score: ", multiclass_roc_auc_score(y_test3, y_pred3))
/usr/local/lib/python3.6/dist-packages/sklearn/model_selection/_split.py:667: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
  % (min_groups, self.n_splits)), UserWarning)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    0.8s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:    3.3s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:    7.3s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:    8.2s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.3s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.3s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
0.5928411633109619
Accuracy score: 
 0.5928411633109619
Test-set confusion matrix:
 [[ 0  0  0 ...  0  0  0]
 [ 0 18  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]
 ...
 [ 0  0  0 ... 10  0  0]
 [ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]]
/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00         6
           1       0.67      0.64      0.65        28
           2       0.00      0.00      0.00         6
           3       0.41      0.79      0.54        29
           4       0.55      0.74      0.63        23
           5       0.67      0.29      0.40         7
           6       0.79      0.88      0.83        17
           7       0.83      1.00      0.91        15
           8       0.64      0.53      0.58        17
           9       0.00      0.00      0.00         7
          10       0.00      0.00      0.00         6
          11       1.00      0.17      0.29         6
          12       0.67      0.80      0.73         5
          13       0.65      0.86      0.74        35
          14       0.53      0.87      0.66        23
          15       0.40      0.55      0.46        11
          16       0.00      0.00      0.00         3
          17       0.43      0.33      0.38         9
          18       0.59      0.53      0.56        19
          19       0.00      0.00      0.00         0
          20       0.33      0.60      0.43         5
          21       0.00      0.00      0.00         1
          22       0.44      0.40      0.42        20
          23       0.73      0.73      0.73        11
          24       0.00      0.00      0.00         1
          25       1.00      0.50      0.67         2
          26       0.00      0.00      0.00         3
          27       0.00      0.00      0.00         1
          28       1.00      0.50      0.67         4
          29       0.75      0.90      0.82        20
          30       0.33      0.56      0.42         9
          31       0.86      0.75      0.80         8
          32       0.00      0.00      0.00         6
          33       0.00      0.00      0.00         1
          34       0.00      0.00      0.00         3
          35       0.00      0.00      0.00         7
          36       0.00      0.00      0.00         1
          37       1.00      0.50      0.67         6
          38       0.00      0.00      0.00         0
          39       0.00      0.00      0.00         1
          40       0.84      0.81      0.82        26
          41       1.00      0.50      0.67         2
          42       1.00      0.50      0.67         2
          43       0.00      0.00      0.00         2
          44       1.00      0.50      0.67         2
          46       0.67      1.00      0.80         2
          47       0.00      0.00      0.00         1
          49       1.00      1.00      1.00         1
          50       0.00      0.00      0.00         1
          51       1.00      0.25      0.40         4
          53       0.00      0.00      0.00         2
          54       0.00      0.00      0.00         1
          56       0.00      0.00      0.00         2
          57       0.00      0.00      0.00         1
          60       0.00      0.00      0.00         1
          61       0.67      0.77      0.71        13
          62       0.00      0.00      0.00         1
          65       0.00      0.00      0.00         1

    accuracy                           0.59       447
   macro avg       0.39      0.34      0.34       447
weighted avg       0.55      0.59      0.55       447

ROC_AUC_Score:  0.6724188922824546
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    0.7s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:    3.2s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:    7.2s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:    8.1s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.3s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.3s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
0.600896860986547
Accuracy score: 
 0.600896860986547
Test-set confusion matrix:
 [[ 2  0  0 ...  0  0  0]
 [ 0 14  0 ...  0  1  0]
 [ 0  0  0 ...  0  0  0]
 ...
 [ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  9  0]
 [ 0  0  0 ...  0  0  0]]
/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report:
               precision    recall  f1-score   support

           0       0.67      0.33      0.44         6
           1       0.67      0.50      0.57        28
           2       0.00      0.00      0.00         6
           3       0.36      0.79      0.49        28
           4       0.67      0.58      0.62        24
           5       1.00      0.71      0.83         7
           6       1.00      0.76      0.87        17
           7       0.94      1.00      0.97        15
           8       0.59      0.59      0.59        17
           9       0.00      0.00      0.00         7
          10       0.50      0.17      0.25         6
          11       0.50      0.67      0.57         6
          12       0.83      1.00      0.91         5
          13       0.72      0.94      0.81        35
          14       0.50      0.83      0.62        23
          15       0.86      0.60      0.71        10
          16       0.00      0.00      0.00         4
          17       0.25      0.12      0.17         8
          18       0.58      0.58      0.58        19
          19       0.00      0.00      0.00         0
          20       0.28      0.83      0.42         6
          21       0.00      0.00      0.00         1
          22       0.68      0.65      0.67        20
          23       0.70      0.64      0.67        11
          25       0.00      0.00      0.00         3
          26       0.00      0.00      0.00         3
          27       0.00      0.00      0.00         1
          28       0.00      0.00      0.00         4
          29       0.60      0.75      0.67        20
          30       0.75      0.38      0.50         8
          31       1.00      0.88      0.93         8
          32       1.00      0.14      0.25         7
          33       0.00      0.00      0.00         1
          34       0.00      0.00      0.00         3
          35       0.33      0.14      0.20         7
          36       0.00      0.00      0.00         1
          37       0.50      0.40      0.44         5
          38       0.00      0.00      0.00         1
          39       0.00      0.00      0.00         1
          40       0.61      0.85      0.71        26
          41       1.00      0.33      0.50         3
          42       1.00      0.50      0.67         2
          43       0.00      0.00      0.00         1
          44       0.00      0.00      0.00         3
          46       0.50      1.00      0.67         1
          47       0.00      0.00      0.00         1
          49       1.00      1.00      1.00         1
          50       1.00      1.00      1.00         1
          51       1.00      0.75      0.86         4
          53       0.00      0.00      0.00         2
          55       0.00      0.00      0.00         1
          56       0.00      0.00      0.00         2
          58       0.00      0.00      0.00         1
          60       0.00      0.00      0.00         1
          61       0.64      0.69      0.67        13
          63       0.00      0.00      0.00         1

    accuracy                           0.60       446
   macro avg       0.41      0.38      0.37       446
weighted avg       0.59      0.60      0.57       446

ROC_AUC_Score:  0.6881036815256089
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    0.8s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:    3.2s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:    7.3s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:    8.2s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.3s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.3s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
0.5717488789237668
Accuracy score: 
 0.5717488789237668
Test-set confusion matrix:
 [[ 1  0  0 ...  0  0  0]
 [ 0 16  0 ...  0  0  0]
 [ 0  2  0 ...  0  0  0]
 ...
 [ 0  0  0 ...  1  0  0]
 [ 0  0  0 ...  0 12  0]
 [ 0  0  0 ...  0  0  0]]
/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report:
               precision    recall  f1-score   support

           0       0.50      0.17      0.25         6
           1       0.53      0.57      0.55        28
           2       0.00      0.00      0.00         6
           3       0.45      0.82      0.58        28
           4       0.56      0.87      0.68        23
           5       0.33      0.25      0.29         8
           6       0.87      0.76      0.81        17
           7       1.00      0.93      0.96        14
           8       0.67      0.56      0.61        18
           9       0.00      0.00      0.00         7
          10       1.00      0.20      0.33         5
          11       0.50      0.57      0.53         7
          12       1.00      1.00      1.00         5
          13       0.66      0.94      0.78        35
          14       0.56      0.68      0.61        22
          15       0.33      0.36      0.35        11
          16       0.00      0.00      0.00         4
          17       0.20      0.12      0.15         8
          18       0.58      0.74      0.65        19
          19       0.00      0.00      0.00         1
          20       0.14      0.50      0.21         6
          21       0.00      0.00      0.00         1
          22       0.52      0.55      0.54        20
          23       0.71      0.45      0.56        11
          25       1.00      0.33      0.50         3
          26       0.00      0.00      0.00         3
          27       0.00      0.00      0.00         1
          28       0.75      1.00      0.86         3
          29       0.62      0.65      0.63        20
          30       0.45      0.56      0.50         9
          31       1.00      0.50      0.67         8
          32       1.00      0.14      0.25         7
          33       0.00      0.00      0.00         1
          34       0.00      0.00      0.00         3
          35       0.00      0.00      0.00         7
          36       0.00      0.00      0.00         1
          37       0.00      0.00      0.00         5
          38       0.00      0.00      0.00         1
          39       0.00      0.00      0.00         1
          40       0.74      0.68      0.71        25
          41       0.00      0.00      0.00         3
          42       1.00      1.00      1.00         1
          43       0.00      0.00      0.00         2
          44       0.00      0.00      0.00         2
          45       0.00      0.00      0.00         1
          46       1.00      1.00      1.00         1
          48       0.00      0.00      0.00         1
          50       0.00      0.00      0.00         2
          51       1.00      0.25      0.40         4
          53       0.00      0.00      0.00         2
          56       0.00      0.00      0.00         2
          57       1.00      1.00      1.00         1
          59       1.00      1.00      1.00         1
          61       0.52      0.86      0.65        14
          63       0.00      0.00      0.00         1

    accuracy                           0.57       446
   macro avg       0.40      0.36      0.36       446
weighted avg       0.54      0.57      0.53       446

ROC_AUC_Score:  0.6779509885014093
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    0.7s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:    3.1s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:    7.1s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:    7.9s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.3s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.2s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
0.5650224215246636
Accuracy score: 
 0.5650224215246636
Test-set confusion matrix:
 [[ 0  0  0 ...  0  0  0]
 [ 0 16  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]
 ...
 [ 0  1  0 ...  0  0  0]
 [ 0  0  0 ...  0 13  0]
 [ 0  0  0 ...  0  0  0]]
/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00         5
           1       0.50      0.57      0.53        28
           2       0.00      0.00      0.00         6
           3       0.43      0.71      0.53        28
           4       0.38      0.48      0.42        23
           5       0.86      0.75      0.80         8
           6       0.72      0.81      0.76        16
           7       0.88      1.00      0.94        15
           8       0.67      0.71      0.69        17
           9       0.00      0.00      0.00         7
          10       0.00      0.00      0.00         6
          11       0.33      0.33      0.33         6
          12       1.00      1.00      1.00         5
          13       0.74      0.81      0.77        36
          14       0.53      0.86      0.66        22
          15       0.55      0.55      0.55        11
          16       0.00      0.00      0.00         3
          17       0.50      0.44      0.47         9
          18       0.69      0.58      0.63        19
          19       0.00      0.00      0.00         1
          20       0.08      0.20      0.11         5
          22       0.42      0.48      0.44        21
          23       0.55      0.55      0.55        11
          25       1.00      1.00      1.00         3
          26       0.67      0.67      0.67         3
          27       0.00      0.00      0.00         0
          28       0.50      0.25      0.33         4
          29       0.71      0.60      0.65        20
          30       0.56      0.56      0.56         9
          31       0.67      0.75      0.71         8
          32       0.00      0.00      0.00         7
          33       0.00      0.00      0.00         1
          34       0.00      0.00      0.00         3
          35       0.00      0.00      0.00         6
          36       0.00      0.00      0.00         2
          37       0.67      0.40      0.50         5
          38       0.00      0.00      0.00         1
          39       0.00      0.00      0.00         1
          40       0.77      0.65      0.71        26
          41       0.00      0.00      0.00         3
          42       0.00      0.00      0.00         1
          43       0.00      0.00      0.00         2
          44       0.00      0.00      0.00         2
          45       0.00      0.00      0.00         1
          46       0.50      1.00      0.67         1
          48       0.00      0.00      0.00         1
          50       0.00      0.00      0.00         1
          51       1.00      0.75      0.86         4
          52       0.00      0.00      0.00         1
          53       0.00      0.00      0.00         2
          54       0.00      0.00      0.00         1
          56       0.00      0.00      0.00         2
          57       1.00      1.00      1.00         1
          59       0.00      0.00      0.00         1
          61       0.54      0.93      0.68        14
          64       0.00      0.00      0.00         1

    accuracy                           0.57       446
   macro avg       0.33      0.35      0.33       446
weighted avg       0.51      0.57      0.53       446

ROC_AUC_Score:  0.6720704806486808
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:    0.8s
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:    3.2s
[Parallel(n_jobs=-1)]: Done 446 tasks      | elapsed:    7.3s
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:    8.1s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.3s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
[Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.1s
[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.3s
[Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.3s finished
0.5627802690582959
Accuracy score: 
 0.5627802690582959
Test-set confusion matrix:
 [[ 1  0  0 ...  0  0  0]
 [ 0 11  0 ...  0  0  0]
 [ 0  0  1 ...  0  0  0]
 ...
 [ 0  0  0 ...  1  0  0]
 [ 0  1  0 ...  0 11  0]
 [ 0  0  0 ...  0  0  0]]
/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report:
               precision    recall  f1-score   support

           0       0.33      0.20      0.25         5
           1       0.69      0.41      0.51        27
           2       1.00      0.17      0.29         6
           3       0.43      0.79      0.55        29
           4       0.43      0.52      0.47        23
           5       0.67      0.29      0.40         7
           6       0.85      0.65      0.73        17
           7       0.94      1.00      0.97        15
           8       0.50      0.41      0.45        17
           9       0.00      0.00      0.00         7
          10       1.00      0.17      0.29         6
          11       0.67      0.33      0.44         6
          12       0.71      1.00      0.83         5
          13       0.62      0.74      0.68        35
          14       0.50      0.70      0.58        23
          15       0.58      0.64      0.61        11
          16       0.00      0.00      0.00         3
          17       0.27      0.33      0.30         9
          18       0.48      0.63      0.55        19
          19       0.00      0.00      0.00         1
          20       0.20      0.80      0.32         5
          21       0.00      0.00      0.00         1
          22       0.48      0.55      0.51        20
          23       0.64      0.58      0.61        12
          25       1.00      0.50      0.67         2
          26       1.00      0.75      0.86         4
          28       0.75      0.75      0.75         4
          29       0.79      0.75      0.77        20
          30       0.50      0.44      0.47         9
          31       0.86      0.75      0.80         8
          32       0.00      0.00      0.00         7
          33       0.00      0.00      0.00         1
          34       0.00      0.00      0.00         2
          35       0.00      0.00      0.00         7
          36       0.00      0.00      0.00         1
          37       0.57      0.67      0.62         6
          38       0.00      0.00      0.00         0
          39       0.00      0.00      0.00         1
          40       0.59      0.85      0.70        26
          41       0.00      0.00      0.00         2
          42       0.00      0.00      0.00         2
          43       0.00      0.00      0.00         2
          44       0.00      0.00      0.00         2
          46       1.00      0.50      0.67         2
          47       0.00      0.00      0.00         1
          49       1.00      1.00      1.00         1
          50       0.00      0.00      0.00         1
          51       0.50      0.50      0.50         4
          53       0.00      0.00      0.00         2
          54       0.00      0.00      0.00         1
          56       0.00      0.00      0.00         2
          57       0.00      0.00      0.00         1
          59       1.00      1.00      1.00         1
          61       0.79      0.79      0.79        14
          64       0.00      0.00      0.00         1

    accuracy                           0.56       446
   macro avg       0.41      0.37      0.36       446
weighted avg       0.55      0.56      0.53       446

ROC_AUC_Score:  0.6823725987988238

Model Evaluation
Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction
We have used stratified K fold cross validation technique to train our data. With stratified K fold, the splitting of data will ensure that each fold has the same proportion of observations with a given categorical value, in our case “L12Grp” and “Assignment Group”

image.png




The accuracy for L12 team is coming out to be fairly good around 80 % but for L3 teams it is coming out be as low as 56 %. This is because L3 contains a greater number of groups and not able to build an efficient classification model.

Learning from the approach
• For L3 model, for most of the groups the recall was coming out 0

• We observed that for stage 1 binary classification, random forest did well whereas for stage 2 multiclass classification LSTM performed better

• Skewed data is making our model not quite effective and need for oversampling has become a prime requirement.
Stacked RNN models performed good in training data where as we have not been able to achieve good accuracy with classical machine learning models.
To handle imbalance of data we have identified L12 and L3 team groups and decided to predict them separately.
As our next approach we will try oversampling out data. We will create entries for our minority classes and will oversample them to have fairly equal distribution in the data.

Approach 5 - Oversampling Approach
image.png

To cater the problem of imbalance data, we have tried another approach of handle our minority class i.e. to up-sample them to make our distribution across target variable more balanced.
There exist many oversampling techniques in the area of machine learning such as SMOTE (Synthetic Minority Over-Sampling Technique) or RandomOverSampler within imbalanced-learn package. Other techniques involve understanding the context of minority classes and creating sentences or jumbling or using key words from the data and generating new data.
Other techniques involve understanding the context of the text in minority classes and generating text of same context or jumbling the words or identifying key words from the data and build synthetic data based on it. But since our most of the minority classes i.e. groups have only one entry compared to thousands of entries of majority class, these techniques might not result in efficient balancing of the data. We will be using below technique to oversample our data:


Random oversampling for the minority class


Random oversampling simply replicates randomly the minority class examples. Random oversampling is known to increase the likelihood of occurring overfitting.
The increase in the number of examples for the minority class, especially if the class skew was severe, can also result in a marked increase in the computational cost when fitting the model, especially considering the model is seeing the same examples in the training dataset again and again.

In [0]:
df = pd.read_excel('EDA_Cleaned_PreProcessed.xlsx')
df.head()
Out[0]:
Assignment group	Complete_Description	Language	EnglishDescription	Tokens	LemmaWords	LemmaString
0	GRP_0	login issue -verified user details.(employee# ...	ENGLISH	login verified details employee manager checke...	['login', 'verified', 'details', 'employee', '...	['login', 'verified', 'details', 'employee', '...	login verified details employee manager chec...
1	GRP_0	outlook received from: hmjdrvpb.komuaywn@gmail...	ENGLISH	outlook hmjdrvpb komuaywn team meetings skype ...	['outlook', 'hmjdrvpb', 'komuaywn', 'team', 'm...	['outlook', 'hmjdrvpb', 'komuaywn', 'team', 'm...	outlook hmjdrvpb komuaywn team meetings skyp...
2	GRP_0	cant log in to vpn received from: eylqgodm.ybq...	ENGLISH	cant log vpn eylqgodm ybqkwiam cannot	['log', 'vpn', 'eylqgodm', 'ybqkwiam', 'not']	['log', 'vpn', 'eylqgodm', 'ybqkwiam', 'not']	log vpn eylqgodm ybqkwiam not
3	GRP_0	unable to access hr_tool page	ENGLISH	hr	['hr']	['hr']	hr
4	GRP_0	skype error	NORWEGIAN	skype	['skype']	['skype']	skype
In [0]:
final_df = df[['LemmaString','Assignment group']].copy()
In [0]:
final_df.head()
Out[0]:
LemmaString	Assignment group
0	login verified details employee manager chec...	GRP_0
1	outlook hmjdrvpb komuaywn team meetings skyp...	GRP_0
2	log vpn eylqgodm ybqkwiam not	GRP_0
3	hr	GRP_0
4	skype	GRP_0
In [0]:
df1 = final_df.copy()

df1['New group'] = LabelEncoder().fit_transform(df1['Assignment group'])
df1.head()
Out[0]:
LemmaString	Assignment group	New group
0	login verified details employee manager chec...	GRP_0	0
1	outlook hmjdrvpb komuaywn team meetings skyp...	GRP_0	0
2	log vpn eylqgodm ybqkwiam not	GRP_0	0
3	hr	GRP_0	0
4	skype	GRP_0	0
In [0]:
df2 = df1[['New group','LemmaString']].copy()
In [0]:
df2.head()
Out[0]:
New group	LemmaString
0	0	login verified details employee manager chec...
1	0	outlook hmjdrvpb komuaywn team meetings skyp...
2	0	log vpn eylqgodm ybqkwiam not
3	0	hr
4	0	skype
Model Building
In [0]:
max_features = 10000
embedding_size = 50
In [0]:
def create_corpus(df):
    corpus=[]
    for desc in tqdm(df['LemmaString'].astype(str)):
        words=[word.lower() for word in word_tokenize(desc) if((word.isalpha()==1))] # & (word not in stop_words))]
        corpus.append(words)
    return corpus
In [0]:
def create_inputvector(data,max_features,embedding_size,target):
  corpus = create_corpus(data)

  nb_words = []
  for i in range(len(data['LemmaString'])):
    nb_words.append(len(data['LemmaString'].iloc[i].split()))

  data['nb_words'] = nb_words
  print(data.columns)
  maxlen = max(data['nb_words'])

  tokenizer_obj=Tokenizer(num_words=max_features)
  tokenizer_obj.fit_on_texts(corpus)
  sequences=tokenizer_obj.texts_to_sequences(corpus)

  word_index = tokenizer_obj.word_index
  num_words = len(word_index)+1

  X = tokenizer_obj.texts_to_sequences(corpus)
  X = pad_sequences(X, maxlen = maxlen)
  y1 = data[target]
  y = to_categorical(y1, num_classes=74)
  return X,y,y1,maxlen,num_words
In [0]:
X,y,y1,maxlen,num_words=create_inputvector(data=df2,max_features=max_features,embedding_size=embedding_size,target='New group')
100%|██████████| 8500/8500 [00:01<00:00, 7871.30it/s]
Index(['New group', 'LemmaString', 'nb_words'], dtype='object')
Performing oversampling
In [0]:
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

ROS = RandomOverSampler(sampling_strategy='auto', random_state=42)

x_res, y_res = ROS.fit_sample(X, y1)

for i in range(y1.unique().shape[0]-2):
  ROS = RandomOverSampler(sampling_strategy='auto', random_state=42)
  x_res, y_res = ROS.fit_sample(x_res, y_res)
In [0]:
print ("Distribution of class labels before resampling {}".format(Counter(y1)))
print ("Distribution of class labels after resampling {}".format(Counter(y_res)))
Distribution of class labels before resampling Counter({0: 3976, 72: 661, 17: 289, 4: 257, 73: 252, 12: 241, 11: 215, 23: 200, 56: 184, 5: 145, 2: 140, 45: 129, 6: 118, 18: 116, 27: 107, 34: 100, 22: 97, 10: 88, 8: 85, 9: 81, 25: 69, 67: 68, 28: 62, 19: 56, 35: 45, 21: 44, 36: 40, 7: 39, 24: 39, 37: 37, 13: 36, 40: 35, 1: 31, 15: 31, 3: 30, 14: 29, 42: 27, 16: 25, 43: 25, 59: 25, 57: 20, 33: 19, 20: 18, 31: 16, 30: 15, 39: 15, 46: 14, 49: 11, 62: 11, 48: 9, 47: 8, 51: 8, 41: 6, 44: 6, 55: 6, 38: 5, 26: 4, 63: 4, 32: 3, 52: 3, 54: 3, 60: 3, 65: 3, 50: 2, 53: 2, 66: 2, 69: 2, 70: 2, 29: 1, 58: 1, 61: 1, 64: 1, 68: 1, 71: 1})
Distribution of class labels after resampling Counter({0: 3976, 1: 3976, 23: 3976, 34: 3976, 45: 3976, 56: 3976, 67: 3976, 72: 3976, 73: 3976, 2: 3976, 3: 3976, 4: 3976, 5: 3976, 6: 3976, 7: 3976, 8: 3976, 9: 3976, 10: 3976, 11: 3976, 12: 3976, 13: 3976, 14: 3976, 15: 3976, 16: 3976, 17: 3976, 18: 3976, 19: 3976, 20: 3976, 21: 3976, 22: 3976, 24: 3976, 25: 3976, 27: 3976, 28: 3976, 29: 3976, 30: 3976, 31: 3976, 32: 3976, 33: 3976, 35: 3976, 36: 3976, 37: 3976, 38: 3976, 39: 3976, 40: 3976, 41: 3976, 42: 3976, 43: 3976, 44: 3976, 46: 3976, 47: 3976, 48: 3976, 49: 3976, 50: 3976, 51: 3976, 52: 3976, 53: 3976, 54: 3976, 55: 3976, 57: 3976, 58: 3976, 26: 3976, 59: 3976, 60: 3976, 61: 3976, 62: 3976, 63: 3976, 64: 3976, 65: 3976, 66: 3976, 68: 3976, 69: 3976, 70: 3976, 71: 3976})
In [0]:
y_res1 = to_categorical(y_res, num_classes=74)
In [0]:
batch_size = 512
epochs = 5
In [0]:
def LSTM_model(maxlen,batch_size,epochs):
  main_input = Input(shape=(maxlen,))
  em = Embedding(max_features, 100, input_length=maxlen) (main_input)

  lstm_out1 = LSTM(128, return_sequences = True)(em)
  lstm_out2 = LSTM(128)(lstm_out1)

  x = Dropout(0.2)(lstm_out2)

  main_output = Dense(74, activation = 'softmax')(x)

  model = Model(inputs = main_input, outputs = main_output)
  # compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  checkpoint = ModelCheckpoint("model-{loss:.2f}.h5", monitor="loss", verbose=1, save_best_only=True,
                               save_weights_only=True, mode="min", period=1)
  stop = EarlyStopping(monitor="loss", patience=5, mode="min")
  reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="min")
  print(model.summary())
  return model,checkpoint,stop,reduce_lr
In [0]:
def plot_graph(history):
  fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
  fig.suptitle("Performance: LSTM Model")
  ax1.plot(history.history['acc'])
  ax1.plot(history.history['val_acc'])
  cut = np.where(history.history['val_acc'] == np.max(history.history['val_acc']))[0][0]
  ax1.axvline(x=cut, color='k', linestyle='--')
  ax1.set_title("Model Accuracy")
  ax1.legend(['train', 'test'])

  ax2.plot(history.history['loss'])
  ax2.plot(history.history['val_loss'])
  cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]
  ax2.axvline(x=cut, color='k', linestyle='--')
  ax2.set_title("Model Loss")
  ax2.legend(['train', 'test'])
In [0]:
skf = StratifiedKFold(n_splits=5,random_state=24, shuffle=True)
models, measures = [], []
for index, (train_indices, val_indices) in enumerate(skf.split(x_res, y_res)):
   ## Extract Kfold
  X_train_sfk, X_test_sfk = x_res[train_indices], x_res[val_indices]
  y_train_sfk, y_test_sfk = y_res[train_indices], y_res[val_indices]
In [0]:
y_train_sfk = to_categorical(y_train_sfk, num_classes= 74)
In [0]:
y_test_sfk = to_categorical(y_test_sfk,num_classes=74)
In [0]:
model2,checkpoint2,stop2,reduce_lr2 = LSTM_model(maxlen=maxlen,batch_size=batch_size,epochs=epochs)
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 415)               0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 415, 100)          1000000   
_________________________________________________________________
lstm_3 (LSTM)                (None, 415, 128)          117248    
_________________________________________________________________
lstm_4 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 74)                9546      
=================================================================
Total params: 1,258,378
Trainable params: 1,258,378
Non-trainable params: 0
_________________________________________________________________
None
In [0]:
history2 = model2.fit(X_train_sfk, y_train_sfk, validation_data=(X_test_sfk,y_test_sfk), batch_size=batch_size, epochs=epochs, callbacks=[stop2,reduce_lr2]) #, class_weight = 'auto')
Train on 235380 samples, validate on 58844 samples
Epoch 1/5
235380/235380 [==============================] - 5137s 22ms/step - loss: 1.4134 - acc: 0.6588 - val_loss: 0.3690 - val_acc: 0.9045
Epoch 2/5
235380/235380 [==============================] - 5115s 22ms/step - loss: 0.2993 - acc: 0.9167 - val_loss: 0.2455 - val_acc: 0.9286
Epoch 3/5
235380/235380 [==============================] - 5133s 22ms/step - loss: 0.2351 - acc: 0.9291 - val_loss: 0.2059 - val_acc: 0.9329
Epoch 4/5
235380/235380 [==============================] - 5089s 22ms/step - loss: 0.2080 - acc: 0.9340 - val_loss: 0.1960 - val_acc: 0.9371
Epoch 5/5
235380/235380 [==============================] - 5106s 22ms/step - loss: 0.1935 - acc: 0.9368 - val_loss: 0.1877 - val_acc: 0.9392
In [0]:
plot_graph(history2)

In [0]:
y = model2.predict(X_test_sfk)
predictedClass = np.argmax(y,axis=1).tolist()
actualClass = np.argmax(y_test_sfk,axis=1).tolist()

tempDf_L12_AssignmentGroups = pd.DataFrame()
tempDf_L12_AssignmentGroups['ActualValue'] = pd.Series(actualClass)
tempDf_L12_AssignmentGroups['PredictedClass'] = pd.Series(predictedClass)
print_scores(tempDf_L12_AssignmentGroups['ActualValue'], tempDf_L12_AssignmentGroups['PredictedClass'])
Accuracy score: 
 0.9349636326558358
Test-set confusion matrix:
 [[476   3  11 ...   0   8   2]
 [  0 795   0 ...   0   0   0]
 [  0   0 665 ...   0   0   0]
 ...
 [  0   0   0 ... 795   0   0]
 [  8   7   2 ...   0 418   0]
 [  0   0   0 ...   0   0 282]]
Classification report:
               precision    recall  f1-score   support

           0       0.75      0.60      0.67       795
           1       0.89      1.00      0.94       795
           2       0.98      0.84      0.90       796
           3       0.99      1.00      1.00       795
           4       0.93      0.94      0.93       795
           5       1.00      0.98      0.99       795
           6       0.99      0.96      0.97       795
           7       0.99      1.00      1.00       796
           8       0.99      0.99      0.99       795
           9       1.00      1.00      1.00       795
          10       1.00      0.96      0.98       795
          11       0.96      0.93      0.95       795
          12       0.97      0.93      0.95       796
          13       1.00      1.00      1.00       795
          14       1.00      1.00      1.00       795
          15       1.00      1.00      1.00       795
          16       0.99      1.00      0.99       795
          17       0.99      1.00      1.00       796
          18       0.98      0.99      0.98       795
          19       0.99      1.00      1.00       795
          20       0.99      1.00      1.00       795
          21       0.99      1.00      0.99       795
          22       0.98      0.98      0.98       796
          23       0.97      0.90      0.94       795
          24       0.83      0.49      0.62       795
          25       0.90      0.85      0.88       795
          26       1.00      1.00      1.00       795
          27       0.99      1.00      1.00       795
          28       0.99      0.98      0.99       795
          29       1.00      1.00      1.00       796
          30       0.95      1.00      0.97       795
          31       1.00      1.00      1.00       795
          32       1.00      1.00      1.00       795
          33       0.99      0.95      0.97       795
          34       0.98      1.00      0.99       795
          35       1.00      0.98      0.99       796
          36       1.00      1.00      1.00       795
          37       1.00      1.00      1.00       795
          38       1.00      1.00      1.00       795
          39       1.00      0.94      0.97       795
          40       0.98      0.80      0.88       796
          41       1.00      1.00      1.00       795
          42       0.75      0.91      0.82       795
          43       0.23      0.97      0.37       795
          44       1.00      1.00      1.00       795
          45       0.85      0.64      0.73       796
          46       0.98      1.00      0.99       796
          47       1.00      1.00      1.00       795
          48       1.00      1.00      1.00       795
          49       1.00      1.00      1.00       795
          50       1.00      1.00      1.00       795
          51       1.00      1.00      1.00       796
          52       1.00      1.00      1.00       795
          53       1.00      0.51      0.68       795
          54       1.00      1.00      1.00       795
          55       1.00      1.00      1.00       795
          56       0.98      0.54      0.69       795
          57       0.95      0.81      0.87       796
          58       1.00      1.00      1.00       795
          59       0.99      0.95      0.97       795
          60       1.00      1.00      1.00       795
          61       1.00      1.00      1.00       796
          62       1.00      1.00      1.00       795
          63       1.00      1.00      1.00       795
          64       1.00      1.00      1.00       795
          65       1.00      1.00      1.00       795
          66       1.00      1.00      1.00       796
          67       0.99      0.99      0.99       795
          68       1.00      1.00      1.00       795
          69       1.00      1.00      1.00       795
          70       0.92      1.00      0.96       795
          71       1.00      1.00      1.00       795
          72       0.95      0.53      0.68       795
          73       0.97      0.35      0.52       795

    accuracy                           0.93     58844
   macro avg       0.97      0.93      0.94     58844
weighted avg       0.97      0.93      0.94     58844


Learning from the approach
• The accuracy is coming out to be really good reaching a satisfactory 94%. Oversampling the data and making the distribution more uniform has resulted in much better results.
• Precision and Recall has also come very good.
• F1 Score is also very good.
• As anticipated the computational cost has been increased significantly as well with each epoch running for nearly 1 hour.

PERFORMANCE COMPARISON
image.png

FINAL SOLUTION
Accordingly, we decided to submit the LSTM model with up-sampled data to be our final model for submission as the model is not over-fitting and is providing high train and validation accuracy as 93.68 % and 93.92%, respectively.
As part of the preprocessing step we will be including the step to oversample out data using random over sampler and create more samples of the minority class.

image.png

LIMITATIONS
• The dataset is highly imbalanced considering our target variable which could affect the performance measurement criteria of our classification model.

• Handling of multiple language data and junk characters.

• The ‘langdetect’ and ’googletrans’ library works much better for longer strings where it can sample more n-grams. For short strings of a few words, it's not quite reliable.

• Language is detected more correctly for English and German but quite incorrectly for other languages.

• Adding caller to input feature was not adding significant improvement so we have assumed that the ‘Caller’ attribute does not include valuable information.

• Even though, the submitted model (oversampling approach) has high accuracy, it is taking too much time to train. The model can be upgraded to run faster, almost in real-time or near real-time or saving the model weights and using it through transfer learning.

CLOSING REFLECTIONS
The journey from beginning to the submission of the Capstone Project has been very educative. Being the real-world problem, it seemed very difficult at the beginning to solve the problem as we struggled a lot in achieving desirable accuracy of the model which can be put in real world. During the journey of achieving the milestones, we have learned many new NLP techniques which are actually used in real-world problem-solving processes.

As we came across some mandatory steps to be followed in Text Processing, we can directly start with the same in next project, which can save a lot of time. Further, we can work towards reducing latency in the present model.

For next time, we may start with unsupervised techniques like clustering and build clusters out of data and try to learn the ground truth by building models on top of it.

We may try different libraries such as BERT next time which we have not used at this time to achieve the milestone. Also, for dealing with multi language data, since google translate was not giving satisfactory results, a more custom based translation system could be needed. For oversampling, more methods could be explored or business tailored system could be built for data synthesis. We would explore more hyperparameters and how they could affect the performance.

Overall, the experience has been very intuitive and full of learning and gives a snippet of real-world idea.

                                                      THANK YOU
