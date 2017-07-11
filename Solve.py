
# coding: utf-8

# In[7]:

import pandas as pd
import numpy as np
import re
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from easymoney.money import EasyPeasy
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_colwidth',100)


# In[8]:

#load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# ## Converting the Currency to Standard USD Currency 

# In[9]:

#Functions

def convert_currency(data):
    ep=EasyPeasy()
    tmp=[]
    for i in range(len(data)):
        tmp.append(ep.currency_converter(amount=data['goal'][i], from_currency=data['currency'][i], to_currency="USD", pretty_print=False))
    return tmp



# In[10]:

train['goal']=convert_currency(train)
test['goal']=convert_currency(test)


# In[11]:

# convert unix time format
unix_cols = ['deadline','state_changed_at','launched_at','created_at']

for x in unix_cols:
    train[x] = train[x].apply(lambda k: datetime.datetime.fromtimestamp(int(k)).strftime('%Y-%m-%d %H:%M:%S'))
    test[x] = test[x].apply(lambda k: datetime.datetime.fromtimestamp(int(k)).strftime('%Y-%m-%d %H:%M:%S'))


# # Feature Extraction
# Extracting the features like the length and the count of the name, description and keywords

# In[12]:

cols_to_use = ['name','desc']
len_feats = ['name_len','desc_len']
count_feats = ['name_count','desc_count']

for i in np.arange(2):
    train[len_feats[i]] = train[cols_to_use[i]].apply(str).apply(len)
    test[len_feats[i]] = test[cols_to_use[i]].apply(str).apply(len)


# In[13]:

train['name_count'] = train['name'].str.split().str.len()
train['desc_count'] = train['desc'].str.split().str.len()

test['name_count'] = test['name'].str.split().str.len()
test['desc_count'] = test['desc'].str.split().str.len()


# In[14]:

train['keywords_len'] = train['keywords'].str.len()
train['keywords_count'] = train['keywords'].str.split('-').str.len()

test['keywords_len'] = test['keywords'].str.len()
test['keywords_count'] = test['keywords'].str.split('-').str.len()


# ## Finding the difference in seconds between the Deadline and launched_at also between launche_at and created_at

# In[15]:

# converting string variables to datetime
unix_cols = ['deadline','state_changed_at','launched_at','created_at']

for x in unix_cols:
    train[x] = train[x].apply(lambda k: datetime.datetime.strptime(k, '%Y-%m-%d %H:%M:%S'))
    test[x] = test[x].apply(lambda k: datetime.datetime.strptime(k, '%Y-%m-%d %H:%M:%S'))


# In[16]:

time1 = []
time3 = []
for i in np.arange(train.shape[0]):
    time1.append(np.round((train.loc[i, 'launched_at'] - train.loc[i, 'created_at']).total_seconds()).astype(int))
    time3.append(np.round((train.loc[i, 'deadline'] - train.loc[i, 'launched_at']).total_seconds()).astype(int))


# In[17]:

#finding Logarithm of the time to avoid outliers
train['time1'] = np.log(time1)
train['time3'] = np.log(time3)


# In[18]:

# for test data
time5 = []
time6 = []
for i in np.arange(test.shape[0]):
    time5.append(np.round((test.loc[i, 'launched_at'] - test.loc[i, 'created_at']).total_seconds()).astype(int))
    time6.append(np.round((test.loc[i, 'deadline'] - test.loc[i, 'launched_at']).total_seconds()).astype(int))


# In[19]:

test['time1'] = np.log(time5)
test['time3'] = np.log(time6)


# In[20]:

#Label Encoding is used to encode the values into digits for the same category
feat = ['disable_communication','country']

for x in feat:
    le = LabelEncoder()
    le.fit(list(train[x].values) + list(test[x].values))
    train[x] = le.transform(list(train[x]))
    test[x] = le.transform(list(test[x]))


# In[21]:

#log1p returns the logarithm of the goal value with a addition of one to avoid the inliers
train['goal'] = np.log1p(train['goal'])
test['goal'] = np.log1p(test['goal'])


# # Cleaning

# In[22]:

desc_ = pd.Series(train['desc'].tolist() + test['desc'].tolist()).astype(str)


# In[23]:

# this function cleans punctuations, digits and irregular tabs. Then converts the sentences to lower
def desc_clean(word):
    p1 = re.sub(pattern='(\W+)|(\d+)|(\s+)',repl=' ',string=word)
    p1 = p1.lower()
    return p1

desc_ = desc_.map(desc_clean)


# In[24]:

stop = set(stopwords.words('english'))
desc_ = [[x for x in x.split() if x not in stop] for x in desc_]

stemmer = SnowballStemmer(language='english')
desc_ = [[stemmer.stem(x) for x in x] for x in desc_]

desc_ = [[x for x in x if len(x) > 2] for x in desc_]

desc_ = [' '.join(x) for x in desc_]


# # Creating Count Features

# In[25]:

cv = CountVectorizer(max_features=650)


# In[26]:

alldesc = cv.fit_transform(desc_).todense()


# In[27]:

#create a data frame
combine = pd.DataFrame(alldesc)
combine.rename(columns= lambda x: 'variable_'+ str(x), inplace=True)


# In[28]:

#split the text features

train_text = combine[:train.shape[0]]
test_text = combine[train.shape[0]:]

test_text.reset_index(drop=True,inplace=True)


# In[29]:

project_id=test['project_id']


# ### Finalizing train and test data before merging

# In[30]:

cols_to_use = ['name_len','desc_len','keywords_len','name_count','desc_count','keywords_count','time1','time3','goal','country','disable_communication']


# In[31]:

target = train['final_status']


# In[32]:

train = train.loc[:,cols_to_use]
test = test.loc[:,cols_to_use]


# In[33]:

X_train = pd.concat([train, train_text],axis=1)
X_test = pd.concat([test, test_text],axis=1)


# In[34]:

train.head()


# ### Model Training

# In[35]:

dtrain = xgb.DMatrix(data=X_train, label = target)
dtest = xgb.DMatrix(data=X_test)


# In[36]:

params = {
    'objective':'binary:logistic',
    'eval_metric':'error',
    'eta':0.025,
    'max_depth':6,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':5
    
}


# In[37]:

bst_train = xgb.train(params, dtrain, num_boost_round=1500)


# In[38]:

p_test = bst_train.predict(dtest)


# In[39]:

sub = pd.DataFrame()
sub['project_id'] = project_id
sub['final_status'] = p_test


# ### The predicted value is the probability of Target value to be 1 hence if the probability is more than 0.5 then Mark it as 1 i.e Funding is Successful else mark it 0 i.e Funding is not successful

# In[40]:

sub['final_status'] = [1 if x > 0.5 else 0 for x in sub['final_status']]


# In[41]:

sub.to_csv("Result.csv",index=False)

