#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import tqdm


# In[2]:


train = pd.read_excel(r'C:\Users\dell\Desktop\news_text-catagorization-with-deep-learning-techniques-main/Data_Train.xlsx')
test = pd.read_excel(r'C:\Users\dell\Desktop\news_text-catagorization-with-deep-learning-techniques-main/Data_Test.xlsx')
submission = pd.read_excel(r'C:\Users\dell\Desktop\news_text-catagorization-with-deep-learning-techniques-main/Sample_submission.xlsx')


# In[6]:


train.head()


# In[7]:


test.head()


# In[8]:


train.columns


# In[9]:


train['SECTION'].value_counts()


# #Build Train and Test Datasets

# In[10]:


# build train and test datasets

train_STORY = train['STORY'].values
train_SECTION = train['SECTION'].values

test_STORY = test['STORY'].values


# In[11]:


train_STORY


# In[12]:


train_SECTION


# In[13]:


test_STORY


# In[14]:


sub_section = submission['SECTION']


# ## Text Wrangling & Normalization

# In[12]:


##conda install pyahocorasick


# In[3]:


##!pip install contractions
import contractions


# In[4]:


import contractions
from bs4 import BeautifulSoup
import numpy as np
import re
import tqdm
import unicodedata


def strip_html_tags(text):
  soup = BeautifulSoup(text, "html.parser")
  [s.extract() for s in soup(['iframe', 'script'])]
  stripped_text = soup.get_text()
  stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
  return stripped_text

def remove_accented_chars(text):
  text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  return text

def pre_process_corpus(docs):
  norm_docs = []
  for doc in tqdm.tqdm(docs):
    doc = strip_html_tags(doc)
    doc = doc.translate(doc.maketrans("\n\t\r", "   "))
    doc = doc.lower()
    doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()  
    norm_docs.append(doc)
  
  return norm_docs


# In[15]:


get_ipython().run_cell_magic('time', '', '\nnorm_train_story = pre_process_corpus(train_STORY)\nnorm_test_story = pre_process_corpus(test_STORY)')


# 
# # Traditional Supervised Machine Learning Models
# ## feature Engineering

# In[16]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n\n# build BOW features on train reviews\ncv = CountVectorizer(binary=False, min_df=5, max_df=1.0, ngram_range=(1,2))\ncv_train_features = cv.fit_transform(norm_train_story)\n\n\n# build TFIDF features on train reviews\ntv = TfidfVectorizer(use_idf=True, min_df=5, max_df=1.0, ngram_range=(1,2),\n                     sublinear_tf=True)\ntv_train_features = tv.fit_transform(norm_train_story)')


# In[17]:


get_ipython().run_cell_magic('time', '', '\n# transform test reviews into features\ncv_test_features = cv.transform(norm_test_story)\ntv_test_features = tv.transform(norm_test_story)')


# In[18]:



print('BOW model:> Train features shape:', cv_train_features.shape, ' Test features shape:', cv_test_features.shape)
print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)


# 
# # Model Training, Prediction and Performance Evaluation
# ## Try out Logistic Regression
# The logistic regression model is actually a statistical model developed by statistician David Cox in 1958. It is also known as the logit or logistic model since it uses the logistic (popularly also known as sigmoid) mathematical function to estimate the parameter values. These are the coefficients of all our features such that the overall loss is minimized when predicting the outcome—

# In[18]:


get_ipython().run_cell_magic('time', '', "\n# Logistic Regression model on BOW features\nfrom sklearn.linear_model import LogisticRegression\n\n# instantiate model\nlr = LogisticRegression(penalty='l2', max_iter=500, C=1, solver='lbfgs', random_state=42)\n\n# train model\nlr.fit(cv_train_features, train_SECTION)\n\n# predict on test data\nlr_bow_predictions = lr.predict(cv_test_features)")


# In[19]:


lr_bow_predictions


# In[20]:


# download from colab


# In[26]:


df_lr = pd.DataFrame (lr_bow_predictions)
submission['SECTION'] = df_lr.values
filepath = r'C:\Users\dell\Desktop\news_text-catagorization-with-deep-learning-techniques-main\MH_news-text_catagorzation_LR.xlsx'
submission.to_excel(filepath, index= False)


# In[24]:


#from google.colab import files
#files.download('MH_news-text_catagorzation_LR.xlsx')


# # Newer Supervised Deep Learning Models

# In[29]:


import gensim
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Activation, Dense
from sklearn.preprocessing import LabelEncoder


# In[30]:


labels = ['Politics', 'Technology', 'Entertainment', 'Business']


# # Prediction class label encoding

# In[31]:


le = LabelEncoder()
# tokenize train reviews & encode train labels
tokenized_train = [nltk.word_tokenize(text)
                       for text in norm_train_story]
y_train = le.fit_transform(train_SECTION)
# tokenize test reviews & encode test labels
tokenized_test = [nltk.word_tokenize(text)
                       for text in norm_test_story]
y_test = le.fit_transform(sub_section)


# In[32]:



# print class label encoding map and encoded labels
print('section class label map:', dict(zip(le.classes_, le.transform(le.classes_))))
print('Sample test label transformation:\n'+'-'*35,
      '\nActual Labels:', sub_section, '\nEncoded Labels:', y_test[:3])


# In[33]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.test.utils import common_texts
from gensim.models import Word2Vec


# In[34]:



get_ipython().run_cell_magic('time', '', '# build word2vec model\nw2v_model = gensim.models.Word2Vec(tokenized_train, window=150,\n                                   min_count=10, workers=4)')


# In[35]:


def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)


# In[36]:


w2v_num_features = 100
# generate averaged word vector features from word2vec model
avg_wv_train_features = averaged_word2vec_vectorizer(corpus=tokenized_train, model=w2v_model,
                                                     num_features=w2v_num_features)
avg_wv_test_features = averaged_word2vec_vectorizer(corpus=tokenized_test, model=w2v_model,
                                                    num_features=w2v_num_features)


# In[37]:


print('Word2Vec model:> Train features shape:', avg_wv_train_features.shape, ' Test features shape:', avg_wv_test_features.shape)


# # Modeling with deep neural networks
# ## Building Deep neural network architecture

# In[39]:


from keras.layers import BatchNormalization


# In[40]:


def construct_deepnn_architecture(num_input_features):
    dnn_model = Sequential()
    dnn_model.add(Dense(512, input_shape=(num_input_features,), kernel_initializer='he_normal'))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Activation('elu'))
    dnn_model.add(Dropout(0.2))
    
    dnn_model.add(Dense(256, kernel_initializer='he_normal'))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Activation('elu'))
    dnn_model.add(Dropout(0.2))
    
    dnn_model.add(Dense(256, kernel_initializer='he_normal'))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Activation('elu'))
    dnn_model.add(Dropout(0.2))
    
    dnn_model.add(Dense(4))
    dnn_model.add(Activation('softmax'))

    dnn_model.compile(loss='categorical_crossentropy', optimizer='adam',                 
                      metrics=['accuracy'])
    return dnn_model


# In[41]:


w2v_dnn = construct_deepnn_architecture(num_input_features=w2v_num_features)


# In[42]:


w2v_dnn.summary()


# # Model Training, Prediction and Performance Evaluation

# In[43]:


import keras
import keras.utils
from keras import utils as np_utils
from keras.utils import to_categorical


# In[44]:


to_categorical(train['SECTION'])


# In[45]:


batch_size = 100
w2v_dnn.fit(avg_wv_train_features, to_categorical(train['SECTION']), epochs=50, batch_size=batch_size, 
            shuffle=True, validation_split=0.1, verbose=1)


# In[46]:


y_pred = w2v_dnn.predict_classes(avg_wv_test_features)
predictions = le.inverse_transform(y_pred)


# In[47]:


df_dnn = pd.DataFrame (predictions)
submission['SECTION'] = df_dnn.values
filepath = 'MH_news-text_catagorzation_DNN.xlsx'
submission.to_excel(filepath, index= False)


# # Implement LSTM
# 

# In[48]:


import tensorflow as tf

t = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
# fit the tokenizer on the documents
t.fit_on_texts(norm_train_story)
t.word_index['<PAD>'] = 0


# In[49]:


VOCAB_SIZE = len(t.word_index)


# In[50]:


train_sequences = t.texts_to_sequences(norm_train_story)
test_sequences = t.texts_to_sequences(norm_test_story)
X_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=1000)
X_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=1000)


# In[51]:


EMBEDDING_DIM = 300 # dimension for dense embeddings for each token
LSTM_DIM = 128 # total LSTM units

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=1000))
model.add(tf.keras.layers.SpatialDropout1D(0.1))
model.add(tf.keras.layers.LSTM(LSTM_DIM, return_sequences=False))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
model.summary()


# In[52]:


batch_size = 100
model.fit(X_train, to_categorical(train['SECTION']), epochs=10, batch_size=batch_size, 
          shuffle=True, validation_split=0.1, verbose=1)


# In[53]:


pred_lstm = model.predict_classes(X_test)
pred_lstm[:20]


# In[56]:


df_lstm = pd.DataFrame (pred_lstm)
submission['SECTION'] = df_lstm.values
filepath = r'C:\Users\dell\Desktop\news_text-catagorization-with-deep-learning-techniques-main\MH_news-text_catagorzation_lstm.xlsx'
submission.to_excel(filepath, index= False)


# In[ ]:




