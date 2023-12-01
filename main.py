import streamlit as st
import subprocess
import warnings

from PIL import Image

st.set_page_config(layout='wide', page_title='Fake News Detection | Innovation_Redefined ', page_icon=':newspaper:', menu_items={'About': '#Fake News Detection\nThis project was created for Minor Project.'})

# load the model and stuff
@st.cache_resource
def load():
  # useful
  import os

  # mathematics and operations
  import math
  import numpy as np

  # capture data
  import pickle
  import requests
  import urllib.request
  import io
  import zipfile

  # html
  from bs4 import BeautifulSoup as bs
  
  # download and unzip resources
  basepath = '.'
  if not os.path.exists(os.path.join(basepath, 'train_val_data.pkl')):
    urllib.request.urlretrieve(
      'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Fake%20News%20Detection/inspirit_fake_news_resources%20(1).zip', 'data.zip'
    )
    
    import zipfile
    with zipfile.ZipFile('data.zip') as zipper:
       zipper.extractall()
    
  with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
    train_data, val_data = pickle.load(f)
  
  warnings.warn('Data loaded.')
  
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import precision_recall_fscore_support
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import confusion_matrix

  # natural language and vocab
  import nltk
  nltk.download('words')
  from nltk.corpus import words
  vocab = words.words()

  y_train = [label for url, html, label in train_data]
  y_val = [label for url, html, label in val_data]

  # prepare data
  def prepare_data(data, featurizer, is_train):
    X = []
    for index, datapoint in enumerate(data):
      url, html, label = datapoint
      html = html.lower()

      features = featurizer(url, html)

      # Gets the keys of the dictionary as descriptions, gets the values as the numerical features.
      feature_descriptions, feature_values = zip(*features.items())

      X.append(feature_values)

    return X, feature_descriptions

  # train model
  def train_model(X_train):
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    return model

  # wrapper function for everything above
  def instantiate_model(compiled_featurizer):
    X_train, feature_descriptions = prepare_data(train_data, compiled_featurizer, True)
    X_val, feature_descriptions = prepare_data(val_data, compiled_featurizer, False)

    model = train_model(X_train)

    return model, X_train, X_val, feature_descriptions

  # a wrapper function that takes in named a list of keyword argument functions
  # each of those functions are given the URL and HTML, and expected to return a list or dictionary with the appropriate features
  def create_featurizer(**featurizers):
    def featurizer(url, html):
      features = {}

      for group_name, featurizer in featurizers.items():
        group_features = featurizer(url, html)

        if type(group_features) == type([]):
          for feature_name, feature_value in zip(range(len(group_features)), group_features):
            features[group_name + ' [' + feature_name + ']'] = feature_value
        elif type(group_features) == type({}):
          for feature_name, feature_value in group_features.items():
            features[group_name + ' [' + feature_name + ']'] = feature_value
        else:
          features[group_name] = feature_value

      return features
    return featurizer

  # evaluate model
  def evaluate_model(model, X_val):
    y_val_pred = model.predict(X_val)

    print_metrics(y_val, y_val_pred)
    confusion_matrix(y_val, y_val_pred)

    return y_val_pred

  # confusion matrices
  from sklearn import metrics
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  def confusion_matrix(y_val, y_val_pred):
    # Create the Confusion Matrix
    cnf_matrix = metrics.confusion_matrix(y_val, y_val_pred)

    # Visualizing the Confusion Matrix
    class_names = [0,1] # Our diagnosis categories

    fig, ax = plt.subplots()
    # Setting up and visualizing the plot (do not worry about the code below!)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g') # Creating heatmap
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion matrix', y = 1.1)
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')

  # other metrics
  def print_metrics(y_val, y_val_pred):
    prf = precision_recall_fscore_support(y_val, y_val_pred)
    return {'Accuracy': accuracy_score(y_val, y_val_pred), 'Precision': prf[0][1], 'Recall': prf[1][1], 'F-1 Score': prf[2][1]}

  # gets the log count of a phrase/keyword in HTML (transforming the phrase/keyword to lowercase).
  def get_normalized_keyword_count(html, keyword):
    # only concern words inside the body, to speed things up
    try:
      necessary_html = html.split('<body')[1].split('</body>')[0]
    except:
      necessary_html = html # if it doesn't have a body...

    return math.log(1 + necessary_html.count(keyword.lower())) # log is a good normalizer

  # count the number of words in a URL
  def count_words_in_url(url):
    for i in range(len(url), 2, -1): # don't count the first letter, because sometimes that might be a word by itself (like why bother counting 'l' a word?)
      if url[:i].lower() in vocab: # if it's a word
        return 1 + count_words_in_url(url[i:]) # get more words, and keep counting
    return 0 # no words in URL (or at least, it doesn't start with a word, such as NYTimes)

  def url_extension_featurizer(url, html):
    features = {}

    features['.com'] = url.endswith('.com')
    features['.org'] = url.endswith('.org')
    features['.edu'] = url.endswith('.edu')
    features['.net'] = url.endswith('.net')
    features['.co'] = url.endswith('.co')
    features['.nz'] = url.endswith('.nz')
    features['.media'] = url.endswith('.media')
    features['.za'] = url.endswith('.za')
    features['.fr'] = url.endswith('.fr')
    features['.is'] = url.endswith('.is')
    features['.tv'] = url.endswith('.tv')
    features['.press'] = url.endswith('.press')
    features['.news'] = url.endswith('.news')
    features['.uk'] = url.endswith('.uk')
    features['.info'] = url.endswith('.info')
    features['.ca'] = url.endswith('.ca')
    features['.agency'] = url.endswith('.agency')
    features['.us'] = url.endswith('.us')
    features['.ru'] = url.endswith('.ru')
    features['.su'] = url.endswith('.su')
    features['.biz'] = url.endswith('.biz')
    features['.ir'] = url.endswith('.ir')

    return features

  def keyword_featurizer(url, html):
    features = {}

    keywords = ['vertical', 'news', 'section', 'light', 'data', 'eq', 'medium', 'large', 'ad', 'header', 'text', 'js', 'nav', 'analytics', 'article', 'menu', 'tv', 'cnn', 'button', 'icon', 'edition', 'span', 'item', 'label', 'link', 'world', 'politics', 'president', 'donald', 'business', 'food', 'tech', 'style', 'amp', 'vr', 'watch', 'search', 'list', 'media', 'wrapper', 'div', 'zn', 'card', 'var', 'prod', 'true', 'window', 'new', 'color', 'width', 'container', 'mobile', 'fixed', 'flex', 'aria', 'tablet', 'desktop', 'type', 'size', 'tracking', 'heading', 'logo', 'svg', 'path', 'fill', 'content', 'ul', 'li', 'shop', 'home', 'static', 'wrap', 'main', 'img', 'celebrity', 'lazy', 'image', 'high', 'noscript', 'inner', 'margin', 'headline', 'child', 'interest', 'john', 'movies', 'music', 'parents', 'real', 'warren', 'opens', 'share', 'people', 'max', 'min', 'state', 'event', 'story', 'click', 'time', 'trump', 'elizabeth', 'year', 'visit', 'post', 'public', 'module', 'latest', 'star', 'skip', 'imagesvc', 'posted', 'ltc', 'summer', 'square', 'solid', 'default', 'super', 'house', 'pride', 'week', 'america', 'man', 'day', 'wp', 'york', 'id', 'gallery', 'inside', 'calls', 'big', 'daughter', 'photo', 'joe', 'deal', 'app', 'special', 'source', 'red', 'table', 'money', 'family', 'featured', 'makes', 'pete', 'michael', 'video', 'case', 'says', 'popup', 'carousel', 'category', 'script', 'helvetica', 'feature', 'dark', 'extra', 'small', 'horizontal', 'bg', 'hierarchical', 'paginated', 'siblings', 'grid', 'active', 'demand', 'background', 'height', 'cn', 'cd', 'src', 'cnnnext', 'dam', 'report', 'trade', 'images', 'file', 'huawei', 'mueller', 'impeachment', 'retirement', 'tealium', 'col', 'immigration', 'china', 'flag', 'track', 'tariffs', 'sanders', 'staff', 'fn', 'srcset', 'green', 'orient', 'iran', 'morning', 'jun', 'debate', 'ocasio', 'cortez', 'voters', 'pelosi', 'barr', 'buttigieg', 'american', 'object', 'javascript', 'uppercase', 'omtr', 'chris', 'dn', 'hfs', 'rachel', 'maddow', 'lh', 'teasepicture', 'db', 'xl', 'articletitlesection', 'founders', 'mono', 'ttu', 'biden', 'boston', 'bold', 'anglerfish', 'jeffrey', 'radius']
    for keyword in keywords:
      features[keyword] = get_normalized_keyword_count(html, keyword)

    return features

  def url_word_count_featurizer(url, html):
    return count_words_in_url(url.split('.')[-2])
    # for example, www.google.com will return google and nytimes.com will return nytimes

  compiled_featurizer = create_featurizer(
    url_extension=url_extension_featurizer,
    keyword=keyword_featurizer,
    url_word_count=url_word_count_featurizer,
    html_length=lambda url, html: len(html),
    url_length=lambda url, html: len(url))
  
  warnings.warn('Beginning to train model.')
  model, X_train, X_val, feature_descriptions = instantiate_model(compiled_featurizer)
  warnings.warn('Trained model.')
    
  return model, feature_descriptions, compiled_featurizer, requests, confusion_matrix, print_metrics, train_data, val_data

model, feature_descriptions, compiled_featurizer, requests, confusion_matrix, print_metrics, train_data, val_data = load()

# columns
left, right = st.columns(2)

# on the left, do the overview
left.title('Fake News Detection')
left.header('Innovation Redefined')
left.subheader('Lets Detect Fake News!')
left.write('**Instructor:** Mr. Dhrittiman Mukherjee')
left.write('**Group Members:** Harshavardhan Bajoria, Aftab Alam, Arghdeep Banerjee, Esita Budhia')

left.divider()

best_matrix = Image.open('best_matrix.png')

left.image(best_matrix, caption='Confusion matrix.')
left.subheader('Here are some metrics for the most accurate model we trained!')
left.write('**Maximum achieved accuracy:** $\\frac{299}{309}\\approx96.8\%$')
left.write('**Maximum achieved precision:** $100\%$')
left.write('**Maximum achieved recall:** $\\frac{131}{141}\\approx92.9\%$')
left.write('**Maximum achieved F-1 score:** $\\frac{131}{136}\\approx96.3\%$')

left.divider()

# on the right side, allow users to submit a URL
right.header('Try it out!')
right.write('*(Note that we do not use the bag-of-words or GloVe features in this model, in order to speed up deployment and save memory. See our [Google Colab](https://colab.research.google.com/drive/1NutMv5iJ2DAbU_YHPRonSrurvHQ2Al9v?usp=sharing) if you would like to view the entirety of the code).*')

with right.form(key='try_it_out'):
  raw_url = st.text_input(label='Enter a news article or site URL to predict validity', key='url')
  st.write('*Make sure your URL is valid.*')

  if st.form_submit_button(label='Submit', type='primary'):
    try:
      if '://' not in raw_url:
        raw_url = 'http://' + raw_url
      
      response = requests.get(raw_url)
      html = response.text.lower()
      url = raw_url.split('/')[2]

      features = compiled_featurizer(url, html)
      warnings.warn(str(features))
      _, feature_values = zip(*features.items())

      prediction = model.predict([feature_values])[0]
      
      st.write('*We predict that your news is ' + ('FAKE' if prediction else 'REAL') + ' news!*')      
      st.divider()
      
      products = [(coef_ * feature_values[index], coef_) for index, coef_ in enumerate(model.coef_[0].tolist())]
      items = sorted(zip(feature_descriptions, products), key=lambda item: (1 if prediction else -1) * item[1][0], reverse=True)
      
      with st.expander('See why.'):
        st.write('The top three features that allowed us to make this decision were: **' + items[0][0] + '**, **' + items[1][0] + '**, and **' + items[2][0] + '**!')
      
      with st.expander('See model parameters.'):
        st.write('The intercept (the value when all features are 0) is: **' + str(model.intercept_[0]) + '**.')
        st.divider()
        st.write('Here are all the feature weights that contributed to the decision.')
        st.write('\n\n'.join(map(lambda feature: f'The feature `{feature[0]}` has a weight of **{feature[1][1]}**. Multiplied by its value gives **{feature[1][0]}**.', items)))
    except:
      advice = st.write('*I don\'t think your URL worked. Please check your spelling or try another.*')

with st.expander('See an overview.'):
  """
  *  Basics
      *  For each news article or site, we extract **features** from the **URL** and **HTML**, such as, a vector describing the domain extension of the URL.
      *  We feed this into a logistic regression model, and out comes the result.
  *  Data
      *  The data was taken from news sites around the world, containing `2002` examples in the training data, and `309` in the testing data.
      *  `52.2%` of the training data is fake, while `54.4%` of the testing data is fake.
      *  For testing, we are only provided the URL of a site and its HTML content.
  *  Features
      *  At no point do we actually determine if the content is legitimate. We only ever employ statistical methods to determine if the site appears fake or not. In other words, if a particularly rude person looked at our code, they could easily write an article that is superficially real, to our model. Luckily, you, dear reader, would never do such a thing.
      *  Domain extension.
          * The domain extension is what comes after the domain name.
            *  The domain extension for `https://www.inspiritai.com/frequently-asked-questions` is `.com`.
            *  The domain extension for `https://en.wikipedia.org` is `.org`.
          *  There are a total of 22 different domain extensions in our training data: `.com`, `.org`, `.edu`, `.net`, `.co`, `.nz`, `.media`, `.za`, `.fr`, `.is`, `.tv`, `.press`, `.news`, `.uk`, `.info`, `.ca`, `.agency`, `.us`, `.ru`, `.su`, `.biz`, and `.ir`
            *  Note that 3 domain extensions: `.il`, `.me`, and `.au` only appeared in the testing data and not the training data, so when they appear, we have nothing to base our judgement on, and we simply count them as incorrect when later testing with just these features.
            *  Also (for fun), we can do some analysis on the domain extensions. The domain extensions for every government (`.au` for Australia, `.ca` for Canada, `.fr` for France, `.il` for Israel, `.ir` for Iran, `.is` for Iceland, `.nz` for New Zealand, `.ru` for Russia, `.su` for the Soviet Union, `.us` for the United States, and `.za` for South Africa) except for `.uk` for the United Kingdom, have not a single real article. This might be some error, or maybe not.
          *  We create individual features for the most common extensions—`1` if the URL ends in an extension and `0` otherwise. This is similar to a one-hot encoding.
          *  Alone, this accurately predicts the validity of the testing data `53.1%` of the time.
      *  Keywords.
          *  This feature is similar to a bag of words, where we count the number of certain keywords in the HTML body text.
          *  The keywords were determined by analyzing the training data. Specifically, we counted the words in real and fake news, weighted by if it was real or fake.
            * For example, if the word `apple` (this can't be controversial, right?) appears `10000` times total in some number of real sites, but only `100` times total in some number of fake sites—where we have the same number of real and fake sites, for simplicity—then our total count is `-9900`. This is very negative, so we add that as a keyword.
          *  We then normalized the count with the function 
    $\\log(1+\\#)$.
          * Issues.
            *  Of course, this is limited by the fact that we can not count word pairs or n-grams in general.
            *  Furthermore, fake news is generally around twice as long as real news, so that is another issue.
          * For example, the letter `u` is strongly correlated with real news, while `president` is strongly correlated with fake news.
          * By itself, this had an accuracy of `89.6\%`.
      *  URL word count.
         *  Some sites, such as `yahoo.com` or `cnn.com` have few (or no) words in them.
         *  Some sites, such as `redflagnews.com` or `consortiumtimes.com` have several.
         *  We hypothesize that real sites have fewer words while fake sites have less.
         *  We count the number of words in the following manner.
            *  Start from the second letter. Keep going to the next letter until you find that all the letters up to that letter form a valid word (according to nltk vocabulary).
            *  Once you find a valid word, increment the number of words by `1`, and start again, this time two letters after the end of the word. Repeat.
      *  Bag of Words.
         *  We count the number of each of a set of the `300` most common words in the description of the website, and create `300` features.
      *  GloVe.
         *  We use the average GloVe word embeddings of the description of the website as a list of several features.
      *  Lengths (of URL and HTML).
         *  We calculated the lengths of the URL and HTML.
  """

with st.expander('See our data.'):
  st.write('**Training data (only first 20 rows).**')
  st.table({'URL': [datapoint[0] for datapoint in train_data[:20]], 'HTML (for 100 characters)': [datapoint[1][:100] for datapoint in train_data[:20]], 'Label': ['Fake' if datapoint[2] else 'Real' for datapoint in train_data[:20]]})
  st.write('**Training data (only first 10 rows).**')
  st.table({'URL': [datapoint[0] for datapoint in val_data[:10]], 'HTML (for 100 characters)': [datapoint[1][:100] for datapoint in val_data[:10]], 'Label': ['Fake' if datapoint[2] else 'Real' for datapoint in val_data[:10]]})