"""
Using the Naive Bayes algorithm to create a model that can classify dataset SMS messages as spam or not spam, based on the training provided to the model. 

"""

#Used minoconda to download pandas and NumPy
import pandas as pd 

#Using a dataset from the UCI Machine Learning repository
#Shows how the data look like
df = pd.read_table('smsspamcollection/SMSSpamCollection', 
                    sep ='\t', 
                    header=None, 
                    names = ['label', 'sms_message'])

df.head()


#Convert labels to binary variables: 0=> "ham", 1=>"spam"
#This is important because Scikit-learn only takes numerical values!
df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
df.head() #returns the sizes in (rows, columns); (5572,2)

"""
Bag of words: Convert the set of text to a frequency distribution matrix
-ML algorithms rely on numerical data to be fed into them as input, and email/sms messages are text heavy!
-Therefore, BoW will take a piece of text and count the frequency of the words in that text, treating each word individually and the order in which the words occur does not matter
-This entire process can be done by using sklearns count vectorizer method!!
  -It tokenizes the string(separates the string into individual words) and gives an integer ID to each token (and converts all tokenized words to their lower case form, and ignores punctuations)
  -It counts the occurance of each of those tokens
  -Ignores stop words; if "english" is provided as a parameter, CountVectorizer will automatically ignore all words
"""

"""
Implementing Bag of Words from scratch:
"""

#Convert to lower case form
documents = ['Hello, how are you!',
              'Win money, win from home.',
              'Call me now',
              'Hello, Call hello you tomorrow?']
lower_case_documents = []

for sentence in documents:
  lower_case_documents.append(sentence.lower())

#Remove all punctuations
sans_punctuation_documents = []
import string 

for sentence in lower_case_documents:
  sans_punctuation_documents.append(sentence.translate(str.maketrans('','',string.punctuation)))

