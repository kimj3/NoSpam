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

#Tokenize the strings using split()
preprocessed_documents = []
for sentence in sans_punctuation_documents:
  preprocessed_documents.append(sentence.split(" "))

#Count frequencies using counter method
frequency_list = []
import pprint
from collections import Counter 

for word in prepocessed_documents:
  frequency_counts = Counter(i) 
  frequency_list.append(frequency_counts)
pprint.pprint(frequency_list)

"""
Implementing Bag of Words in scikit-learn!
"""
documents = ['Hello, how are you!',
              'Win money, win from home.',
              'Call me now',
              'Hello, Call hello you tomorrow?']

from sklearn.feature_extraction.text import CountVectorizer 
count_vector = CountVectorizer(stop_words='english') #lowercase parameter default = True , removes all punctuations

count_vector.fit(documents)
count_vector.get_feature_names(stop_words="english") #returns feature names for this dataset, which make up the vocab for documents
doc_array = count_vector.transform(documents).toarray()
doc_array

#Convert the array into a dataframe and set the column names to the word names
frequency_matrix = pd.DataFrame(doc_array, 
                                columns = count_vector.get_feature_names())


#____________________________________________________________________________________ Bag of Words problem completed!

"""
Training and testing set

                      *Note*
1) x_train = training data for the "sms_message" column
2) y_train = training data for the 'label' column
3) x_test = testing data for the 'sms_message' column
4) y_test = testing data for the 'label' column 
"""

#Split the dataset into a training and testing set in order to test the model later

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)





"""Bag of words and convert the data into the desired matrix format using CountVectorizer()
-First, fir the training data(x_train) into CountVectorizer() and return the matrix
-Second, transform the testing data(x_test) to return the matrix
x_test is the testing data for the 'sms_message' column and will be used to make predictions on by comparing the predictions with y_test later
"""

#Instantitate the CountVectorizer method
count_vector = CountVectorizer()

#Fit the training data and return the matrix
training_data = count_vector.fit_transform(x_train)

#Transform testing ata and return the matrix. 

