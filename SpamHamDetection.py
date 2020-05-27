# Imports the libraries
import nltk
import pandas as pd

# Print length of dataset
messages = [line.rstrip() for line in open('SMSSpamCollection')]
print(len(messages))

# Print sample messages
for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')
    
# Import Dataset
messages = pd.read_csv('SMSSpamCollection', sep='\t',names=["label", "message"])

# EXPLORATORY DATA ANALYSIS
# Print details of dataset
print(messages.head())   
print(messages.describe())
print(messages.groupby('label').describe())

# Add new column LENGTH to dataset
messages['length'] = messages['message'].apply(len)
print(messages.head())


# DATA VISUALIZATION

import matplotlib.pyplot as plt
import seaborn as sns

messages['length'].plot(bins=50, kind='hist') 
messages.length.describe()
messages[messages['length'] == 910]['message'].iloc[0]
messages.hist(column='length', by='label', bins=50,figsize=(12,4))

# TEXT PREPROCESSING

import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)

# Show some stop words
stopwords.words('english')[0:10] 

nopunc.split()

# Now just remove any stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
print(clean_mess)

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Check to make sure its working
messages.head()    
messages['message'].head(5).apply(text_process)

# VECTORIZATION

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))

# Take one text message and get its bag-of-words counts as a vector
message4 = messages['message'][3]
print(message4)
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)

print(bow_transformer.get_feature_names()[4073])
print(bow_transformer.get_feature_names()[9570])

messages_bow = bow_transformer.transform(messages['message'])

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))

# TD-IDF
# TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# TRAINING THE MODEL

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])


# MODEL EVALUATION

all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)

from sklearn.metrics import classification_report, accuracy_score
print (classification_report(messages['label'], all_predictions))
print (accuracy_score(messages['label'], all_predictions))
# Train-Test-Split

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# Creating a PIPELINE

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)

# Print test set report
print(classification_report(predictions,label_test))
print(accuracy_score(predictions, label_test))

# Take a sample file
import sys
from tkinter import *
from tkinter.filedialog import askopenfilename   

fname = "unassigned"

def openFile():
    global fname
    fname = askopenfilename()
    root.destroy()

# =============================================================================
# if __name__ == '__main__':
# 
#     root = Tk()
#     Button(root, text='File Open', command = openFile).pack(fill=X)
#     mainloop()
# 
#     print (fname)
# =============================================================================


# EXTRACT the Contents of the IMAGE/PDF
# =============================================================================
#     
# from PIL import Image
# import pytesseract
# from pdf2image import convert_from_path 
#     
# if fname[-3:] == "pdf":
#     
#     # Store all the pages of the PDF in a variable 
#     pages = convert_from_path(fname, 500) 
#       
#     # Counter to store images of each page of PDF to image 
#     image_counter = 1
#       
#     # Iterate through all the pages stored above 
#     for page in pages: 
#       
#         # Declaring filename for each page of PDF as JPG 
#         filename = "page_"+str(image_counter)+".jpg"
#           
#         # Save the image of the page in system 
#         page.save(filename, 'JPEG') 
#       
#         # Increment the counter to update filename 
#         image_counter = image_counter + 1
#     
#     # Variable to get count of total number of pages 
#     filelimit = image_counter-1
#       
#     text = ""
#       
#     # Iterate from 1 to total number of pages 
#     for i in range(1, filelimit + 1): 
#       
#         filename = "page_"+str(i)+".jpg"
#              
#         text = text + str(((pytesseract.image_to_string(Image.open(filename)))))
#         #print(text)
# else:
#     im = Image.open(fname)
#     text = pytesseract.image_to_string(im, lang = 'eng')
#     #print(text)
#     
# #Image preview
# im.show()
# 
# # Predict the result
# print(pipeline.predict([text]))    
# =============================================================================

