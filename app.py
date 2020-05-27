from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os 

import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import shutil

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def extract_text(fname):
    if fname[-3:] == "pdf":
    
        # Store all the pages of the PDF in a variable 
        pages = convert_from_path(fname, 500) 
        
        # Counter to store images of each page of PDF to image 
        image_counter = 1
        
        # Iterate through all the pages stored above 
        for page in pages: 
        
            # Declaring filename for each page of PDF as JPG 
            filename = "page_"+str(image_counter)+".jpg"
            
            # Save the image of the page in system 
            page.save(filename, 'JPEG') 
        
            # Increment the counter to update filename 
            image_counter = image_counter + 1
        
        # Variable to get count of total number of pages 
        filelimit = image_counter-1
        
        text = ""
        
        # Iterate from 1 to total number of pages 
        for i in range(1, filelimit + 1): 
        
            filename = "page_"+str(i)+".jpg"
                
            text = text + str(((pytesseract.image_to_string(Image.open(filename)))))
            return text
    else:
        im = Image.open(fname)
        text = pytesseract.image_to_string(im, lang = 'eng')
        return text

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

model = joblib.load('./model/model.pkl')

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/result", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        #print("Couldn't create upload directory: {}".format(target))
        pass

    upload = request.files['file']
    filename = upload.filename
    destination = "/".join([target, filename])
    upload.save(destination)
    
    fname = "static/" + filename

    res = extract_text(fname)
    
    # Class predictions from the model
    prediction = model.predict([res])
    if prediction == "spam":
        shutil.copy(fname, "spam/"+filename)
        return render_template("spam.html", f = fname)
    else:
        shutil.copy(fname, "inbox/"+filename)
        return render_template("ham.html", f = fname)

@app.route('/train', methods=['GET'])
def train():

    # Import Dataset
    messages = pd.read_csv('SMSSpamCollection', sep='\t',names=["label", "message"])

    # Add new column LENGTH to dataset
    messages['length'] = messages['message'].apply(len)

    bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
    messages_bow = bow_transformer.transform(messages['message'])

    # TD-IDF
    # TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
    # IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

    msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)
    pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    pipeline.fit(msg_train,label_train)

    # Saving the trained model on disk
    joblib.dump(pipeline, './model/model.pkl')

    # Return success message for user display on browser
    return 'Success'





if __name__ == "__main__":
    app.run(port=8080, debug=True)

