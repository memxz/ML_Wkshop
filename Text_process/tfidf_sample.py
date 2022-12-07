import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

'''
Download NLTK's resources if they are not present in local machine.
'''
def init_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


'''
Performs lemmatization, stopwords-removal and punctuation-removal.
'''
def preprocess(docs):
    clean_docs = []

    # get a lemmatizer object from NLTK
    lemma = WordNetLemmatizer()
    
    # get NLTK's list of stopwords
    stop_words = stopwords.words('english')

    # create a mapper that replaces punctuations (defined 
    # in string.punctuation) to an empty string 
    punc = str.maketrans('', '', string.punctuation)
    
    for doc in docs:
        # remove punctuation
        doc_no_punc = doc.translate(punc)
        # convert all characters to lowercase (normalization)
        words = doc_no_punc.lower().split()    

        # any word that is not found in NLTK's list of stopwords
        # is lemmatized to its root-form ('v' means 'verb')
        # and stored in the 'words' array
        words = [lemma.lemmatize(word, 'v')
                    for word in words if word not in stop_words]    

        # join each word in our list to form back a document
        clean_docs.append(' '.join(words))
    
    return clean_docs    


'''
Performs TF-IDF to extract features from our corpus.
'''
def TF_IDF(docs):
    tfidf = TfidfVectorizer()

    # toarray() transforms results in a sparse matrix form
    # to a dense matrix form
    feature_vectors = tfidf.fit_transform(docs).toarray()

    # returning both feature-vectors and feature-names. the 
    # feature-vectors are aligned with the feature-names (vocab)
    return feature_vectors, tfidf.get_feature_names_out()



'''
Display feature-vectors along with feature-names (aka - our vocab)
'''
def pretty_print(feat_vecs, feat_names):
    df = pd.DataFrame(data=feat_vecs,
            index=['doc1', 'doc2', 'doc3'],
            columns=feat_names)

    print(df)



'''
Main Program
'''

# make sure our ntlk resources are ready before script runs 
init_nltk_resources()

# we start off with a tiny corpus of 3 documents
docs = [
    'John has some cats.',
    'Cats, being cats, eat fish.',
    'I ate a big fish.'
]

# data cleansing
clean_docs = preprocess(docs)
print(clean_docs)

# getting feature-vectors and feature-names from 
# after our TF-IDF process
feat_vecs, feat_names = TF_IDF(clean_docs)

# display the count-frequencies against the
# vocabulary (features) and documents
pretty_print(feat_vecs, feat_names)

