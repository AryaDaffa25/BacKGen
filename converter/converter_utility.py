import spacy
import re
import string

# Loading modules to the pipeline. 
nlp = spacy.load("en_core_web_sm") 

# Define sentencizer function
def spacy_sentencizer(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# Define tokenizer function
def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc]

# Define function to get token SpaCy tag
def token_spacy_tag(token,spacy_doc,text):
    compiled_token = re.compile(token,re.IGNORECASE | re.VERBOSE)
    loc_indexes = [m.start(0) for m in re.finditer(compiled_token, text)]
    tag = [i.tag_ for i in spacy_doc if i.idx in loc_indexes ]
    return tag[0]

# Data preprocessing function
punc_cleaning = str.maketrans(string.punctuation, ' '*len(string.punctuation))
def clean_preprocessing(text):
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*', '', text)
    text = text.translate(punc_cleaning)
    text = ' '.join(text.split())
    text = ' '.join(spacy_tokenizer(text))
    return text