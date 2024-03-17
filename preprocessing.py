import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import itertools
from string import ascii_lowercase
import pandas as pd
import pickle
import tensorflow as tf
from keras.preprocessing import sequence
from tensorflow import keras


#global variables defined
stopword_list = []
dual_alpha_list = []
train_text = []
lemma_train_text = []
processed_train_text = []

RE_PATTERNS = {
    ' american ':
        [
            'amerikan'
        ],

    ' adolf ':
        [
            'adolf'
        ],


    ' hitler ':
        [
            'hitler'
        ],

    ' fuck':
        [
            '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
            ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',
            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',
            'feck ', ' fux ', 'f\*\*', 
            'f\-ing', 'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck','fuk', 'wtf','fucck','f cking'
        ],

    ' ass ':
        [
            '[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\$\$'
                                                           '[^a-z]anus', ' a\*s\*s', '[^a-z]ass[^a-z ]',
            'a[@#\$%\^&\*][@#\$%\^&\*]', '[^a-z]anal ', 'a s s'
        ],

    ' asshole ':
        [
            ' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole', 'ass hole'
        ],

    ' bitch ':
        [
            'b[w]*i[t]*ch', 'b!tch',
            'bi\+ch', 'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',
            'biatch', 'bi\*\*h', 'bytch', 'b i t c h','beetch'
        ],

    ' bastard ':
        [
            'ba[s|z]+t[e|a]+rd'
        ],

    ' transgender':
        [
            'transgender','trans gender'
        ],

    ' gay ':
        [
            'gay'
        ],

    ' cock ':
        [
            '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
            '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'
        ],

    ' dick ':
        [
            ' dick[^aeiou]', 'deek', 'd i c k','diick '
        ],

    ' suck ':
        [
            'sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'
        ],

    ' cunt ':
        [
            'cunt', 'c u n t'
        ],

    ' bullshit ':
        [
            'bullsh\*t', 'bull\$hit','bs'
        ],

    ' homosexual':
        [
            'homo sexual','homosex'
        ],

    ' jerk ':
        [
            'jerk'
        ],

    ' idiot ':
        [
            'i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots', 'i d i o t'
        ],

    ' dumb ':
        [
            '(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'
        ],

    ' shit ':
        [
            'shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\$hit', 's h i t'
        ],

    ' shithole ':
        [
            'shythole','shit hole'
        ],

    ' retard ':
        [
            'returd', 'retad', 'retard', 'wiktard', 'wikitud'
        ],

    ' rape ':
        [
            ' raped'
        ],

    ' dumbass':
        [
            'dumb ass', 'dubass'
        ],

    ' asshead':
        [
            'butthead', 'ass head'
        ],

    ' sex ':
        [
            's3x', 'sexuality',
        ],


    ' nigger ':
        [
            'nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'
        ],

    ' shut the fuck up':
        [
            'stfu'
        ],

    ' pussy ':
        [
            'pussy[^c]', 'pusy', 'pussi[^l]', 'pusses'
        ],

    ' faggot ':
        [
            'faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
            '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot',
        ],

    ' motherfucker':
        [
            ' motha ', ' motha f', ' mother f', 'motherucker', 'mother fucker'
        ],

    ' whore ':
        [
            'wh\*\*\*', 'w h o r e'
        ],
}
potential_stopwords=['editor', 'reference', 'thank', 'work','find', 'good', 'know', 'like', 'look', 'thing', 'want', 
                     'time', 'list', 'section','wikipedia', 'doe', 'add','new', 'try', 'think', 'write','use', 'user', 'way', 'page']

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
def clean_text(text, remove_repeat_text=True, remove_patterns_text=True, is_lower=True):

    if is_lower:
        text=text.lower()
        
    if remove_patterns_text:
        for target, patterns in RE_PATTERNS.items():
            for pat in patterns:
                text=str(text).replace(pat, target)

    if remove_repeat_text:
        text = re.sub(r'(.)\1{2,}', r'\1', text) 

    text = str(text).replace("\n", " ")
    text = re.sub(r'[^\w\s]',' ',text)
    text = re.sub('[0-9]',"",text)
    text = re.sub(" +", " ", text)
    text = re.sub("([^\x00-\x7F])+"," ",text)
    return text 
    
def lemma(text, lemmatization=True):
    lemmatizer = WordNetLemmatizer()
    output=''
    if lemmatization:
        text=text.split(' ')
        for word in text:
            word1 = lemmatizer.lemmatize(word, pos = "n") #noun 
            word2 = lemmatizer.lemmatize(word1, pos = "v") #verb
            word3 = lemmatizer.lemmatize(word2, pos = "a") #adjective
            word4 = lemmatizer.lemmatize(word3, pos = "r") #adverb
            output=output + " " + word4
    else:
        output=text
    
    return str(output.strip())

def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)
    
def dual_alpha():
    for s in iter_all_strings():
        dual_alpha_list.append(s)
        if s == 'zz':
            break

def alter_dual_alpha():
    dual_alpha_list.remove('i')
    dual_alpha_list.remove('a')
    dual_alpha_list.remove('am')
    dual_alpha_list.remove('an')
    dual_alpha_list.remove('as')
    dual_alpha_list.remove('at')
    dual_alpha_list.remove('be')
    dual_alpha_list.remove('by')
    dual_alpha_list.remove('do')
    dual_alpha_list.remove('go')
    dual_alpha_list.remove('he')
    dual_alpha_list.remove('hi')
    dual_alpha_list.remove('if')
    dual_alpha_list.remove('is')
    dual_alpha_list.remove('in')
    dual_alpha_list.remove('me')
    dual_alpha_list.remove('my')
    dual_alpha_list.remove('no')
    dual_alpha_list.remove('of')
    dual_alpha_list.remove('on')
    dual_alpha_list.remove('or')
    dual_alpha_list.remove('ok')
    dual_alpha_list.remove('so')
    dual_alpha_list.remove('to')
    dual_alpha_list.remove('up')
    dual_alpha_list.remove('us')
    dual_alpha_list.remove('we')

    for letter in dual_alpha_list:
        stopword_list.append(letter)
    
def alter_stopwords():
    for word in potential_stopwords:
        stopword_list.append(word)
    print(len(stopword_list))

def remove_stopwords(text, remove_stop=True):
    output = ""
    if remove_stop:
        text=text.split(" ")
        for word in text:
            if word not in stopword_list:
                output=output + " " + word
    else :
        output=text

    return str(output.strip())


def modelLoad_Tokenize(text):
# Load tokenizer and model
    tokenizer = 'C:/Users/Hetvi/Desktop/OneDrive/Projects/Kaggle competition - jigsaw\data_transformation/tokenizer'
    with open(tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = tf.keras.models.load_model('C:/Users/Hetvi/Desktop/OneDrive/Projects/Kaggle competition - jigsaw/model_trainer')

    # Tokenize and pad the text
    text = tokenizer.texts_to_sequences([text])
    text = keras.preprocessing.sequence.pad_sequences(text, maxlen=200, padding='post')

    # Make prediction
    prediction = model.predict(text)
    return prediction


def dataTransform(text):
     
    text = clean_text(text)

    
    text = lemma(text)

   
    dual_alpha()

   
    alter_dual_alpha()

    
    alter_stopwords()

   
    text = remove_stopwords(text)

    prediction = modelLoad_Tokenize(text)

    return prediction

result = dataTransform("You are an asshole")
print(result)