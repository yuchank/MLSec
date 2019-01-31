import string
import email
import nltk

punctuations = list(string.punctuation)

print(punctuations)

stopword = set(nltk.corpus.stopwords.words('english'))

print(stopword)