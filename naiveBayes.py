# This Python script was written using Python 2.7.13

from __future__ import division
import math
import json
import datetime
import math

from decimal import *
# set precision of decimal numbers. 
# needs to be large to handle how small posterior probabilities get when making predictions
getcontext().prec = 400 

# Class to store and retrieve data about each email

"""
  Class to store and retrieve data about each email
  Args:
    emailId: {String} Id of email
    isSpam: {Boolean} is this email spam or not
    words: {Dict} dictionary of all words with key as word, count of that word in the email as its value
"""
class Email:
  def __init__(self, emailId, isSpam, words):
    self.emailId = emailId
    self.isSpam = isSpam
    self.words = words # dictionary word. key is word, value is count of word

  def getEmailId(self):
    return self.emailId

  def getIsSpam(self):
    return self.isSpam

  def getWords(self):
    return self.words

"""
  Function turn a line of data into an Email object
"""
def convertLineToEmail(line):
  data = [x.strip() for x in line.split(' ')]
  emailId = data[0]
  isSpam = True if data[1] == 'spam' else False
  words = {}
  for i in range(2, len(data), 2):
    words[data[i]] = int(data[i + 1])
  return Email(emailId, isSpam, words)

"""
  Function to build out a dictionary that represents our entire vocabulary
"""
def calcVocabulary(emailsList):
  vocabulary = {}
  for email in emailsList:
    for word in email.getWords().iteritems():
      wordKey = word[0]
      if wordKey not in vocabulary:
        vocabulary[wordKey] = True
  return vocabulary

"""
  Function to loop through a list of emails and count the number of occurences of each word along with the total number of words
"""
def calcWordLikelihoods(emailsList, vocabulary):
  totalWords = 0
  wordCounts = {}

  for email in emailsList:
    for word in email.getWords().iteritems():
      wordKey = word[0]
      wordCount = word[1]

      if wordKey not in wordCounts:
        wordCounts[wordKey] = 0
      
      wordCounts[wordKey] += wordCount
      totalWords += wordCount

  # after getting all counts of each word, calculate their probabilities
  sizeOfVocabulary = len(vocabulary.keys())
  likelihoods = {}
  for wordKey in vocabulary:
    countOfWord = 0

    if wordKey in wordCounts:
      countOfWord = wordCounts[wordKey]

    # without smoothing
    # likelihoods[wordKey] = (countOfWord) / (totalWords + sizeOfVocabulary)

    # with Laplace smoothing
    smoothingFactor = 1
    likelihoods[wordKey] = (countOfWord + smoothingFactor) / ((smoothingFactor * totalWords) + sizeOfVocabulary)

  return likelihoods

"""
  Function to make a prediction. Returns True if predicting spam, False if ham
"""
def makePrediction(email, P_spam, P_ham, Spam_word_likelihoods, Ham_word_likelihoods, vocabulary):
  ham_posterior_prob = Decimal(P_ham)
  spam_posterior_prob = Decimal(P_spam)

  for word in email.getWords().iteritems():
    wordKey = word[0]
    wordCount = word[1]
    if wordKey in vocabulary:
      for i in range(0, wordCount): # for each position of word multiply its probability
        ham_posterior_prob *= Decimal(Ham_word_likelihoods[wordKey])
        spam_posterior_prob *= Decimal(Spam_word_likelihoods[wordKey])
      
  if (ham_posterior_prob > spam_posterior_prob):
    return False
  else:
    return True

# lists of spam and ham emails
emails = {'spam': [], 'ham': []}

with open('data/train', 'r') as trainingDataFile:
  for line in trainingDataFile:
    email = convertLineToEmail(line)
    if email.getIsSpam():
      emails['spam'].append(email)
    else:
      emails['ham'].append(email)

# calc probabilities of each class
totalEmails = len(emails['spam']) + len(emails['ham'])
P_spam = len(emails['spam']) / totalEmails
P_ham = len(emails['ham']) / totalEmails

# calculate entire vocabulary
allEmails = emails['spam'] + emails['ham']
vocabulary = calcVocabulary(allEmails)

# calculate likelihoods of words in spam and ham emails
Spam_word_likelihoods = calcWordLikelihoods(emails['spam'], vocabulary)
Ham_word_likelihoods = calcWordLikelihoods(emails['ham'], vocabulary)

# open test data and make predictions
with open('data/test', 'r') as testDataFile:
  numPredictions = 0
  numCorrectPredictions = 0
  for line in testDataFile:
    email = convertLineToEmail(line)
    prediction = makePrediction(email, P_spam, P_ham, Spam_word_likelihoods, Ham_word_likelihoods, vocabulary)
    
    numPredictions += 1
    if prediction == email.getIsSpam():
      numCorrectPredictions += 1
    
  print str(numPredictions) + ' emails classified with ' + str(numCorrectPredictions * 100 / numPredictions) + '% accuracy'