
import re
import random
import math
import numpy as np
import pandas as pd

random.seed(10)
"""
Read text data from file and pre-process text by doing the following
1. convert to lowercase
2. convert tabs to spaces
3. remove "non-word" characters
Store resulting "words" into an array
"""
FILENAME='SMSSpamCollection'
all_data = open(FILENAME).readlines()

# split into train and test
num_samples = len(all_data)
all_idx = list(range(num_samples))
random.shuffle(all_idx)
idx_limit = int(0.8*num_samples)
train_idx = all_idx[:idx_limit]
test_idx = all_idx[idx_limit:]
train_examples = [all_data[ii] for ii in train_idx]
test_examples = [all_data[ii] for ii in test_idx]


# Preprocess train and test examples
train_words = []
train_labels = []
test_words = []
test_labels = []
prediction_train = []
prediction_test = []

# train examples
for line in train_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige returne
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0
    line_words = line_words[1:]
    train_words.append(line_words)
    train_labels.append(label)
    
# test examples
for line in test_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige return
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0

    line_words = line_words[1:]
    test_words.append(line_words)
    test_labels.append(label)

prob_spam_train = train_labels.count(1)/len(train_labels)
prob_ham_train = train_labels.count(0)/len(train_labels)
prob_spam_test = test_labels.count(1)/len(test_labels)
prob_ham_test = test_labels.count(0)/len(test_labels)

spam_words = []
ham_words = []
alpha = 0.1
for ii in range(len(train_words)):  # we pass through words in each (train) SMS
    words = train_words[ii]
    label = train_labels[ii]
    if label == 1:
        spam_words += words
    else:
        ham_words += words
input_words = spam_words + ham_words  # all words in the input vocabulary

# Count spam and ham occurances for each word
spam_counts = {}; ham_counts = {}
# Spamcounts
for word in spam_words:
    try:
        word_spam_count = spam_counts.get(word)
        spam_counts[word] = word_spam_count + 1
    except:
        spam_counts[word] = 1 + alpha  # smoothening

for word in ham_words:
    try:
        word_ham_count = ham_counts.get(word)
        ham_counts[word] = word_ham_count + 1
    except:
        ham_counts[word] = 1 + alpha  # smoothening

num_spam = len(spam_words)
num_ham = len(ham_words)


p_spam_test = []
p_ham_test = []
prediction = []
for line in test_words:
    prob_word_given_spam = 1
    prob_word_given_ham = 1
    for word in line:
        try:
            prob_word_given_spam *= (spam_counts[word]/(num_spam + alpha * 20000))
        except:
            prob_word_given_spam *= (alpha/(num_spam + alpha * 20000))
        try:
            prob_word_given_ham *= (ham_counts[word]/(num_ham + alpha * 20000))
        except:
            prob_word_given_ham *= (alpha/(num_ham + alpha * 20000))
    
    prob_word_given_spam *= prob_spam_train
    prob_word_given_ham *= prob_ham_train
    p_spam_test.append(prob_word_given_spam)
    p_ham_test.append(prob_word_given_ham)
    if (prob_word_given_spam > prob_word_given_ham):
        prediction.append(1)
    else:
        prediction.append(0)

data = {'test_Predicted': prediction,
        'test_Actual': test_labels
        }

df = pd.DataFrame(data, columns=['test_Actual','test_Predicted'])

confusion_matrix = pd.crosstab(df['test_Actual'], df['test_Predicted'], rownames=['Actual'], colnames=['Predicted'])

true_positive = confusion_matrix[1][1]
false_positive = confusion_matrix[0][1]
true_negative = confusion_matrix[0][0]
false_negative = confusion_matrix[1][0]

accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative )
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f_score = 2 * ((precision * recall) / (precision + recall))

print("\nTesting accuracy: " + str(accuracy))
print("\nConfusion matrix:\n")
print (confusion_matrix)
print("\nPrecision: " + str(precision))
print("\nrecall: " + str(recall))
print("\nF-score: " + str(f_score))


