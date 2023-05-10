# NaiveBayesPy

The above code is a Python script that implements a Naive Bayes algorithm for text classification.
It is designed to work with the 20 Newsgroups dataset, which consists of a collection of 20,000 documents that have been partitioned into 20 different newsgroups.
The script uses a training set of 1000 documents per newsgroup to learn the probability distribution of each word in each class, and then uses this information
to classify a set of test documents.

The code reads in the training set of documents and preprocesses them by removing newlines and spaces, removing stopwords, lemmatizing the words,
and converting everything to lowercase. It then creates a word count dictionary for each class and calculates the probability of each word occurring in that class.
These probabilities are stored in a list of dictionaries, one for each class.

Once the training set has been processed, the script reads in a test set of documents and applies the Naive Bayes algorithm to classify them.
For each document, it calculates the probability of that document belonging to each class by multiplying the probabilities of the individual words in the document.
The script then assigns the document to the class with the highest probability.

The script uses the scikit-learn library to calculate the accuracy of the classification.
It also includes some additional stopwords and removes all words with less than four characters in order to improve the accuracy of the algorithm.
