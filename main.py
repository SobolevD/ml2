import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, normalize

from utils import processing_raw_data, get_confusion_matrix, print_confusion_matrix, get_tf, get_idf, calculate_accuracy

if __name__ == '__main__':
    # read data and set numeric marks to classes ham and spam
    full_data = pd.read_csv("resource/spam.csv", usecols=[0, 1], encoding='latin-1')

    label_encoder = LabelEncoder()
    full_data['v1'] = label_encoder.fit_transform(full_data['v1'])
    full_data['v2'] = full_data['v2'].apply(processing_raw_data)

    X_train, X_test, y_train, y_test = train_test_split(full_data['v2'].values, full_data['v1'].values,
                                                        test_size=0.33, random_state=42)

    # Bag of Words
    # Set vectorizer to remove stop words again and get word pairs as tokens
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), stop_words='english')

    # Learn vocabulary and idf, return document-term matrix. Returns Tf-idf-weighted document-term matrix.
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    elements_to_remove  = np.where(np.sum(X_train, axis = 1) == 0)
    X_train             = np.delete(X_train, elements_to_remove, axis = 0)
    y_train             = np.delete(y_train, elements_to_remove, axis = 0)

    # tf-idf
    idf     = get_idf(X_train)
    tf      = get_tf(X_train)
    X_train = normalize(np.multiply(tf, idf))

    tf = get_tf(X_test)
    X_test        = normalize(np.multiply(tf, idf))

    # naive Bayes
    classificator = MultinomialNB()
    classificator.fit(X_train, y_train)

    y_predicted   = classificator.predict(X_test)

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    conf_matrix = get_confusion_matrix(y_test.astype(bool), y_predicted.astype(bool))
    cf_matrix = confusion_matrix(y_test, y_predicted, labels=[0, 1])
    print_confusion_matrix(cf_matrix)

    print(f"\nConfusion matrix:\n{conf_matrix}", )
    print(f"\nAccuracy provided by manual realization: {calculate_accuracy(conf_matrix)}")
    print(f"sklearn accuracy: {accuracy_score(y_test, y_predicted)}")
