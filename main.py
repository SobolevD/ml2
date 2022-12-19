import re
import nltk
import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, normalize


# nltk.download('punkt')
# nltk.download('stopwords')


# Сложный (написать собственную реализацию TF-IDF + свои метрики, без использования готовых)
# 45. Анализ пар слов / TF-IDF / sklearn* / accuracy

#   Metrics and scoring: quantifying the quality of predictions provided by sklearn
# ‘accuracy’ metrics.accuracy_score
#
# ‘balanced_accuracy’ metrics.balanced_accuracy_score
#
# ‘top_k_accuracy’ metrics.top_k_accuracy_score
#
# ‘average_precision’ metrics.average_precision_score
#
# ‘neg_brier_score’ 	metrics.brier_score_loss
#
# ‘f1’ 	metrics.f1_score for binary targets
#
# ‘f1_micro’	 metrics.f1_score micro-averaged
#
# ‘f1_macro’ 	metrics.f1_score macro-averaged
#
# ‘f1_weighted’ 	metrics.f1_score weighted average
#
# ‘f1_samples’ 	metrics.f1_score by multilabel sample
#
# ‘neg_log_loss’ 	metrics.log_loss requires predict_proba support
#
# ‘precision’ etc. 	metrics.precision_score suffixes apply as with ‘f1’
#
# ‘recall’ etc. 	metrics.recall_score suffixes apply as with ‘f1’
#
# ‘jaccard’ etc. 	metrics.jaccard_score  suffixes apply as with ‘f1’
#
# ‘roc_auc’ 	metrics.roc_auc_score
#
# ‘roc_auc_ovr’	metrics.roc_auc_score
#
# ‘roc_auc_ovo’	metrics.roc_auc_score
#
# ‘roc_auc_ovr_weighted’	metrics.roc_auc_score
#
# ‘roc_auc_ovo_weighted’	metrics.roc_auc_score


def processing_raw_data(string: str):
    # Match a single character not present in the list below [^\w\s]
    # regular expression: \w  matches any word character (equivalent to [a-zA-Z0-9_])
    # \s  matches any whitespace character (equivalent to [\r\n\t\f\v ])
    # re.sub pattern - строка шаблона регулярного выражения, repl - строка замены, string - строка для поиска

    remove_useless_symbols = re.sub(r'[^\w\s]', ' ', string)

    tokens = word_tokenize(remove_useless_symbols)
    # cast to lowercase
    tokens = [token.lower() for token in tokens]
    # remove stopwords
    tokens = [token for token in tokens if (token not in stopwords.words('english'))]

    # stemming
    stemmatizer = nltk.stem.SnowballStemmer('english')
    tokens = [stemmatizer.stem(i) for i in tokens]
    return ' '.join(tokens)


def get_tf(X_test_after_transform):
    return X_test_after_transform


# idf(t, D) = log( |D| / di from D where t from di)
# число слов текста / число текстов из коллекции текстов D  в которых хоть раз было втречено t
def get_idf(token_entry_string_matrix):
    N = token_entry_string_matrix.shape[0]
    sub_result_matrix = np.copy(token_entry_string_matrix)
    sub_result_matrix[token_entry_string_matrix > 0] = 1
    document_frequency = np.sum(sub_result_matrix, axis=0)
    return np.log((N + 1) / (document_frequency + 1)) + 1


def get_confusion_matrix(y_true, y_pred):
    true_positive = np.sum(y_pred & y_true)
    false_positive = np.sum(y_pred[y_true == False])
    true_negative = np.sum(~y_pred & ~y_true)
    false_negative = np.sum(~y_pred[y_true == True])
    return np.array([[true_negative, false_positive], [false_negative, true_positive]])


def print_confusion_matrix(conf_matrix):
    class_names = ['ham', 'spam']
    upd_conf_matrix = pd.DataFrame(conf_matrix, index=[i for i in class_names], columns=[i for i in class_names])
    print(f"Confusion matrix:\n{upd_conf_matrix}")


def calculate_accuracy(conf_matrix):
    return (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix.flatten())


if __name__ == '__main__':

    # read data and set numeric marks to classes ham and spam
    full_data = pd.read_csv("resource/spam.csv", usecols=[0, 1], encoding='latin-1')

    # set numeric marks
    label_encoder = LabelEncoder()
    full_data['v1'] = label_encoder.fit_transform(full_data['v1'])

    # operations on data to prepare it for analysis
    full_data['v2'] = full_data['v2'].apply(processing_raw_data)

    print(full_data)

    # Split arrays or matrices into random train and test subsets.
    X_train, X_test, y_train, y_test = train_test_split(full_data['v2'].values, full_data['v1'].values,
                                                        test_size=0.33, random_state=42)

    #  getting (Bag of Words)
    # Set vectorizer to remove stop words again and get word pairs as tokens
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), stop_words='english')

    # Learn vocabulary and idf, return document-term matrix. Returns Tf-idf-weighted document-term matrix.
    # columns - tokens, rows - string from text train.  In cross: number of the token entry in the string
    X_train = vectorizer.fit_transform(X_train).toarray()

    # Learn vocabulary and idf from training set. Returns Tf-idf-weighted document-term matrix.
    # columns - tokens, rows - string from text train.  In cross: number of the token entry in the string
    X_test = vectorizer.transform(X_test).toarray()

    # Remove meaningless strings
    elements_to_remove = np.where(np.sum(X_train, axis=1) == 0)
    X_train = np.delete(X_train, elements_to_remove, axis=0)
    y_train = np.delete(y_train, elements_to_remove, axis=0)

    # find tf-idf for training dataset
    idf = get_idf(X_train)
    tf = get_tf(X_train)
    X_train = normalize(np.multiply(tf, idf))

    # find tf-idf for test dataset
    tf = get_tf(X_test)
    X_test = normalize(np.multiply(tf, idf))

    # Using naive Bayes classifier
    classificator = MultinomialNB()
    classificator.fit(X_train, y_train)

    # getting predictions
    y_predicted = classificator.predict(X_test)

    # Evaluate classfication quality with accuracy metrics
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    conf_matrix = get_confusion_matrix(y_test.astype(bool), y_predicted.astype(bool))
    cf_matrix = confusion_matrix(y_test, y_predicted, labels=[0, 1])
    print_confusion_matrix(cf_matrix)
    print("\nconfusion matrix:\n", conf_matrix)
    print("\nAccuracy provided by manual realization:", calculate_accuracy(conf_matrix))
    # from sklearn.metrics import accuracy_score
    print("Accuracy provided by sklearn:", accuracy_score(y_test, y_predicted))




