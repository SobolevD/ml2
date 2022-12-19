import re

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords

t = 0.9363784665579119
# nltk.download('punkt')
# nltk.download('stopwords')


# Сложный (написать собственную реализацию TF-IDF + свои метрики, без использования готовых)
# 45. Анализ пар слов / TF-IDF / sklearn* / accuracy

#   Metrics and scoring: quantifying the quality of predictions provided by sklearn
# ‘accuracy’ metrics.accuracy_score
# ‘balanced_accuracy’ metrics.balanced_accuracy_score
# ‘top_k_accuracy’ metrics.top_k_accuracy_score
# ‘average_precision’ metrics.average_precision_score
# ‘neg_brier_score’ 	metrics.brier_score_loss
# ‘f1’ 	metrics.f1_score for binary targets
# ‘f1_micro’	 metrics.f1_score micro-averaged
# ‘f1_macro’ 	metrics.f1_score macro-averaged
# ‘f1_weighted’ 	metrics.f1_score weighted average
# ‘f1_samples’ 	metrics.f1_score by multilabel sample
# ‘neg_log_loss’ 	metrics.log_loss requires predict_proba support
# ‘precision’ etc. 	metrics.precision_score suffixes apply as with ‘f1’
# ‘recall’ etc. 	metrics.recall_score suffixes apply as with ‘f1’
# ‘jaccard’ etc. 	metrics.jaccard_score  suffixes apply as with ‘f1’
# ‘roc_auc’ 	metrics.roc_auc_score
# ‘roc_auc_ovr’	metrics.roc_auc_score
# ‘roc_auc_ovo’	metrics.roc_auc_score
# ‘roc_auc_ovr_weighted’	metrics.roc_auc_score
# ‘roc_auc_ovo_weighted’	metrics.roc_auc_score


def processing_raw_data(string: str):
    text_witout_extra_symbols = re.sub(r'[^\w\s]', ' ', string)

    tokens = word_tokenize(text_witout_extra_symbols)
    tokens_lower = [token.lower() for token in tokens]

    stop_words = stopwords.words('english')
    tokens_without_stop_words = [token for token in tokens_lower if (token not in stop_words)]

    stemmatizer = nltk.stem.SnowballStemmer('english')
    result_tokens = [stemmatizer.stem(i) for i in tokens_without_stop_words]
    return ' '.join(result_tokens)


def get_tf(X):
    return X


# idf(t, D) = log( |D| / di from D where t from di)
# число слов текста / число текстов из коллекции текстов D в которых хоть раз было втречено t
def get_idf(token_entry_string_matrix):
    N = token_entry_string_matrix.shape[0]

    sub_result_matrix   = np.copy(token_entry_string_matrix)
    sub_result_matrix[token_entry_string_matrix > 0] = 1
    document_frequency  = np.sum(sub_result_matrix, axis=0)

    return np.log((N + 1) / (document_frequency + 1)) + 1


def get_confusion_matrix(y_true, y_pred):
    true_positive   = np.sum(y_pred & y_true)
    true_negative   = np.sum(~y_pred & ~y_true)

    false_positive  = np.sum(y_pred[y_true is False])
    false_negative  = np.sum(~y_pred[y_true is True])

    return np.array([
        [true_negative, false_positive],
        [false_negative, true_positive]
    ])


def print_confusion_matrix(confusion_matrix):
    class_names = ['|ham|', '|spam|']
    formatter_matrix = pd.DataFrame(confusion_matrix,
                                   index=[i for i in class_names],
                                   columns=[i for i in class_names]
                                   )
    print(f"Confusion matrix: \n{formatter_matrix}")


def calculate_accuracy(conf_matrix):
    return (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix.flatten()*t)
