import helpers

# First import naive bayes and create a pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def optimize_countvect_nb(x_train, y_train, x_val, y_val, n_jobs=3):
    countPipe = Pipeline([
        ('countVect', CountVectorizer()),
        ('clf', MultinomialNB()),
    ])

    print("CounteVectorizer with Naive Bayes and Lemmatization option")

    params = {'countVect__binary': [False, True],
              'countVect__tokenizer': [None, LemmaTokenizer()],
              'countVect__ngram_range': [(1, 1), (1, 3), (1, 5), (3, 5), (5, 5)],
              'countVect__stop_words': [None, 'english'],
              'countVect__max_df': [0.2, 0.4, 0.6,],
              }

    enc = LabelEncoder()

    cv = GridSearchCV(countPipe, param_grid=params, n_jobs=n_jobs, scoring='accuracy', refit=True)
    cv.fit(x_train, y_train)

    helpers.print_gridSearch_report(cv)

    bestCv = cv.best_estimator_

    ## best_estimator_ returns fitted estimator when refit=True option
    # bestCv.fit(x_val, y_val)

    print("CountVect + NB: %s" % bestCv.score(x_val, y_val))
    lloss = log_loss(y_pred=enc.fit_transform(bestCv.predict(x_val)), y_true=enc.fit_transform(y_val))
    print("CountVect + NB Log Loss:",  lloss)



def optimize_tfidfvect_nb(x_train, y_train, x_val, y_val, n_jobs=3):

    print("TFIDF Vectorizer with Naive Bayes and Lemmatization option")

    tfidfPipe = Pipeline([
        ('tfidfVect', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])

    params = {'tfidfVect__binary': [False, True],
              'tfidfVect__tokenizer': [None, LemmaTokenizer()],
              'tfidfVect__ngram_range': [(1, 1), (1, 3), (1, 5), (3, 3), (5,5)],
              'tfidfVect__stop_words': [None, 'english'],
              'tfidfVect__max_df': [0.2, 0.4, 0.6,],
              }
    enc = LabelEncoder()

    cv = GridSearchCV(tfidfPipe, param_grid=params, n_jobs=n_jobs, scoring='accuracy')
    cv.fit(x_train, y_train)

    helpers.print_gridSearch_report(cv)

    bestCv = cv.best_estimator_

    # Best Estimator should already been fit on the training data
    # bestCv.fit(x_traintr, y_traintr)

    print("TfidfVect + NB: %s" % bestCv.score(x_val, y_val))
    lloss = log_loss(y_pred=enc.fit_transform(bestCv.predict(x_val)), y_true=enc.fit_transform(y_val))
    print("TfidfVect + NB Log Loss:",  lloss)


def optimize_countvect_log(x_train, y_train, x_val, y_val, n_jobs=3):

    print("CountVectorizer with Logistic Regression and Lemmatization option")

    countPipe = Pipeline([
        ('countVect', CountVectorizer()),
        ('clf', LogisticRegression()),
    ])

    params = {'countVect__binary': [False, True],
              'countVect__tokenizer': [None, LemmaTokenizer()],
              'countVect__ngram_range': [(1, 3), (1, 5), (3, 5)],
              'countVect__stop_words': [None, 'english'],
              'countVect__max_df': [0.1, 0.2, 0.3, 0.4, 0.6],
              'clf__C': [0.1, 1, 10, 100, 1000],
              }

    enc = LabelEncoder()

    cv = GridSearchCV(countPipe, param_grid=params, n_jobs=n_jobs, scoring='accuracy', refit=True)
    cv.fit(x_train, y_train)

    helpers.print_gridSearch_report(cv)
    bestCv = cv.best_estimator_

    # bestCv.fit(x_traintr, y_traintr)

    print("CountVect + Logistic Accuracy: %s" % bestCv.score(x_val, y_val))
    lloss = log_loss(y_pred=enc.fit_transform(bestCv.predict(x_val)), y_true=enc.fit_transform(y_val))
    print("CountVect + Logistic Log Loss:",  lloss)



def optimize_tfidfvect_log(x_train, y_train, x_val, y_val):
    print("TFIDFVectorizer with Logistic Regression and Lemmatization option")

    tfidfPipe = Pipeline([
        ('tfidfVect', TfidfVectorizer()),
        ('clf', LogisticRegression()),
    ])

    params = {'tfidfVect__binary': [False, True],
              'tfidfVect__tokenizer': [None, LemmaTokenizer()],
              'tfidfVect__ngram_range': [(1, 3), (1, 5), (3, 5), (5,5)],
              'tfidfVect__stop_words': [None, 'english'],
              'tfidfVect__max_df': [0.2, 0.4, 0.6],
              'clf__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              }

    enc = LabelEncoder()

    cv = GridSearchCV(tfidfPipe, param_grid=params, n_jobs=n_jobs, scoring='accuracy', refit=True)
    cv.fit(x_train, y_train)

    helpers.print_gridSearch_report(cv)
    bestCv = cv.best_estimator_

    # bestCv.fit(x_traintr, y_traintr)
    print("TfidfVect + Logistic Accuracy: %s" % bestCv.score(x_val, y_val))
    lloss = log_loss(y_pred=enc.fit_transform(bestCv.predict(x_val)), y_true=enc.fit_transform(y_val))
    print("TfidfVect + Logistic Log Loss:", lloss)



if __name__ == "__main__":
    path = 'data/imdb_master.csv'
    random_state = 60
    train_val_ratio = 0.2

    n_jobs = 3

    # Load the original dataset
    x_train, y_train, x_test, y_test = helpers.load_imdb(path)

    # Split the training set further into a train and validation set
    x_traintr, x_val, y_traintr, y_val = train_test_split(x_train, y_train, test_size=train_val_ratio,
                                                          random_state=random_state)

    # optimize_countvect_nb(x_traintr, y_traintr, x_val, y_val, n_jobs=n_jobs)
    # optimize_tfidfvect_nb(x_traintr, y_traintr, x_val, y_val, n_jobs=n_jobs)
    # optimize_tfidfvect_log(x_traintr, y_traintr, x_val, y_val, n_jobs=n_jobs)
    optimize_countvect_log(x_traintr, y_traintr, x_val, y_val, n_jobs=n_jobs)