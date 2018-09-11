import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss

def print_gridSearch_report(clf):
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in sorted(zip(means, stds, clf.cv_results_['params']), key=lambda x: x[0], reverse=True):
        stopKey = [key for key in params.keys() if 'stop_words' in key.lower()]
        if len(stopKey)>0 and params[stopKey[0]] != None:
            params[stopKey[0]] = True
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


def load_imdb(path):
    # Need cp1252 - Windows Western Europe encoding here
    oneHot = OneHotEncoder()
    df = pd.read_csv(path, index_col=0, encoding='cp1252')
    print("Loading IMDB dataset...")
    print("Original dataframe shape: ",df.shape)
    print("Original DF columns:", df.columns)
    print("Original Train/Test split:", df['type'].value_counts())
    print("Dropping the file column...")
    df = df.drop('file', axis=1)
    print("Dropping the unlabeled rows and splitting into train/test...")
    x_train = df.loc[(df['type'] == 'train') & (df['label'] != 'unsup'), 'review']
    y_train = df.loc[(df['type'] == 'train') & (df['label'] != 'unsup'), 'label']

    ##NO NEED
    # One Hot Encoding the target variable after label encoding
    # y_train_enc = LabelEncoder().fit_transform(y_train).reshape(-1, 1)
    # y_train_enc = OneHotEncoder().fit_transform(y_train_enc)
    
    print("X_train, y_train, shapes:", x_train.shape, y_train.shape)
    print("y_train counts:", y_train.value_counts())
    x_test = df.loc[(df['type'] == 'test') & (df['label'] != 'unsup'), 'review']
    y_test = df.loc[(df['type'] == 'test') & (df['label'] != 'unsup'), 'label']
    print("X_test, y_test, shapes:", x_test.shape, y_test.shape)
    print("y_test counts:", y_test.value_counts())

## NO NEED
    # One Hot Encoding the target variable after label encoding
    # y_test_enc = LabelEncoder().fit_transform(y_test).reshape(-1, 1)
    # y_test_enc = OneHotEncoder().fit_transform(y_test_enc)

    return x_train, y_train, x_test, y_test

# Here we build a train function that will automatically split the train set into train and validation
# so that our test set is separate
def train_test_score(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)
    classifier.fit(X_train, y_train)
    print("Accuracy: %s" % classifier.score(X_test, y_test))
    print("Log Loss: " % log_loss(y_pred=classifier.predict(X_test), y_true=y_test))
    return classifier