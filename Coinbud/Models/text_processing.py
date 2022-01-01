import pickle
import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer

stemmer = SnowballStemmer(language='english')



class CommonWordAdjustments(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def common_word_adjustments(self, title):
        title = title.lower()

        return title

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['cleaned_title'] = X_['title'].apply(self.common_word_adjustments)

        return X_

class RemoveURLs(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_urls(self, title):
        cleaned_title = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' URL ', title)
        cleaned_title = cleaned_title.lower()
        return cleaned_title

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['cleaned_title'] = X_['cleaned_title'].apply(self.remove_urls)

        return X_

class RemoveStopwords(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_stopwords(self, title):
        title = title.replace('\n', '')
        try:
            text = nlp(title)
        except:
            return title
        cleaned_title = [token.orth_.lower() for token in text if not token.is_stop]
        return ' '.join(cleaned_title)

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['cleaned_title'] = X_['cleaned_title'].apply(self.remove_stopwords)

        return X_

class RemovePunctuation(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_punctuation(self, title):
        title = title.replace('\n', '')
        try:
            text = nlp(title)
        except:
            return title
        cleaned_title = [token.orth_.lower() for token in text if not token.is_punct]
        return ' '.join(cleaned_title)

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['cleaned_title'] = X_['cleaned_title'].apply(self.remove_punctuation)

        return X_

class RemoveNumbers(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_numbers(self, title):

        # Other numbers
        numbers = re.findall(r'\d+(?:,\d+)?', title)
        for number in numbers:
            title = title.replace(number, 'NUMBER ')

        return title

    def transform(self, X, y=None):

        X_ = X.copy()
        X_['cleaned_title'] = X_['cleaned_title'].apply(self.remove_numbers)

        return X_

class Lemmatize(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def lemmatize(self, title):
        try:
            text = nlp(title)
        except:
            return title
        return ' '.join([token.lemma_ for token in text])

    def transform(self, X, y=None):

        X_ = X.copy()
        X_['cleaned_title'] = X_['cleaned_title'].apply(self.lemmatize)

        return X_

class Stem(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def stem(self, title):
        text = title.split()
        cleaned_title = []
        for token in text:
            cleaned_title.append(stemmer.stem(token))

        return ' '.join(cleaned_title)


    def transform(self, X, y=None):

        X_ = X.copy()
        X_['cleaned_title'] = X_['cleaned_title'].apply(self.stem)

        return X_

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.feature_names]

class Export(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.to_csv('processed_text.csv')
        return X

class TokenizeText(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Copy the df
        X_ = X.copy()

        # Load tokenizer
        with open(f'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        X_ = tokenizer.transform(X_)

        return X_
