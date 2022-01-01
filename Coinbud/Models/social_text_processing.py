import pickle
import re
import csv
import spacy
from spacymoji import Emoji
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer

stemmer = SnowballStemmer(language='english')
nlp = spacy.load("en")
emoji_pipeline = Emoji(nlp)
nlp.add_pipe(emoji_pipeline, first=True)


class CommonWordAdjustments(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def common_word_adjustments(self, text):
        
        text = text.lower()
        print (text)
        
        changes = {
            'rocket': 'moon',
            'bullish': 'moon',
            'bull': 'moon',   
        }
        
        for k, v in changes.items():
            text = text.replace(k, v)
        
        return text

    def transform(self, X, y=None):
        X = [self.common_word_adjustments(_) for _ in X]

        return X

class RemoveUsernames(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def remove_usernames(self, text):
        text = re.sub('@[^\s]+', ' USER ', text)
        text = re.sub('\\$[^\s]+', ' USER ', text)
        return text

    def transform(self, X, y=None):
        
        X = [self.remove_usernames(_) for _ in X]
        
        return X
    

class ConvertEmojis(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def convert_emoji(self, text):

        try:
            text_obj = nlp(text)
        except ValueError:
            return text

        converted_text = ""
        for token in text_obj:
            if token._.is_emoji:
                converted_text += f"{token._.emoji_desc} "
            else:
                converted_text += f"{str(token)} "

        return converted_text

    def transform(self, X, y=None):
        
        X = [self.convert_emoji(_) for _ in X]
        
        return X

class RemoveURLs(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_urls(self, text):
        cleaned_text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' URL ', text)
        
        return cleaned_text

    def transform(self, X, y=None):
        X = [self.remove_urls(_) for _ in X]

        return X

class RemoveStopwords(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_stopwords(self, text):
        cleaned_text = text.replace('\n', '')
        try:
            cleaned_text = nlp(cleaned_text)
        except:
            return text
        cleaned_text = [token.orth_.lower() for token in cleaned_text if not token.is_stop]
        return ' '.join(cleaned_text)

    def transform(self, X, y=None):
        X = [self.remove_stopwords(_) for _ in X]

        return X

class RemovePunctuation(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_punctuation(self, text):

        cleaned_text = text.replace('\n', '')
        try:
            cleaned_text = nlp(cleaned_text)
        except:
            return cleaned_text
        cleaned_text = [token.orth_.lower() for token in cleaned_text if not token.is_punct]
        
        return ' '.join(cleaned_text)

    def transform(self, X, y=None):
        X = [self.remove_punctuation(_) for _ in X]

        return X

class RemoveNumbers(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_numbers(self, text):
        
        if text:

            # Other numbers
            numbers = re.findall(r'\d+(?:,\d+)?', text)
            for number in numbers:
                text = text.replace(number, 'NUMBER ')

        return text

    def transform(self, X, y=None):

        X = [self.remove_numbers(_) for _ in X]
        
        return X

class Lemmatize(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def lemmatize(self, text):
        try:
            cleaned_text = nlp(text)
        except:
            return text
        return ' '.join([token.lemma_ for token in cleaned_text])

    def transform(self, X, y=None):

        X = [self.lemmatize(_) for _ in X]
        
        return X

class Stem(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def stem(self, text):
        if text:
            text_split = text.split()
            cleaned_text = []
            for token in text_split:
                cleaned_text.append(stemmer.stem(token))

            return ' '.join(cleaned_text)
        
        else:
            return text


    def transform(self, X, y=None):

        X = [self.stem(_) for _ in X] 

        return X

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
        
        X = [[i] for i in X]
        file = open('processed_text.csv', 'w+', newline ='')
  
        # writing the data into the file
        print(X[0:5])
        with file:
            write = csv.writer(file)
            write.writerow(['text'])
            write.writerows(X)
        
        return X

class TokenizeText(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Load tokenizer
        with open(f'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        X = tokenizer.transform(X)

        return X
