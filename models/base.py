import string
import html
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import pickle


class BaseModel:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop = stopwords.words('english')

        self.model = None
        self.vec = None

    # Load Vec
    def load_vec(self, vec_path, mode='rb'):
        with open(vec_path, mode) as pkl_file:
            self.vec = pickle.load(pkl_file)

    # Load Model
    def load_model(self, model_path, mode='rb'):
        with open(model_path, mode) as pkl_file:
            self.model = pickle.load(pkl_file)

    # Preprocessing
    def preprocessing(self, line: str) -> str:
        line = html.unescape(str(line))
        line = str(line).replace("can't", "cann't")
        line = word_tokenize(line.lower())

        tokens = []
        negated = False
        for t in line:
            if t in ['not', "n't", 'no']:
                negated = not negated
            elif t in string.punctuation or not t.isalpha():
                negated = False
            else:
                tokens.append('not_' + t if negated else t)

        tokens = [self.lemmatizer.lemmatize(t, 'v') for t in tokens if t not in self.stop]

        return ' '.join(tokens)

    # Predict
    def predict(self, line):
        if self.model is None or self.vec is None:
            print('Modle / Vec is not loaded')
            return ""

        line = self.preprocessing(line)
        features = self.vec.transform([line])

        return self.model.predict(features)[0]
