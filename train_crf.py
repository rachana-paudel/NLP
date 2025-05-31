import os
import pandas as pd
from sklearn_crfsuite import CRF
import joblib

def load_conll2003(path):
    sentences = []
    sentence = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            if line.startswith('-DOCSTART-'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                word, label = parts[0], parts[-1]
                sentence.append((word, label))
    if sentence:
        sentences.append(sentence)
    return sentences

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word': word,
        'is_title': word.istitle(),
        'is_upper': word.isupper(),
        'is_digit': word.isdigit(),
        'prefix': word[:1],
        'suffix': word[-1:],
    }
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

# Path to your CoNLL-2003 train file (update if needed)
conll_path = os.path.join(os.path.dirname(__file__), 'eng.train')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crf_model.joblib')

if os.path.exists(conll_path):
    train_sents = load_conll2003(conll_path)
else:
    print('CoNLL-2003 train file not found.')
    exit(1)

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

crf = CRF(algorithm='lbfgs', max_iterations=100)
crf.fit(X_train, y_train)
joblib.dump(crf, MODEL_PATH)
print(f"Model trained and saved to {MODEL_PATH}")
