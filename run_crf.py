import os
import pandas as pd
import joblib

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

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crf_model.joblib')

# Load the trained CRF model
if not os.path.exists(MODEL_PATH):
    print(f"Trained model not found at {MODEL_PATH}. Please run train_crf.py first.")
    exit(1)
crf = joblib.load(MODEL_PATH)

# Get user input and tokenize it for prediction
user_input = input("Enter a sentence: ")
test_sent = [(w,) for w in user_input.split()]
X_test = [[word2features(test_sent, i) for i in range(len(test_sent))]]
y_pred = crf.predict(X_test)

# Display results in a table
# Each word and its predicted NER label
tokens = [w[0] for w in test_sent]
labels = y_pred[0]
df = pd.DataFrame({'Text': tokens, 'Predicted Label': labels})
print(df)
