import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Functions copied from train_crf.py ---
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

# --- Load test data ---
test_path = os.path.join(os.path.dirname(__file__), 'eng.train')
if not os.path.exists(test_path):
    print(f"Test data not found at {test_path}")
    exit(1)
test_sents = load_conll2003(test_path)
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# Load trained CRF model
model_path = os.path.join(os.path.dirname(__file__), 'crf_model.joblib')
if not os.path.exists(model_path):
    print(f"Trained model not found at {model_path}")
    exit(1)
crf = joblib.load(model_path)

# Predict labels
y_pred = crf.predict(X_test)

# Get all unique labels
labels = list(crf.classes_)


# Flatten the lists for confusion matrix
flat_y_test = [label for sent in y_test for label in sent]
flat_y_pred = [label for sent in y_pred for label in sent]

# Compute confusion matrix
cm = confusion_matrix(flat_y_test, flat_y_pred, labels=labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CRF Model Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("Confusion matrix image saved as confusion_matrix.png")
