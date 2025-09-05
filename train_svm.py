import os
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score

# --- Data loading and feature extraction functions ---
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
    # The following line extracts the word from the sentence for feature creation
    word = sent[i][0]
    features = {
        'word': word,  # <-- This is where the 'word' feature is defined
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

# --- Load and flatten data ---
conll_path = os.path.join(os.path.dirname(__file__), 'eng.train')
if not os.path.exists(conll_path):
    print('CoNLL-2003 train file not found.')
    exit(1)
sentences = load_conll2003(conll_path)
features = []
labels = []
for sent in sentences:
    features.extend(sent2features(sent))
    labels.extend(sent2labels(sent))

# --- Convert features to DataFrame and encode categorical features ---
df = pd.DataFrame(features)
# Limit prefix/suffix to top 5 most common, others as 'other'
for col in ['prefix', 'suffix']:
    top = df[col].value_counts().nlargest(5).index
    df[col] = df[col].where(df[col].isin(top), 'other')
# Only one-hot encode prefix and suffix, drop 'word' feature
feature_cols = ['is_title', 'is_upper', 'is_digit', 'prefix', 'suffix']
df = df[feature_cols]
df = pd.get_dummies(df, columns=['prefix', 'suffix'])

# --- Train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(df, labels, test_size=0.2, random_state=42)

# --- Train SVM ---
svm = LinearSVC(max_iter=1000)
svm.fit(X_train, y_train)
joblib.dump(svm, 'svm_model.joblib')
print('SVM model trained and saved to svm_model.joblib')

# --- Predict and evaluate ---
y_pred = svm.predict(X_val)
print('\nValidation Results:')
print(classification_report(y_val, y_pred, digits=4))
print(f"F1 Score (macro): {f1_score(y_val, y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_val, y_pred, average='macro'):.4f}")

# --- Optional: Test set evaluation ---
test_path = os.path.join(os.path.dirname(__file__), 'eng.test')
if os.path.exists(test_path):
    test_sents = load_conll2003(test_path)
    test_features = []
    test_labels = []
    for sent in test_sents:
        test_features.extend(sent2features(sent))
        test_labels.extend(sent2labels(sent))
    df_test = pd.DataFrame(test_features)
    for col in ['prefix', 'suffix']:
        top = df_test[col].value_counts().nlargest(5).index
        df_test[col] = df_test[col].where(df_test[col].isin(top), 'other')
    df_test = df_test[feature_cols]
    df_test = pd.get_dummies(df_test, columns=['prefix', 'suffix'])
    # Align test columns with train columns
    df_test = df_test.reindex(columns=df.columns, fill_value=0)
    y_test_pred = svm.predict(df_test)
    print('\nTest Results:')
    print(classification_report(test_labels, y_test_pred, digits=4))
    print(f"F1 Score (macro): {f1_score(test_labels, y_test_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(test_labels, y_test_pred, average='macro'):.4f}")
else:
    print("\nTest file not found. Please add 'eng.test' to your workspace for test evaluation.")
