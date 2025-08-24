# CRF-based Named Entity Recognition (NER) with CoNLL-2003 Dataset

This project demonstrates how to train and use a Conditional Random Field (CRF) model for Named Entity Recognition (NER) using the CoNLL-2003 dataset in Python.

## Features
- Trains a CRF model on the CoNLL-2003 NER dataset
- Saves the trained model for fast future predictions
- Provides a simple CLI to predict NER tags for user-input sentences

## Project Structure
```
NLP/
├── crf_model.joblib         # Saved CRF model (created after training)
├── eng.train                # CoNLL-2003 training data (download separately)
├── main.py                  # (Legacy) Combined script
├── run_crf.py               # Run predictions using the trained model
├── train_crf.py             # Train and save the CRF model
├── requirements.txt         # Python dependencies
```

## Setup
1. **Clone the repository**
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv ner_crf
   .\ner_crf\Scripts\Activate.ps1
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the CoNLL-2003 training data:**
   ```
   import kagglehub

   # Download latest version
   path = kagglehub.dataset_download("alaakhaled/conll003-englishversion")

   print("Path to dataset files:", path)

   ```

## Training the Model

Run the following command to train the CRF model on the CoNLL-2003 dataset and evaluate its performance:
```
python train_crf.py
```
This will:
- Split your data into training and validation sets (80/20 split by default)
- Train the CRF model on the training set
- Save the trained model as `crf_model.joblib` in the NLP directory
- **Print evaluation metrics on the validation set, including F1 score, recall, and a full classification report**

**Sample output:**
```
Model trained and saved to /home/codespace/NLP/crf_model.joblib

Validation Results:
          precision    recall  f1-score   support

       LOC     0.90       0.85      0.87      200
       ORG     0.82       0.80      0.81      150
       PER     0.95       0.94      0.95      180
       MISC    0.78       0.80      0.79      100
       O       0.99       1.00      0.99     3000

    accuracy                         0.97      3630
   macro avg     0.89       0.88      0.88     3630
weighted avg     0.97       0.97      0.97     3630

F1 Score (macro): 0.88
Recall (macro): 0.88
```

## Running Predictions
After training, you can run the prediction script:
```
python run_crf.py
```
You will be prompted to enter a sentence. The script will output each word and its predicted NER label in a table.

## Notes
- The first run (training) may take a few minutes depending on your hardware.
- You only need to train once. For future predictions, just use `run_crf.py`.
- If you want to retrain, delete `crf_model.joblib` and run `train_crf.py` again.

## Requirements
- Python 3.7+
- See `requirements.txt` for Python package dependencies

## License
MIT
