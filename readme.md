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
   python3 -m venv ner_crf
   source ner_crf/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the CoNLL-2003 training data:**
   ```bash
   wget https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/train.txt -O eng.train
   ```

## Training the Model
Run the following command to train the CRF model on the CoNLL-2003 dataset:
```bash
python train_crf.py
```
This will create a file named `crf_model.joblib` in the NLP directory.

## Running Predictions
After training, you can run the prediction script:
```bash
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
