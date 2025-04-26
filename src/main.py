# -*- coding: utf-8 -*-
"""Text Classification using BERT.ipynb
"""

# Load packages 
import pandas as pd
import numpy as np



# Load & inspect the data
df = pd.read_csv(r"data/IMDB Dataset.csv")                # Load IMDB CSV
df.head()


df['sentiment'].value_counts()



# Basic feature engineering: map sentiment to numeric
df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})



# Exploratory Data Analysis (EDA)
import seaborn as sns
import matplotlib.pyplot as plt

# Class distribution
sns.countplot(x='sentiment', data=df)
plt.title('Class Distribution: Positive vs Negative')
plt.show()



# Review length distribution
df['review_len'] = df['review'].apply(lambda x: len(x.split()))
plt.hist(df['review_len'], bins=50)
plt.xlabel('Review Length (words)')
plt.ylabel('Number of Reviews')
plt.title('Distribution of Review Lengths')
plt.show()



# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)



# Baseline model: TF-IDF + Logistic Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc



# Vectorize text to TF-IDF features
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)



# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)



# Evaluate baseline
y_pred_lr = lr.predict(X_test_tfidf)
print("Baseline Logistic Regression\n")
print(classification_report(y_test, y_pred_lr))



# Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', xticklabels=['neg','pos'], yticklabels=['neg','pos'])
plt.title('LR Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.show()



# ROC curve for baseline
y_proba_lr = lr.predict_proba(X_test_tfidf)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f'LR (AUC={roc_auc_lr:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(); plt.show()



import warnings
warnings.filterwarnings('ignore')

# Fine-tune BERT with Hugging Face Transformers
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
import evaluate



# Tokenizer and dataset setup 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Attempt bits-and-bytes 8-bit quantization
use_8bit = True
try:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        quantization_config=bnb_config,
        num_labels=2,
    )
    print("✅ Loaded model with 8-bit quantization")
except Exception as e:
    print("❌ 8-bit quant failed:", e)
    print("⚙️  Falling back to standard FP16 model")
    use_8bit = False
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )



# Prepare Hugging Face Datasets
train_df = pd.DataFrame({'text': X_train.values, 'label': y_train.values})
test_df  = pd.DataFrame({'text': X_test.values,  'label': y_test.values})
train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)



# Tokenization function
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)



train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)



# Set format for PyTorch
train_ds = train_ds.remove_columns(['text'])
test_ds  = test_ds.remove_columns(['text'])
train_ds.set_format('torch', columns=['input_ids','attention_mask','label'])
test_ds.set_format('torch', columns=['input_ids','attention_mask','label'])



# Load model & define Trainer
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)



accuracy_metric = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)



training_args = TrainingArguments(
    output_dir='./bert_out',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    save_strategy='no',
    logging_steps=100,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)



# Train & evaluate
trainer.train()
bert_results = trainer.evaluate()
print("BERT Evaluation:", bert_results)



# BERT: Confusion matrix & ROC
import torch
pred_out = trainer.predict(test_ds)
bert_preds = np.argmax(pred_out.predictions, axis=-1)



cm_bert = confusion_matrix(test_df['label'], bert_preds)
sns.heatmap(cm_bert, annot=True, fmt='d',
            xticklabels=['neg','pos'], yticklabels=['neg','pos'])
plt.title('BERT Confusion Matrix'); plt.show()



bert_probs = torch.softmax(torch.tensor(pred_out.predictions), dim=1)[:,1].numpy()
fpr_b, tpr_b, _ = roc_curve(test_df['label'], bert_probs)
roc_auc_b = auc(fpr_b, tpr_b)
plt.plot(fpr_b, tpr_b, label=f'BERT (AUC={roc_auc_b:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.title('BERT ROC Curve'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.show()















