# Telegram Spam Message Classifier

A Machine Learning project using **Python** and **scikit-learn** to detect and classify spam messages collected from Telegram channels and chats using a **Bernoulli Naive Bayes** model.

---

## 📌 Project Overview

Telegram channels and group chats frequently suffer from automated spam messages promoting fraudulent work-from-home schemes, cryptocurrency scams, and suspicious links. This project processes raw, cleaned Telegram text messages, converts them into numerical feature vectors, and trains a binary text classification model to accurately identify spam content.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Data Manipulation:** `pandas`
* **Machine Learning:** `scikit-learn`
* `CountVectorizer` (Text Feature Extraction)
* `BernoulliNB` (Bernoulli Naive Bayes Classifier)
* `train_test_split`, `accuracy_score`, `classification_report`



---

## 📊 Dataset Structure

The primary dataset used is `telegram_spam_cleaned.csv`.

| Column Name | Data Type | Description |
| --- | --- | --- |
| `id` | `int64` | Unique message identifier *(dropped during preprocessing)* |
| `date` | `datetime64` | Timestamp when the message was sent |
| `text` | `object` | Raw text content of the Telegram message |
| `label` | `object` | Message classification label (`spam`) |
| `has_media` | `int64` | Binary flag indicating attached media (1 = Yes, 0 = No) |
| `forwarded_from` | `int64` | Binary flag indicating if forwarded (1 = Yes, 0 = No) |

---

## ⚙️ Workflow & Implementation

1. **Data Loading & Preprocessing:**
* Loaded `telegram_spam_cleaned.csv` into a Pandas DataFrame.
* Converted the `date` string column to datetime format using `pd.to_datetime()`.
* Checked for null values across all features.
* Dropped redundant identifier columns (`id`).


2. **Feature Extraction & Data Splitting:**
* Split dataset into training and test sets using `train_test_split` (80% train, 20% test, `random_state=42`).
* Vectorized textual message data using `CountVectorizer`.


3. **Model Training & Evaluation:**
* Trained a **Bernoulli Naive Bayes** (`BernoulliNB`) classifier on the vectorized text data.
* Evaluated model predictions against test targets using Accuracy, Classification Report, and Confusion Matrix.



---

## 📈 Model Performance

* **Model:** Bernoulli Naive Bayes (`BernoulliNB`)
* **Accuracy:** `100.00%`

```text
Classification Report :
               precision    recall  f1-score   support

        spam       1.00      1.00      1.00        40

    accuracy                           1.00        40
   macro avg       1.00      1.00      1.00        40
weighted avg       1.00      1.00      1.00        40

```

---

## 🚀 How to Run

1. **Install required dependencies:**
```bash
pip install pandas scikit-learn

```


2. **Run the Notebook or Python Script:**
Open and execute `spam_detection.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

---
## Model Performance
The Telegram Spam Classification model achieved **100% accuracy** on the test dataset. The classification report shows **1.00 precision, 1.00 recall, and 1.00 F1-score** for the Spam class.

This means the model correctly classified all **40 test samples** as spam. The results indicate excellent performance on the given test dataset. However, further validation on a larger and more diverse dataset would be useful to evaluate the model's ability to generalize to unseen Telegram messages.

## 📁 Repository Structure

```text
├── telegram_spam_cleaned.csv   # Cleaned Telegram dataset
├── spam_detection.ipynb        # Jupyter Notebook with EDA & modeling
└── README.md                   # Project documentation
