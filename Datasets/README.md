# 🐦 Hybrid BERT + Metadata Twitter Bot Detection
### Using TwiBot-20 Dataset

This repository implements a **Hybrid Deep Learning Framework** for detecting Twitter bot accounts by combining **BERT-based textual representations** with **numerical user metadata features**. The system is designed to capture both **linguistic behavior** and **account-level characteristics**, enabling robust and real-world bot detection.

The model is trained and evaluated using the **TwiBot-20 dataset**, a large-scale and widely accepted benchmark dataset for Twitter bot detection research.

---

## 📌 Dataset Information

- **Dataset Name:** TwiBot-20  
- **Source:** Kaggle  
- **Dataset Link:**  
 https://www.kaggle.com/datasets/marvinvanbo/twibot-20  

TwiBot-20 is a large-scale, human-annotated Twitter dataset created to simulate real-world bot detection scenarios by including diverse bot behaviors and genuine human activity.

---

## 📊 Dataset Overview

- **Total Twitter Accounts:** 229,573  
- **Labeled Accounts:** 11,830  
- **Bot Accounts:** ~6,547  
- **Human Accounts:** ~5,283  
- **Unlabeled Accounts:** ~217,000  
- **Annotation Type:** Manual + Rule-based  
- **Task:** Binary Classification (Bot / Human)  
- **File Format:** JSON  

Only **labeled users** are used for supervised training and evaluation.

---

## 🧠 Why TwiBot-20?

- Realistic Twitter behavior patterns  
- Multiple bot strategies (spam bots, follow bots, content bots)  
- Rich user metadata  
- Multiple tweets per user  
- Ideal for **text + metadata fusion models**  
- Frequently used in **IEEE and research publications**

---

## 📁 Dataset Directory Structure

After extracting the dataset:
TwiBot-20/
├── train.json
├── dev.json
├── test.json
└── README.md


---

## 📄 File-wise Description

### 🔹 train.json
- Used for model training  
- Contains the largest portion of labeled users  
- Each entry corresponds to one Twitter account  
- Includes tweets, user metadata, and ground truth label  

### 🔹 dev.json
- Used for validation  
- Helps in hyperparameter tuning and early stopping  
- Same structure as train.json  

### 🔹 test.json
- Used for final evaluation  
- Never seen during training  
- Used to compute accuracy, precision, recall, F1-score, and ROC-AUC  

---

## 🧬 JSON Data Structure

Each JSON file is a dictionary indexed by **user_id**.



{
"user_id": {
"label": "bot / human",
"profile": { ... },
"tweet": { ... }
}
}


---

## 🏷️ Label Field

- `"bot"`   → Automated Twitter account  
- `"human"` → Genuine human-operated account  

---

## 👤 Profile Object (Metadata Features)

The `profile` field contains numerical and categorical user metadata used as input to the **MLP branch** of the hybrid model.



"profile": {
"followers_count": 1200,
"friends_count": 300,
"listed_count": 15,
"statuses_count": 4500,
"verified": false
}


### Metadata Features Used

- followers_count – Number of followers  
- friends_count – Number of accounts followed  
- listed_count – Number of public lists user appears in  
- statuses_count – Total number of tweets posted  
- verified – Twitter verification status  

All numerical features are **normalized** before being fed into the model.

---

## 📝 Tweet Object (Textual Features)

The `tweet` field contains multiple tweets posted by the user.



"tweet": {
"tweet_id_1": {
"text": "This is a sample tweet"
},
"tweet_id_2": {
"text": "Another tweet text here"
}
}
## 📚 Citation

If you use the TwiBot-20 dataset in academic research, please cite the original authors accordingly.

---

## ✅ Summary

This project demonstrates that **combining linguistic signals from tweets with behavioral metadata** significantly improves Twitter bot detection performance compared to text-only approaches. The TwiBot-20 dataset enables realistic evaluation and makes this system suitable for both academic research and real-world deployment.

