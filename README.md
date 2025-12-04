# Team 12 – HybridBERT + Metadata Deep Learning Model for Twitter Bot Detection

## Team Info
- **22471A05O1 — Shaik Shakeer Ahamad** 
  Work Done: Model design, BERT integration, training, evaluation, research writing, and Front-end.

- **22471A05P0 — Shaik Chinna Mastan Vali 2** 
  Work Done: Major documentation.

- **22471A05P3 — M Phani Kumar 3**  
  Work Done: PPT Documentation



---

## Abstract
This work presents a hybrid bot-detection system that merges BERT-based text representations with numerical metadata from Twitter user profiles. The architecture captures linguistic signals from tweets and behavioural traits from user information, enabling stronger and more stable predictions. The model was trained on the TwiBot-20 dataset and provides improved detection of automated accounts under noisy real-world scenarios.

---

## Paper Reference (Inspiration)
**Paper:**  
A Deep Learning Approach for Robust Detection of Bots in Twitter Using Transformers  
Authors: Subhabrata Mukherjee et al.  
Paper Link: (https://ieeexplore.ieee.org/document/10040412/)

---

## Our Improvement Over Existing Paper
- Added a metadata-processing branch using an MLP.
- Combined BERT embeddings + metadata using a fusion layer.
- Used weighted sampling and pos_weight to address imbalance.
- Added early stopping, warmup, and cosine LR decay.
- Implemented real-time prediction through Flask + React.
- Added ROC-AUC, confusion matrix, and confidence plots.
- More stable performance on human class compared to text-only models.

---

## About the Project
- Detects whether a Twitter account behaves like a human or a bot.  
- Uses both tweet content and metadata (followers, friends, statuses, verified, etc.).  
- Workflow: Input → Tokenizer/Metadata Scaling → Hybrid Model → Output (Bot/Human + Confidence).  
- Useful in spam filtering, misinformation tracking, and social media integrity studies.

---

## Dataset Used
Dataset: **TwiBot-20**  
URL: https://www.kaggle.com/datasets/marvinvanbo/twibot-20

### Dataset Details
- 11k+ Twitter users  
- ~500k tweets  
- Labels: bot or human  
- Includes train/dev/test JSON files  
- Contains full user profile metadata + tweets  

---

## Dependencies Used
- Python 3.x  
- PyTorch  
- HuggingFace Transformers  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- tqdm  
- Flask  
- ReactJS  
- CUDA (for GPU training)

---

## EDA & Preprocessing
- Cleaned string-based metadata fields and converted to numeric.
- Merged all tweets of each user into a single text block.
- Tokenized text using BERT tokenizer.  
- Normalized metadata using StandardScaler.  
- Visualized distributions: followers, friends, statuses, tweet length.  
- Applied weighted sampling for class balance.

---

## Model Training Info
- Architecture: Hybrid BERT + Metadata MLP Fusion  
- Loss: BCEWithLogitsLoss (with pos_weight)  
- Optimizer: AdamW  
- Scheduler: Warmup + Cosine Annealing  
- Epochs: 5–8  
- Batch Size: 16–32  
- Early stopping based on dev F1  
- GPU: Colab T4/P100  

---

## Model Testing / Evaluation
- Calculated Accuracy, Precision, Recall, F1  
- Computed ROC-AUC  
- Generated confusion matrix  
- Visualized confidence distribution  
- Compared:
  - BERT only  
  - Metadata only  
  - Hybrid model  

---

## Results
- Achieved F1-Score: ~0.93–0.94  
- ROC-AUC: >0.95  
- Reduced false positives for human class  
- Stable predictions even with noisy tweet content  
- Real-time API gives confidence score for each prediction  

---

## Limitations & Future Work
- Requires GPU for fast inference  
- Missing metadata can reduce accuracy  
- Future enhancements:
  - Add user-network features (graph-based)  
  - Multilingual tweet handling  
  - Larger transformer models (RoBERTa/DeBERTa)  
  - ONNX model for deployment on low-resource systems  

---

## Deployment Info
- Backend: Flask API ( `/predict` endpoint )  
- Frontend: ReactJS UI  
- Supports:
  - Twitter handle input (via API)  
  - Manual input for text/metadata  
  - Confidence and visualization output  


