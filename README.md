# 🎬 Movie Review Sentiment Analysis

**🔍 NLP | 💬 Sentiment Analysis | 🎓 Machine Learning | 🐍 Python**

A project to classify IMDb movie reviews as **Positive** or **Negative** using traditional ML models.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-Scikit--learn-yellow?logo=scikit-learn)
![UI](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)

---

## 🚀 Project Overview

This beginner-friendly project uses **natural language processing (NLP)** to classify movie reviews from the IMDb dataset into **positive** or **negative** categories. It includes end-to-end implementation of:

* Text cleaning and preprocessing
* Feature extraction using TF-IDF
* Classification using machine learning algorithms
* Model evaluation using accuracy and F1-score
* (Optional) A simple web interface for real-time predictions

---

## 🧐 Features

* ✅ Clean and tokenize raw text
* ✅ Remove stopwords and lowercase input
* ✅ Vectorize text using TF-IDF
* ✅ Train ML models: Logistic Regression, Naïve Bayes, or SVM
* ✅ Evaluate model with accuracy and F1-score
* ✅ Optional: Deploy with a Streamlit web UI

---

## 📁 Project Structure

```
movie-review-sentiment-analysis/
|
├── data/                   # Raw and processed dataset files
├── notebooks/              # Jupyter notebooks for EDA and training
├── models/                 # Saved models (optional)
├── app/                    # Streamlit or Flask interface
├── main.py                 # Main script for training and testing
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and setup instructions

```

---

## 📊 Dataset

**IMDb Movie Reviews Dataset**

* 50,000 labeled movie reviews (positive & negative)
* Balanced dataset
* Commonly used for sentiment classification tasks

🔗 [Download from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## ⚙️ Tech Stack

| Tool              | Purpose                                      |
| ----------------- | -------------------------------------------- |
| 🐍 Python         | Main programming language                    |
| 📦 Scikit-learn   | Model training and evaluation                |
| 🧹 NLTK / spaCy   | Text preprocessing (tokenization, stopwords) |
| 📊 Pandas / NumPy | Data manipulation and statistics             |
| 🌐 Streamlit      | Web-based UI (optional)                      |

---

## 🧪 Evaluation Metrics

The model is evaluated using:

* 🌟 **Accuracy**: Overall percentage of correct predictions
* 🔠 **F1-Score**: Harmonic mean of precision and recall
* 🔀 **Cross-validation** (optional): To check model robustness

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Aalyan-butt/Movie-Review-Sentiment-Analyzer
cd movie-review-sentiment-analysis
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python main.py
```

### 5. (Optional) Launch Web App

```bash
streamlit run app/app.py
```

---

## 🖼️ Example

> **Input Review:** "This movie had great acting and a compelling story."
> **Prediction:** ✅ Positive

---

## 🎯 Future Improvements

* [ ] Integrate deep learning (LSTM, BERT)
* [ ] Use Word2Vec or GloVe for richer embeddings
* [ ] Add multilingual support
* [ ] Enhance the UI with charts and examples

---



## 🙌 Acknowledgments

* IMDb for the dataset
* Scikit-learn and NLTK for NLP and ML tools
* Streamlit for an easy-to-use deployment framework
* Kaggle for dataset hosting

---

## 👤 Author

**Your Name**
📧 [aalyanriasatali@gmail.com](mailto:your.email@example.com)
🔗  • [GitHub](https://github.com/Aalyan-butt)

---
