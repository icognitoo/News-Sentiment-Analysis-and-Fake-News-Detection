📰 Fake News & Sentiment Analysis with Explainable AI

This project implements end-to-end NLP pipelines for:

Fake News Detection 📰

Sentiment Analysis 😊😡😐

using deep learning (LSTM, BiLSTM, CNN, Hybrid CNN–BiLSTM), transformer models (BERT, RoBERTa), and Explainable AI techniques (LIME, SHAP).
It also includes a continuous learning mechanism to adapt models with new data over time.

🚀 Features

Data Preprocessing (tokenization, stopword removal, lemmatization, named entity recognition with spaCy).

Multiple Model Architectures (LSTM, BiLSTM, CNN, Hybrid CNN-BiLSTM).

Evaluation (Accuracy, Precision, Recall, F1-score, Confusion Matrices).

Explainability (LIME-based word importance visualization).

Continuous Learning (incremental model updates with new data).

Works with real datasets (WELFake dataset, Twitter Sentiment dataset) or sample data if unavailable.

🛠 Environment Setup

Run the following in Google Colab / VS Code terminal:

pip install transformers torch torchvision torchaudio
pip install datasets
pip install lime shap
pip install nltk spacy
pip install scikit-learn pandas numpy matplotlib seaborn
pip install wordcloud
pip install tensorflow


Download required NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')


Download spaCy model:

python -m spacy download en_core_web_sm

📂 Dataset

Fake News Dataset (WELFake): Zenodo Link

Twitter Sentiment Dataset: GitHub Link

The script automatically downloads these if not present.
If download fails, sample data is generated for demonstration.

▶️ How to Run

Clone the repository:

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>


Open the code in Google Colab or VS Code Jupyter Notebook.

Run cells in order:

Environment setup

Data loading & preprocessing

Model training

Evaluation

Explainable AI (LIME visualizations)

Continuous learning updates

📊 Results

Trains and evaluates multiple models.

Generates accuracy/loss plots and confusion matrices.

Produces word-level importance explanations for predictions.

🔍 Example Outputs

Confusion Matrix

Training Accuracy/Loss plots

Word importance visualization from LIME

📌 Future Improvements

Integrate SHAP visualizations for deeper interpretability.

Deploy as an API / Streamlit app for real-world use.

Expand dataset for broader coverage.

👨‍💻 Author

Developed by Tanmay Singh

📧 Contact: stanmay2504@gmail.com
