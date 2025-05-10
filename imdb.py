import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import string
from collections import Counter
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

# Download required NLTK resources (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load model and vectorizer with caching
@st.cache_resource
def load_resources():
    try:
        model = load_model("ffnn_model.keras")
    except:
        # Try alternative extensions if .keras fails
        try:
            model = load_model("ffnn_model.h5")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None, None
    
    try:
        tfidf = joblib.load("ffnn.pkl")
    except Exception as e:
        st.error(f"Failed to load vectorizer: {str(e)}")
        return None, None
    
    return model, tfidf

model, tfidf = load_resources()

# Text preprocessing functions
def text_preprocessing(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+|https?|http', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U000025A0-\U000025FF"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)

    # 5. Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    # 6. Remove domain suffixes
    text = re.sub(r'\b(?:\.?com|\.?net|\.?org|\.?in|\.?edu|\.?gov)\b', '', text)

    # 7. Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # 8. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def stage2_token_preprocessing(text, correct_spelling=False):
    if not text:
        return ""
    
    # 1. Tokenization
    tokens = word_tokenize(text)

    # 2. Stopword removal
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # 3. POS tagging
    pos_tags = pos_tag(tokens)

    # 4. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    lemmatized_text = " ".join(lemmatized_tokens)

    return lemmatized_text

synonym_dict = {
    'excellent': 'good',
    'great': 'good',
    'awesome': 'good',
    'amazing': 'good',
    'terrible': 'bad',
    'awful': 'bad',
    'horrible': 'bad',
    'poor': 'bad'
}

def normalize_synonyms(text):
    tokens = word_tokenize(text)
    return ' '.join([synonym_dict.get(word, word) for word in tokens])

def frequency_filter(text):
    words = text.split()
    if not words:
        return ""
    
    word_freq = Counter(words)
    min_freq = 2
    max_freq = 0.8 * len(words)
    
    return ' '.join([word for word in words if min_freq <= word_freq[word] <= max_freq])

def preprocess_pipeline(text):
    text = text_preprocessing(text)
    text = stage2_token_preprocessing(text)
    text = normalize_synonyms(text)
    text = frequency_filter(text)
    return text

def predict_sentiment(review_text):
    if not model or not tfidf:
        return "Error", 0.0
    
    cleaned = preprocess_pipeline(review_text)
    try:
        X_vec = tfidf.transform([cleaned])
        prediction = model.predict(X_vec)
        sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
        return sentiment, float(prediction[0][0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0

# Streamlit UI
def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")
    
    st.title("Sentiment Analysis with FFNN")
    st.write("Analyze the sentiment of your text using a Feedforward Neural Network")
    
    with st.expander("ℹ️ About this app"):
        st.write("""
        This app uses a pre-trained neural network to classify text sentiment as Positive or Negative.
        The model processes text through several NLP techniques before making predictions.
        """)
    
    # Input section
    input_option = st.radio("Input type:", ("Single text", "Batch processing"))
    
    if input_option == "Single text":
        text_input = st.text_area("Enter your text here:", height=150)
        
        if st.button("Analyze Sentiment"):
            if text_input.strip():
                with st.spinner("Processing..."):
                    sentiment, confidence = predict_sentiment(text_input)
                    
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment", sentiment)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}")
                    
                    # Visualize confidence
                    st.progress(confidence if sentiment == "Positive" else 1-confidence)
                    
                    # Show processed text
                    with st.expander("View processed text"):
                        processed = preprocess_pipeline(text_input)
                        st.write(processed)
            else:
                st.warning("Please enter some text to analyze")
    else:
        uploaded_file = st.file_uploader("Upload a text file (one entry per line)", type=["txt"])
        if uploaded_file:
            lines = uploaded_file.read().decode("utf-8").splitlines()
            results = []
            
            if st.button("Analyze All"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, line in enumerate(lines):
                    if line.strip():
                        sentiment, confidence = predict_sentiment(line)
                        results.append({
                            "Text": line[:100] + "..." if len(line) > 100 else line,
                            "Sentiment": sentiment,
                            "Confidence": f"{confidence:.2f}"
                        })
                    
                    progress = (i + 1) / len(lines)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{len(lines)}...")
                
                progress_bar.empty()
                status_text.empty()
                
                if results:
                    st.subheader("Batch Results")
                    st.dataframe(results)
                    
                    # Download results
                    csv = "\n".join([f"{r['Text']},{r['Sentiment']},{r['Confidence']}" for r in results])
                    st.download_button(
                        "Download Results",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
    
    # Model information
    st.sidebar.header("Model Information")
    st.sidebar.write("""
    - **Model Type**: Feedforward Neural Network
    - **Input Processing**: TF-IDF Vectorization
    - **Preprocessing Steps**:
        - Text cleaning
        - Tokenization
        - Lemmatization
        - Synonym normalization
        - Frequency filtering
    """)

if __name__ == "__main__":
    main()