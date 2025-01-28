import re
import string
import numpy as np
import pandas as pd
import streamlit as st
import nltk
import requests


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

lemmer = WordNetLemmatizer()
stopwords_list = stopwords.words("english")


stop_words_list = stopwords_list + [
    "things", "that's", "something", "take", "don't", "may", "want", "you're",
    "set", "might", "says", "including", "lot", "much", "said", "know", "good",
    "step", "often", "going", "thing", "things", "think", "back", "actually",
    "better", "look", "find", "right", "example", "verb", "verbs"
]

def preprocess_text(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"@\w+", "", text)
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return " ".join([lemmer.lemmatize(word) for word in text.split() if word not in stop_words_list])

def extract_text_from_file(file):
    if file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf_reader.pages])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    return None

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([para.get_text() for para in paragraphs])
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def run_search_tool():
    st.set_page_config(page_title="Document & Web Search", page_icon="📄", layout="wide")

    st.title("📄 Search Tool")
    st.markdown("Upload files or provide URLs for content search.")

    uploaded_files = st.file_uploader("Upload File", accept_multiple_files=True)
    website_links = st.text_area("Enter URLs (one per line)")

    documents = []
    titles = []

    if uploaded_files:
        for file in uploaded_files:
            extracted_text = extract_text_from_file(file)
            if extracted_text:
                documents.append(preprocess_text(extracted_text))
                titles.append(file.name)

    if website_links:
        for link in website_links.strip().split("\n"):
            extracted_text = extract_text_from_url(link)
            if extracted_text:
                documents.append(preprocess_text(extracted_text))
                titles.append(link)

    if documents:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=10000,
            stop_words=stop_words_list
        )
        tfidf_matrix = vectorizer.fit_transform(documents)

        query = st.text_input("Enter search query:")

        if query:
            query_vector = vectorizer.transform([preprocess_text(query)]).toarray().reshape(-1)

            similarity_scores = {}
            for i, doc_vector in enumerate(tfidf_matrix.toarray()):
                score = np.dot(doc_vector, query_vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(query_vector))
                similarity_scores[i] = score

            ranked_results = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:5]

            st.subheader("🔍 Results")
            for rank, (index, score) in enumerate(ranked_results):
                if score > 0:
                    st.markdown(f"**Title:** {titles[index]} **Score:** {score:.2f}")
                    st.write(f"**result:** {documents[index][:500]}...")
                    st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.info("Upload files or enter URLs to search.")

if __name__ == "__main__":
    run_search_tool()
