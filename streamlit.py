import streamlit as st
from search_engine import extract_text_from_pdf, extract_text_from_url, search_similar_articles, vectorize_text

# Streamlit App
st.title("Search Tool: PDF and Web Content")

# Upload PDF File
uploaded_file = st.file_uploader("Upload a PDF file for search", type=["pdf"])
if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text from PDF", pdf_text, height=200)

# Input URL
url = st.text_input("Enter a website URL for search")
if url:
    web_text = extract_text_from_url(url)
    st.text_area("Extracted Text from Website", web_text, height=200)

# Enter search query
query = st.text_input("Enter your search query:")
if st.button("Search"):
    documents = []
    if uploaded_file:
        documents.append(pdf_text)
    if url:
        documents.append(web_text)

    if documents:
        X, vectorizer = vectorize_text(documents)
        results = search_similar_articles(query, documents, vectorizer)
        st.write("Search Results:")
        for result, score in results:
            st.write(f"Score: {score:.2f}")
            st.write(result)
    else:
        st.warning("Please upload a file or enter a URL to search.")
