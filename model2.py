import requests
from bs4 import BeautifulSoup
import streamlit as st
from pathlib import Path
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import google.generativeai as genai
import re
import difflib
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Set up Gemini API with your provided key
genai.configure(api_key="AIzaSyAwRfaNb9foTWvAUUmpqZfbf54Ed5VcFvo")

@dataclass
class SimilarityResult:
    overall_similarity: float
    matching_sections: List[Tuple[str, str]]

# Define functions for text extraction
def extract_text(file, file_extension):
    """Extract text from the uploaded document."""
    if file_extension == ".txt":
        return file.read().decode("utf-8")
    elif file_extension == ".csv":
        return file.read().decode("utf-8")
    elif file_extension == ".docx":
        import docx
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    elif file_extension == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages])
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# Function to normalize and vectorize text using transformers (BERT)
def vectorize_text(text, tokenizer, model):
    """Convert text into embeddings using a transformer model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Average token embeddings
    return embeddings

def compute_similarity(text1, text2, tokenizer, model):
    """Compute cosine similarity between two text embeddings."""
    embeddings1 = vectorize_text(text1, tokenizer, model)
    embeddings2 = vectorize_text(text2, tokenizer, model)
    similarity = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())[0][0]
    return similarity * 100  # Return as percentage

def compute_code_similarity(code1, code2):
    """Compute similarity between two code files."""
    matcher = difflib.SequenceMatcher(None, code1, code2)
    return matcher.ratio() * 100

def scrape_web_and_compare(doc_text, tokenizer, model):
    """Scrape the web and compare with the uploaded document."""
    st.info("Scraping the web for similar content...")
    query = "+".join(doc_text.split()[:10])  # Use the first 10 words as the search query
    search_url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    results = []
    try:
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a['href'] for a in soup.select('a') if 'http' in a.get('href', '')]

        for link in links[:5]:  # Limit to the first 5 links
            try:
                page_response = requests.get(link, headers=headers)
                page_soup = BeautifulSoup(page_response.text, "html.parser")
                page_text = re.sub(r'\s+', ' ', page_soup.get_text(strip=True))
                similarity = compute_similarity(doc_text, page_text, tokenizer, model)
                results.append((link, similarity))
            except Exception:
                continue

        results = sorted(results, key=lambda x: x[1], reverse=True)
        report = "# Web Similarity Report\n\n"
        for i, (link, similarity) in enumerate(results, 1):
            report += f"### Result {i}\n"
            report += f"**Link:** {link}\n"
            report += f"**Similarity:** {similarity:.2f}%\n\n"
        return report
    except Exception as e:
        st.error(f"Error scraping the web: {str(e)}")
        return "Error during web scraping."

# Function to convert code using the Gemini API
def convert_code(source_code, target_language):
    """Convert the source code to the target programming language using Gemini API."""
    prompt = f"""Convert the following code to {target_language} while maintaining its functionality:

{source_code}

Provide only the converted code without any explanations or markdown."""
    try:
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise ValueError(f"Error converting code: {str(e)}")

# Main Streamlit App
def main():
    st.set_page_config(page_title="Plagiarism Detector", layout="wide")
    st.title("Document and Code Plagiarism Detector")

    st.sidebar.title("Instructions")
    st.sidebar.write("""
    1. Upload two documents to compare for plagiarism.
    2. Upload two code files for similarity analysis.
    3. Upload one document to check against web content.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.header("First Document")
        file1 = st.file_uploader("Upload first document", type=["txt", "pdf", "docx", "csv"])
    with col2:
        st.header("Second Document")
        file2 = st.file_uploader("Upload second document", type=["txt", "pdf", "docx", "csv"])

    code_file1 = st.file_uploader("Upload first code file", type=["py", "js", "cpp", "c"])
    code_file2 = st.file_uploader("Upload second code file", type=["py", "js", "cpp", "c"])

    web_file = st.file_uploader("Upload document to check against web", type=["txt", "pdf", "docx", "csv"])

    # Initialize tokenizer and model for vectorization (BERT)
    tokenizer = BertTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    model = BertModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

    # Document comparison
    if file1 and file2:
        ext1, ext2 = Path(file1.name).suffix, Path(file2.name).suffix
        with st.spinner("Processing documents..."):
            text1 = extract_text(file1, ext1)
            text2 = extract_text(file2, ext2)
        similarity = compute_similarity(text1, text2, tokenizer, model)
        st.header("Document Similarity Results")
        st.metric("Overall Similarity", f"{similarity:.2f}%")
        st.text_area("First Document", text1[:500], height=150)
        st.text_area("Second Document", text2[:500], height=150)

    # Code comparison
    if code_file1 and code_file2:
        code1 = code_file1.read().decode("utf-8")
        code2 = code_file2.read().decode("utf-8")
        # Convert code to the same language (e.g., Python to C++)
        if Path(code_file1.name).suffix != Path(code_file2.name).suffix:
            if Path(code_file1.name).suffix != '.cpp':
                code1 = convert_code(code1, "C++")
            if Path(code_file2.name).suffix != '.cpp':
                code2 = convert_code(code2, "C++")
        similarity = compute_code_similarity(code1, code2)
        st.header("Code Similarity Results")
        st.metric("Code Similarity", f"{similarity:.2f}%")
        st.text_area("First Code File", code1[:500], height=150)
        st.text_area("Second Code File", code2[:500], height=150)

    # Web scraping comparison
    if web_file:
        ext = Path(web_file.name).suffix
        with st.spinner("Processing document..."):
            web_text = extract_text(web_file, ext)
        with st.spinner("Scraping the web..."):
            web_report = scrape_web_and_compare(web_text, tokenizer, model)
        st.header("Web Similarity Results")
        st.text_area("Web Report", web_report, height=300)

if __name__ == "__main__":
    main()




