import streamlit as st
import PyPDF2
import docx
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from pathlib import Path
import google.generativeai as genai
import difflib
from dataclasses import dataclass
from typing import List, Tuple
import tempfile
import os
import re

# Set up Gemini API with your provided key
genai.configure(api_key="AIzaSyAwRfaNb9foTWvAUUmpqZfbf54Ed5VcFvo")

# Define custom exceptions
class DocumentProcessingError(Exception):
    pass

class CodeConversionError(Exception):
    pass

@dataclass
class DocumentContent:
    text: str
    code_snippets: List[str]
    original_format: str

@dataclass
class SimilarityResult:
    overall_similarity: float
    matching_sections: List[Tuple[str, str]]
    code_similarities: List[Tuple[str, str, float]]

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = {'.txt', '.pdf', '.docx', '.csv'}

    def extract_text(self, file, file_format: str) -> DocumentContent:
        if file_format not in self.supported_formats:
            raise DocumentProcessingError(f"Unsupported format: {file_format}")
            
        try:
            if file_format == '.txt':
                return self._process_txt(file)
            elif file_format == '.pdf':
                return self._process_pdf(file)
            elif file_format == '.docx':
                return self._process_docx(file)
            elif file_format == '.csv':
                return self._process_csv(file)
        except Exception as e:
            raise DocumentProcessingError(f"Error processing {file_format} file: {str(e)}")

    def _process_txt(self, file) -> DocumentContent:
        content = file.read().decode('utf-8')
        return self._separate_code_and_text(content, '.txt')

    def _process_pdf(self, file) -> DocumentContent:
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(file.read())
                temp_file.flush()
                
                pdf_reader = PyPDF2.PdfReader(temp_file.name)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
            
            return self._separate_code_and_text(content, '.pdf')
        except Exception as e:
            raise DocumentProcessingError(f"Error processing PDF file: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def _process_docx(self, file) -> DocumentContent:
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(file.read())
                temp_file.flush()
                
                doc = docx.Document(temp_file.name)
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            return self._separate_code_and_text(content, '.docx')
        except Exception as e:
            raise DocumentProcessingError(f"Error processing DOCX file: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def _process_csv(self, file) -> DocumentContent:
        content = file.read().decode('utf-8')
        df = pd.read_csv(pd.StringIO(content))
        text_content = df.to_string()
        return DocumentContent(text=text_content, code_snippets=[], original_format='.csv')

    def _separate_code_and_text(self, content: str, format_type: str) -> DocumentContent:
        code_markers = ['def ', 'class ', 'import ', 'function', '{', '}', 'for(', 'while(']
        lines = content.split('\n')
        code_snippets = []
        text_lines = []
        
        current_code = []
        in_code_block = False
        
        for line in lines:
            is_code_line = any(marker in line for marker in code_markers)
            
            if is_code_line and not in_code_block:
                in_code_block = True
                if current_code:
                    code_snippets.append('\n'.join(current_code))
                    current_code = []
            elif not is_code_line and in_code_block:
                in_code_block = False
                if current_code:
                    code_snippets.append('\n'.join(current_code))
                    current_code = []
                text_lines.append(line)
            elif in_code_block:
                current_code.append(line)
            else:
                text_lines.append(line)
        
        if current_code:
            code_snippets.append('\n'.join(current_code))
            
        return DocumentContent(
            text='\n'.join(text_lines),
            code_snippets=code_snippets,
            original_format=format_type
        )

class CodeConverter:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            raise CodeConversionError(f"Error initializing Gemini API: {str(e)}")
        
    def convert_code(self, source_code: str, target_language: str) -> str:
        try:
            prompt = f"""Convert the following code to {target_language} while maintaining its functionality:

{source_code}

Provide only the converted code without any explanations or markdown."""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise CodeConversionError(f"Error converting code: {str(e)}")

class SimilarityAnalyzer:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
            self.model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
            self.max_tokens = 512
        except Exception as e:
            raise Exception(f"Error initializing BERT model: {str(e)}")

    def normalize_code(self, code: str) -> str:
        """Remove extra whitespace and comments to normalize code for comparison."""
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)  # Remove single-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Remove multi-line comments
        code = re.sub(r'\s+', ' ', code).strip()  # Normalize whitespace
        return code

    def compute_code_similarity(self, code1: str, code2: str) -> float:
        """Compute similarity between two code strings after normalization."""
        code1 = self.normalize_code(code1)
        code2 = self.normalize_code(code2)
        
        similarity = difflib.SequenceMatcher(None, code1, code2).ratio()
        return similarity * 100  # Return as percentage

    def compute_similarity(self, doc1: DocumentContent, doc2: DocumentContent) -> SimilarityResult:
        """Compute overall text and code similarity between two documents."""
        text_similarity = self._compute_bert_similarity(doc1.text, doc2.text)
        matching_sections = self._find_matching_sections(doc1.text, doc2.text)

        code_similarities = []
        for code1 in doc1.code_snippets:
            for code2 in doc2.code_snippets:
                similarity = self.compute_code_similarity(code1, code2)
                if similarity > 80:  # Threshold for similarity
                    code_similarities.append((code1, code2, similarity))

        return SimilarityResult(
            overall_similarity=text_similarity,
            matching_sections=matching_sections,
            code_similarities=code_similarities
        )

    def _compute_bert_similarity(self, text1: str, text2: str) -> float:
        """Compute BERT-based text similarity between two documents."""
        encodings1 = self.tokenizer(text1, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt')
        encodings2 = self.tokenizer(text2, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt')
        
        with torch.no_grad():
            output1 = self.model(**encodings1)
            output2 = self.model(**encodings2)
        
        embeddings1 = torch.mean(output1.last_hidden_state, dim=1)
        embeddings2 = torch.mean(output2.last_hidden_state, dim=1)
        
        similarity = F.cosine_similarity(embeddings1, embeddings2).item()
        return max(0, similarity) * 100  # Return as percentage

    def _find_matching_sections(self, text1: str, text2: str) -> List[Tuple[str, str]]:
        matcher = difflib.SequenceMatcher(None, text1, text2)
        matching_sections = []
        for block in matcher.get_matching_blocks():
            if block.size > 50:
                match1 = text1[block.a:block.a + block.size]
                match2 = text2[block.b:block.b + block.size]
                matching_sections.append((match1, match2))
        return matching_sections

def create_similarity_report(result: SimilarityResult) -> str:
    report = f"""# Plagiarism Detection Report

## Overall Similarity: {result.overall_similarity:.2f}%

## Matching Text Sections
"""
    for i, (text1, text2) in enumerate(result.matching_sections, 1):
        report += f"\n### Match {i}\n"
        report += f"Document 1: {text1}\n"
        report += f"Document 2: {text2}\n\n"
    
    report += "\n## Matching Code Sections\n"
    for i, (code1, code2, similarity) in enumerate(result.code_similarities, 1):
        report += f"\n### Code Match {i} (Similarity: {similarity:.2f}%)\n"
        report += f"Document 1 Code:\n{code1}\n\n"
        report += f"Document 2 Code:\n{code2}\n\n"
    
    return report

def main():
    st.set_page_config(page_title="Document and Code Plagiarism Detector", layout="wide")
    
    st.title("Document and Code Plagiarism Detector")
    
    st.sidebar.title("Instructions")
    st.sidebar.write("""
    1. Upload two documents to compare
    2. Upload two code files for plagiarism detection
    3. Wait for the analysis to complete
    4. Review the results and download the report
    """)

    st.sidebar.title("Supported Document Formats")
    st.sidebar.write("- Text (.txt), PDF (.pdf), Word (.docx), CSV (.csv)")

    st.sidebar.title("Supported Code Formats")
    st.sidebar.write("- C (.c), C++ (.cpp), Python (.py), JavaScript (.js)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("First Document")
        file1 = st.file_uploader("Upload first document", type=['txt', 'pdf', 'docx', 'csv'])
    
    with col2:
        st.header("Second Document")
        file2 = st.file_uploader("Upload second document", type=['txt', 'pdf', 'docx', 'csv'])

    code_file1 = st.file_uploader("Upload first code file", type=['c', 'cpp', 'py', 'js'])
    code_file2 = st.file_uploader("Upload second code file", type=['c', 'cpp', 'py', 'js'])
    
    if file1 and file2:
        processor = DocumentProcessor()
        analyzer = SimilarityAnalyzer()

        with st.spinner('Processing documents...'):
            doc1 = processor.extract_text(file1, Path(file1.name).suffix)
            doc2 = processor.extract_text(file2, Path(file2.name).suffix)

        with st.spinner('Analyzing similarity...'):
            result = analyzer.compute_similarity(doc1, doc2)

        st.header("Document Analysis Results")
        st.metric("Overall Similarity", f"{result.overall_similarity:.2f}%")
        
        report = create_similarity_report(result)
        st.download_button("📥 Download Document Report", data=report, file_name="document_report.md", mime="text/markdown")

    if code_file1 and code_file2:
        converter = CodeConverter()
        analyzer = SimilarityAnalyzer()
        
        code1 = code_file1.read().decode('utf-8')
        code2 = code_file2.read().decode('utf-8')
        
        if Path(code_file1.name).suffix != '.cpp':
            code1 = converter.convert_code(code1, "C++")
        if Path(code_file2.name).suffix != '.cpp':
            code2 = converter.convert_code(code2, "C++")

        similarity = analyzer.compute_code_similarity(code1, code2)
        st.header("Code Analysis Results")
        st.metric("Code Similarity", f"{similarity:.2f}%")
        
        code_report = f"# Code Similarity: {similarity:.2f}%"
        st.download_button("📥 Download Code Report", data=code_report, file_name="code_report.md", mime="text/markdown")

if __name__ == "__main__":
    main()
