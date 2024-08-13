pip install pandas numpy matplotlib seaborn nltk PyPDF2


import PyPDF2
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

pip install fpdf
from fpdf import FPDF

def create_sample_pdf(pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="This is a sample PDF.", ln=True, align='C')
    pdf.cell(200, 10, txt="You can extract text from this PDF.", ln=True, align='C')
    pdf.output(pdf_path)


sample_pdf_path = 'sample.pdf'
create_sample_pdf(sample_pdf_path)
print(f"Sample PDF created at: {sample_pdf_path}")

import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


pdf_path = 'sample.pdf' 
extracted_text = extract_text_from_pdf(pdf_path)
print(extracted_text)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    
    tokens = word_tokenize(text.lower())
    
    
    tokens = [word for word in tokens if word.isalpha()]
    
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens


text = "This is a sample text for preprocessing. It includes punctuation and stopwords."
processed_text = preprocess_text(text)
print(processed_text)

import nltk
import pandas as pd

def analyze_document(tokens):
    
    freq_dist = nltk.FreqDist(tokens)
    
    
    freq_df = pd.DataFrame(freq_dist.items(), columns=['Word', 'Frequency'])
    
    
    freq_df = freq_df.sort_values(by='Frequency', ascending=False)
    
    return freq_df


tokens = ['this', 'is', 'a', 'sample', 'text', 'with', 'sample', 'words', 'for', 'frequency', 'analysis']
frequency_df = analyze_document(tokens)
print(frequency_df)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk

def analyze_document(tokens):
    
    freq_dist = nltk.FreqDist(tokens)
    
    
    freq_df = pd.DataFrame(freq_dist.items(), columns=['Word', 'Frequency'])
    
    
    freq_df = freq_df.sort_values(by='Frequency', ascending=False)
    
    return freq_df

def visualize_word_frequency(freq_df, top_n=20):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Word', data=freq_df.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.show()

tokens = ['this', 'is', 'a', 'sample', 'text', 'with', 'sample', 'words', 'for', 'frequency', 'analysis', 'text', 'is', 'a', 'sample']
frequency_df = analyze_document(tokens)
visualize_word_frequency(frequency_df)


def document_analysis_pipeline(pdf_path):
    
    text = extract_text_from_pdf(pdf_path)
    
    
    tokens = preprocess_text(text)
    
    
    freq_df = analyze_document(tokens)
    
    
    visualize_word_frequency(freq_df)


pdf_path = 'sample.pdf'


document_analysis_pipeline(pdf_path)
