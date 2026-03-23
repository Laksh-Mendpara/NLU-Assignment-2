import os
import re
import string
import nltk

from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download required nltk resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def clean_text(text: str) -> str:
    """
    Remove boilerplate and formatting artifacts.
    Since we used Trafilatura and PyMuPDF, most boilerplate is gone,
    but we can strip URLs, emails, and markdown artifacts.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    # Remove markdown header artifacts
    text = re.sub(r'#+', '', text)
    # Fix whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_document(doc_text: str) -> list[str]:
    """
    (ii) Tokenization
    (iii) Lower-casing
    (iv) Removal of excessive punctuation and non-textual content
    """
    doc_text = clean_text(doc_text)
    
    # Lowercase
    doc_text = doc_text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(doc_text)
    
    clean_tokens = []
    for token in tokens:
        # Remove punctuation entirely from the token
        token_clean = token.translate(str.maketrans('', '', string.punctuation))
        # Drop tokens that are just numbers or empty
        if token_clean and not token_clean.isnumeric():
            clean_tokens.append(token_clean)
            
    return clean_tokens

def main():
    input_file = "output/data.txt"  # Default output dir from scraper
    output_corpus = "output/processed_corpus.txt"
    wordcloud_img = "output/wordcloud.png"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run dataset_generation first.")
        return

    print("Reading corpus...")
    with open(input_file, "r", encoding="utf-8") as f:
        # Documents are separated by \n\n in data.txt
        raw_documents = f.read().split("\n\n")
    
    # Filter empty splits
    raw_documents = [d for d in raw_documents if d.strip()]
    total_docs = len(raw_documents)
    
    print(f"Preprocessing {total_docs} documents...")
    all_tokens = []
    
    # Process and save
    os.makedirs(os.path.dirname(output_corpus), exist_ok=True)
    with open(output_corpus, "w", encoding="utf-8") as out_f:
        for doc in raw_documents:
            tokens = preprocess_document(doc)
            if tokens:
                all_tokens.extend(tokens)
                # Save each document as a line of space-separated tokens
                out_f.write(" ".join(tokens) + "\n")
                
    total_tokens = len(all_tokens)
    vocab = set(all_tokens)
    vocab_size = len(vocab)
    
    print("=" * 40)
    print("DATASET STATISTICS")
    print(f"Total Documents : {total_docs}")
    print(f"Total Tokens    : {total_tokens}")
    print(f"Vocabulary Size : {vocab_size}")
    print("=" * 40)
    
    print("Generating Word Cloud...")
    # Get top frequencies for a cleaner wordcloud, or just pass full text
    word_freq = Counter(all_tokens)
    
    # Remove standard stop words for the word cloud specifically so it looks nicer
    from nltk.corpus import stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    # additional common academic/meaningless words might be filtered out here if desired
    filtered_freq = {w: c for w, c in word_freq.items() if w not in stop_words}
    
    wc = WordCloud(width=800, height=400, background_color='white', max_words=200)
    wc.generate_from_frequencies(filtered_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("IITJ Corpus Word Cloud")
    plt.tight_layout()
    plt.savefig(wordcloud_img)
    print(f"Word Cloud saved to {wordcloud_img}")

if __name__ == "__main__":
    main()
