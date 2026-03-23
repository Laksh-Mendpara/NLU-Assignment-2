import os
import time
from gensim.models import Word2Vec

def load_corpus(file_path: str):
    """
    Generator to load corpus line by line to save memory.
    Each line is a space-separated string of tokens.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                yield tokens

def train_model(sentences: list, model_type: str, vector_size: int, window: int, negative: int, out_dir: str):
    """
    Train a Word2Vec model (CBOW or Skip-gram) on the given corpus.
    """
    sg = 1 if model_type.lower() == "skip-gram" else 0
    model_name = f"{model_type}_dim{vector_size}_win{window}_neg{negative}"
    
    print(f"Training {model_name}...")
    start_time = time.time()
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=5,     # ignore infrequent words
        sg=sg,           # 0 for CBOW, 1 for Skip-gram
        negative=negative,
        workers=4,       # multicore
        epochs=10
    )
    
    duration = time.time() - start_time
    print(f"  -> Finished in {duration:.2f} seconds.")
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{model_name}.model")
    model.save(save_path)
    return model

def main():
    corpus_file = "output/processed_corpus.txt"
    out_dir = "output/models"
    
    if not os.path.exists(corpus_file):
        print("Corpus not found. Run preprocessing first.")
        return
        
    print("Loading corpus...")
    sentences = list(load_corpus(corpus_file))
        
    # Hyperparameters for the experiment
    architectures = ["CBOW", "Skip-gram"]
    dimensions = [50, 100]
    windows = [2, 5]
    negatives = [5, 10]
    
    print("=" * 60)
    print("STARTING WORD2VEC EXPERIMENTS")
    print("=" * 60)
    
    for arch in architectures:
        for dim in dimensions:
            for win in windows:
                for neg in negatives:
                    train_model(sentences, arch, dim, win, neg, out_dir)
                    
    print("\nAll models trained and saved successfully.")

if __name__ == "__main__":
    main()
