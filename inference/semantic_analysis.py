import os
from gensim.models import Word2Vec

def run_semantic_analysis(model_path: str, model_name: str):
    print("\n" + "=" * 60)
    print(f" SEMANTIC ANALYSIS: {model_name}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
        
    model = Word2Vec.load(model_path)
    
    # 1. Top 5 Nearest Neighbors
    target_words = ["research", "student", "phd", "exam"]
    print("\n--- 1. Top 5 Nearest Neighbors ---")
    for word in target_words:
        if word in model.wv.key_to_index:
            neighbors = model.wv.most_similar(word, topn=5)
            print(f"\nNeighbors for '{word}':")
            for neighbor, sim in neighbors:
                print(f"  - {neighbor} (sim: {sim:.4f})")
        else:
            print(f"\nWord '{word}' not in vocabulary.")
            
    # 2. Analogy Experiments
    # Format: A is to B as C is to D. We provide A, B, C and look for D.
    # Gensim syntax: most_similar(positive=[B, C], negative=[A])
    analogies = [
        # (A, B, C)
        ("ug", "btech", "pg"),
        ("institute", "academic", "department"),
        ("student", "exam", "faculty"),
        ("theory", "practical", "lecture")
    ]
    print("\n--- 2. Analogy Experiments---")
    
    for a, b, c in analogies:
        # Check if words are in vocab
        missing = [w for w in [a, b, c] if w not in model.wv.key_to_index]
        if missing:
            print(f"\nSkipping analogy '{a}' : '{b}' :: '{c}' : ? (Missing: {missing})")
            continue
            
        print(f"\nAnalogy: '{a}' is to '{b}' as '{c}' is to ?")
        # Evaluate
        preds = model.wv.most_similar(positive=[b, c], negative=[a], topn=3)
        for d, sim in preds:
            print(f"  -> {d} (sim: {sim:.4f})")
            
    print("\nAnalysis complete for", model_name)

def main():
    models_dir = "output/models"
    
    # Lets evaluate a baseline model configuration for both CBOW and Skip-gram
    cbow_model = os.path.join(models_dir, "CBOW_dim100_win5_neg5.model")
    sg_model = os.path.join(models_dir, "Skip-gram_dim100_win5_neg5.model")
    
    run_semantic_analysis(cbow_model, "CBOW (dim100, win5, neg5)")
    run_semantic_analysis(sg_model, "Skip-gram (dim100, win5, neg5)")

if __name__ == "__main__":
    main()
