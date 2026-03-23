import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

def plot_embeddings(model_path: str, title: str, output_file: str):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"Loading {title}...")
    model = Word2Vec.load(model_path)
    
    # Select subset of words for plotting (e.g. top 300 frequent words + specific queries)
    target_words = ["research", "student", "phd", "exam", "course", "faculty", "btech", "mtech", "ug", "pg", "institute", "department", "academic"]
    words_to_plot = target_words.copy()
    
    # Add top frequent words
    for word, _ in model.wv.most_similar("research", topn=200):
        if word not in words_to_plot:
            words_to_plot.append(word)
            
    # Filter words actually in the vocab
    words_to_plot = [w for w in words_to_plot if w in model.wv.key_to_index]
    
    # Get vectors
    word_vectors = [model.wv[w] for w in words_to_plot]
    
    # Dimensionality Reduction
    print(f"Projecting {len(words_to_plot)} words using t-SNE...")
    # PCA to 50d first as recommended for t-SNE
    pca = PCA(n_components=min(50, len(words_to_plot)))
    pca_result = pca.fit_transform(word_vectors)
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(pca_result)
    
    # Plotting
    plt.figure(figsize=(14, 10))
    for i, word in enumerate(words_to_plot):
        x, y = tsne_result[i]
        
        # Highlight target words
        if word in target_words:
            plt.scatter(x, y, color='red', s=50, edgecolors='k')
            plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=12, fontweight='bold', color='darkred')
        else:
            plt.scatter(x, y, color='steelblue', s=20, alpha=0.5)
            # Only annotate a small random subset of context words to avoid clutter
            if i % 10 == 0:
                plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=8, alpha=0.7)
                
    plt.title(f"t-SNE Projection: {title}")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()

def main():
    models_dir = "output/models"
    plots_dir = "output/plots"
    
    cbow_model = os.path.join(models_dir, "CBOW_dim100_win5_neg5.model")
    sg_model = os.path.join(models_dir, "Skip-gram_dim100_win5_neg5.model")
    
    plot_embeddings(cbow_model, "CBOW (dim100, win5, neg5)", os.path.join(plots_dir, "cbow_tsne.png"))
    plot_embeddings(sg_model, "Skip-gram (dim100, win5, neg5)", os.path.join(plots_dir, "skipgram_tsne.png"))
    
    print("\nVisualization Step Complete.")
    print("Interpretation: CBOW typically generates tighter clusters for functionally equivalent words ")
    print("whereas Skip-gram is better at capturing fine-grained semantic properties and rare word relations.")

if __name__ == "__main__":
    main()
