import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
# Data is one level up then into data/real/fashion-mnist
data_dir = os.path.abspath(os.path.join(current_dir, "..", "data", "real", "fashion-mnist"))
# Output is inside 'visual/outputs'
output_dir = os.path.join(current_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

x_path = os.path.join(data_dir, "fashion_mnist_X.csv")
y_path = os.path.join(data_dir, "fashion_mnist_y.csv")

# Labels for Fashion MNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def visualize():
    print("Loading data from CSV...")
    if not os.path.exists(x_path):
        print(f"Error: {x_path} not found!")
        print(f"Looked at: {x_path}")
        return

    # Read subset for visualization
    X = np.loadtxt(x_path, delimiter=",", max_rows=7000)
    y = np.loadtxt(y_path, delimiter=",", max_rows=7000, dtype=int)

    print(f"Processing visualization for {len(X)} samples...")

    # 1. PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 2. t-SNE
    print("Running t-SNE (this may take 1-2 minutes)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # PCA Plot
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=5, alpha=0.6)
    ax1.set_title("Fashion-MNIST: PCA Projection (2D)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    
    # t-SNE Plot
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=5, alpha=0.6)
    ax2.set_title("Fashion-MNIST: t-SNE Projection (2D)")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    # Legend
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes", loc="lower left", bbox_to_anchor=(1, 0))
    for i, text in enumerate(legend1.get_texts()):
        text.set_text(class_names[i])

    plt.tight_layout()
    output_plot = os.path.join(output_dir, "fashion_mnist_structure.png")
    plt.savefig(output_plot, dpi=300)
    print(f"Saved structure plot to: {output_plot}")

    # 4. Samples
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = X[i].reshape(28, 28)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.xlabel(class_names[y[i]])
    
    output_samples = os.path.join(output_dir, "fashion_mnist_samples.png")
    plt.savefig(output_samples)
    print(f"Saved samples plot to: {output_samples}")

if __name__ == "__main__":
    visualize()
