import scanpy as sc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.classifier(encoded)
        return decoded, logits

def train_autoencoder(X_tensor, cell_types_tensor, latent_dim):
    # Define hyperparameters
    input_dim = X_tensor.shape[1]  # Number of features
    num_classes = len(torch.unique(cell_types_tensor))  # Number of unique cell types

    # Initialize the autoencoder model
    autoencoder = Autoencoder(input_dim, latent_dim, num_classes)

    # Define loss functions and optimizer
    criterion_recon = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, cell_types_tensor, test_size=0.1, random_state=42)

    # Train the autoencoder
    num_epochs = 100
    batch_size = 32
    prev_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        autoencoder.train()
        train_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i + batch_size]
            batch_labels = Y_train[i:i + batch_size]
            optimizer.zero_grad()
            recon_batch, logits = autoencoder(batch)
            recon_loss = criterion_recon(recon_batch, batch)
            cls_loss = criterion_cls(logits, batch_labels)
            loss = recon_loss + cls_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(X_train)

        # Validation phase
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch = X_val[i:i + batch_size]
                batch_labels = Y_val[i:i + batch_size]
                recon_batch, logits = autoencoder(batch)
                recon_loss = criterion_recon(recon_batch, batch)
                cls_loss = criterion_cls(logits, batch_labels)
                loss = recon_loss + cls_loss
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(X_val)

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Add stopping criterion based on validation loss
        if prev_val_loss <= val_loss or val_loss < 0.1:
            break
        prev_val_loss = val_loss

    return autoencoder

def compute_cluster_distances(embeddings, cell_types_numeric):
    # Compute pairwise distances between embeddings
    pairwise_dist = pairwise_distances(embeddings)

    # Initialize variables to store inter-cluster and intra-cluster distances
    inter_cluster_dist = 0.0
    intra_cluster_dist = 0.0

    # Iterate over unique clusters
    unique_clusters = np.unique(cell_types_numeric)
    for cluster in unique_clusters:
        # Get indices of points belonging to the current cluster
        cluster_indices = np.where(cell_types_numeric == cluster)[0]

        # Calculate intra-cluster distance (average distance within the cluster)
        intra_cluster_dist += np.mean(pairwise_dist[cluster_indices[:, None], cluster_indices])

        # Calculate inter-cluster distance (average distance to points in other clusters)
        other_cluster_indices = np.where(cell_types_numeric != cluster)[0]
        inter_cluster_dist += np.mean(pairwise_dist[cluster_indices[:, None], other_cluster_indices])

    # Calculate average inter-cluster and intra-cluster distances
    inter_cluster_dist /= len(unique_clusters)
    intra_cluster_dist /= len(unique_clusters)

    return inter_cluster_dist, intra_cluster_dist

def compute_clustering_evaluation_metrics(embeddings, cell_types_numeric):
    # Silhouette Score
    silhouette = silhouette_score(embeddings, cell_types_numeric, random_state=42)

    # Classification Accuracy
    X_train, X_val, Y_train, Y_val = train_test_split(embeddings, cell_types_numeric, test_size=0.1, random_state=42)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, Y_train)
    accuracy = classifier.score(X_val, Y_val)

    inter_cluster_dist, intra_cluster_dist = compute_cluster_distances(embeddings, cell_types_numeric)

    return silhouette, accuracy, inter_cluster_dist, intra_cluster_dist

def main():
    # Load H5AD file
    adata = sc.read_h5ad('stereo_seq_olfactory_bulb_bin140_annotation.h5ad')

    # Extract gene expression matrix
    X = adata.X

    # Extract cell types
    cell_types = adata.obs['cell type']

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit LabelEncoder and transform cell types into numerical labels
    cell_types_numeric = label_encoder.fit_transform(cell_types)

    # Convert variables to PyTorch tensors
    X_tensor = torch.nn.functional.normalize(torch.tensor(X, dtype=torch.float32))
    cell_types_tensor = torch.tensor(cell_types_numeric, dtype=torch.long)
    latent_dim = 128  # Adjust based on experiments

    # Train autoencoder
    autoencoder = train_autoencoder(X_tensor, cell_types_tensor, latent_dim)

    # Get autoencoder embeddings from the encoder
    with torch.no_grad():
        encoded_data = autoencoder.encoder(X_tensor).numpy()

    # PCA embeddings

    # Project autoencoder embeddings onto 2 dimensions using PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(encoded_data)

    silhouette, accuracy, inter_cluster_dist, intra_cluster_dist = \
        compute_clustering_evaluation_metrics(embeddings_pca, cell_types_numeric)

    print("--------------------------------------------------")
    print("PCA evaluation metrics:")
    print("--------------------------------------------------")
    print("Silhouette Score:", silhouette)
    print("Classification Accuracy:", accuracy)
    print("Inter-Cluster Distance:", inter_cluster_dist)
    print("Intra-Cluster Distance:", intra_cluster_dist)

    # Plot PCA embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=cell_types_numeric, cmap='viridis', s=10)
    plt.title('PCA Embeddings')
    plt.colorbar(label='Cell Type')
    plt.show()

    # UMAP embeddings

    # Project autoencoder embeddings onto 2 dimensions using UMAP
    umap = UMAP(n_components=2, random_state=42)
    embeddings_umap = umap.fit_transform(encoded_data)

    silhouette, accuracy, inter_cluster_dist, intra_cluster_dist = \
        compute_clustering_evaluation_metrics(embeddings_umap, cell_types_numeric)

    print("--------------------------------------------------")
    print("UMAP evaluation metrics:")
    print("--------------------------------------------------")
    print("Silhouette Score:", silhouette)
    print("Classification Accuracy:", accuracy)
    print("Inter-Cluster Distance:", inter_cluster_dist)
    print("Intra-Cluster Distance:", intra_cluster_dist)

    # Plot UMAP embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=cell_types_numeric, cmap='viridis', s=10)
    plt.title('UMAP Embeddings')
    plt.colorbar(label='Cell Type')
    plt.show()

if __name__ == "__main__":
    main()
