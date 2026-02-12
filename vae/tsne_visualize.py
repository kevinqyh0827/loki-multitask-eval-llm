import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization of latent clusters')
    parser.add_argument('--cluster_dir', type=str, required=True, help='Path to cluster directory (e.g., vae/latent_cluster20_new_webdataset)')
    parser.add_argument('--n_clusters', type=int, default=20, help='Number of clusters')
    parser.add_argument('--latent_path', type=str, required=True, help='Path to train_z_mu.npy')
    parser.add_argument('--label_path', type=str, required=True, help='Path to cluster labels JSON')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples for t-SNE (subsampled for speed)')
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load latent embeddings
    print("Loading latent embeddings...")
    latents = np.load(args.latent_path)  # (N, 11, 32)
    N = latents.shape[0]
    latents_flat = latents.reshape(N, -1)  # (N, 352)
    print(f"Latents shape: {latents.shape} -> flattened: {latents_flat.shape}")

    # Load cluster labels
    print("Loading cluster labels...")
    with open(args.label_path, 'r') as f:
        cluster_labels_dict = json.load(f)

    # The saved train_z_mu.npy contains the training split (90% of data).
    # cluster_labels.json maps keys to labels for all 100k samples.
    # We use the first N entries from the label dict to match the training embeddings.
    all_labels = list(cluster_labels_dict.values())
    labels = np.array(all_labels[:N])
    print(f"Labels shape: {labels.shape}, unique clusters: {np.unique(labels)}")

    # Subsample for t-SNE speed
    np.random.seed(args.seed)
    n_samples = min(args.n_samples, N)
    indices = np.random.choice(N, size=n_samples, replace=False)
    latents_sub = latents_flat[indices]
    labels_sub = labels[indices]
    print(f"Subsampled {n_samples} points for t-SNE")

    # Run t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed, n_iter=1000)
    embeddings_2d = tsne.fit_transform(latents_sub)
    print("t-SNE complete.")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.get_cmap('tab20', args.n_clusters)

    for i in range(args.n_clusters):
        mask = labels_sub == i
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[cmap(i)], label=f'Cluster {i}', s=5, alpha=0.6)

    ax.set_title(f't-SNE Visualization of {args.n_clusters} Latent Clusters (n={n_samples})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(markerscale=3, fontsize=8, loc='best', ncol=2)

    save_path = f"{args.cluster_dir}/tsne_{args.n_clusters}clusters_{n_samples}samples.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
