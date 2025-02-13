import argparse
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from mario_scenes.load_data import load_annotation_data, load_reduced_data
import pickle
import os
import os.path as op

def generate_clusters(list_n_clusters=[10]):
    X = load_annotation_data()
    hier = linkage(X, method='ward', metric='euclidean')  # scipy's hierarchical clustering
    res = dendrogram(hier, labels=X.index, get_leaves=True)  # Generate a dendrogram from the hierarchy
    order = res.get('leaves')  # Extract the order on papers from the dendrogram
    part = {}
    for ind, n_clusters in enumerate(list_n_clusters):
        # Cut the hierarchy and turn the parcellation into a dataframe
        clusters = np.squeeze(cut_tree(hier, n_clusters=n_clusters))
        part[ind] = {
            'index': clusters,
            'n_clusters': n_clusters,
            'summary': summary_clusters(clusters)
        }
    return part

def summary_clusters(part, threshold=0.05):
    X = load_annotation_data()
    X['clusters'] = part
    all_cluster = X.groupby('clusters').mean()
    summary = {}
    for index, row in all_cluster.iterrows():
        # Filter, sort in descending order
        summary[index] = {
            'n_scenes': len(X[X['clusters'] == index]),
            'labels': row[row > threshold].sort_values(ascending=False),
            'homogeneity': row
        }
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hierarchical clusters and save results as a pickle file.")
    parser.add_argument("--n_clusters", type=int, nargs='+', default=[10], help="List of numbers of clusters to generate.")

    args = parser.parse_args()

    # Generate clusters
    clusters = generate_clusters(args.n_clusters)

    # Create output directory and save results
    BASE_DIR = op.dirname(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))
    OUTPUT_DIR = op.join(BASE_DIR, 'outputs', 'cluster_scenes')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = op.join(OUTPUT_DIR, "hierarchical_clusters.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(clusters, f)

    print(f"Clustering completed. Results saved to {output_file}")
