from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def train_kmeans(df, features, n_clusters):
    print(f"Training KMeans with {n_clusters} clusters on features {features}")
    kmodel = KMeans(n_clusters=n_clusters).fit(df[features])
    df['Cluster'] = kmodel.labels_
    return kmodel, df

def evaluate_kmeans(df, features, k_range):
    WCSS = []
    silhouette_scores = []
    for k in k_range:
        print(f"Evaluating KMeans with {k} clusters")
        kmodel = KMeans(n_clusters=k).fit(df[features])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        sil_score = silhouette_score(df[features], kmodel.labels_)
        silhouette_scores.append(sil_score)
    return WCSS, silhouette_scores
