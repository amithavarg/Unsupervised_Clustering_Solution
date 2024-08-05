import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from utils.data_utils import load_data, preprocess_data
from models.kmeans_model import train_kmeans, evaluate_kmeans

logging.basicConfig(filename='logs/model_logs.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    try:
        # Load Data
        df = load_data('data/mall_customers.csv')
        logging.info("Data loaded successfully")

        # Preprocess Data
        df = preprocess_data(df)
        logging.info("Data preprocessed successfully")

        # Initial Data Check
        print("Initial Data Check:")
        print(df.head())
        print(df.describe())
        print(df.shape)

        # Visualize Pairplot
        sns.pairplot(df[['Age', 'Annual_Income', 'Spending_Score']])
        plt.show()

        # Train KMeans Model
        kmodel, df = train_kmeans(df, ['Annual_Income', 'Spending_Score'], 5)
        logging.info("KMeans model trained successfully")
        print("Cluster Centers:", kmodel.cluster_centers_)
        print("Cluster Labels:", kmodel.labels_)
        print(df.head())
        print(df['Cluster'].value_counts())

        # Visualize Clusters
        sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, hue='Cluster', palette='colorblind')
        plt.show()

        # Evaluate KMeans Model
        k_range = range(3, 9)
        WCSS, silhouette_scores = evaluate_kmeans(df, ['Annual_Income', 'Spending_Score'], k_range)
        logging.info("KMeans model evaluated successfully")
        wss = pd.DataFrame({'cluster': k_range, 'WSS_Score': WCSS, 'Silhouette_Score': silhouette_scores})
        print(wss)

        # Elbow Plot
        wss.plot(x='cluster', y='WSS_Score')
        plt.xlabel('No. of clusters')
        plt.ylabel('WSS Score')
        plt.title('Elbow Plot')
        plt.show()

        # Silhouette Plot
        wss.plot(x='cluster', y='Silhouette_Score')
        plt.xlabel('No. of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Plot')
        plt.show()

        # Train and Evaluate Model on 'Age', 'Annual_Income', 'Spending_Score'
        silhouette_scores_age = evaluate_kmeans(df, ['Age', 'Annual_Income', 'Spending_Score'], k_range)
        print(silhouette_scores_age)
        
    except Exception as e:
        logging.error("Error occurred: " + str(e))
        print("An error occurred: ", e)

if __name__ == "__main__":
    main()
