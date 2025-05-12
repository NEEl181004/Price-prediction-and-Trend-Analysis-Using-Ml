import psycopg2
import pandas as pd
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def compute_cosine_similarity(product_titles, n_neighbors=5):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(product_titles)
        tfidf_matrix_sparse = csr_matrix(tfidf_matrix)  # Sparse format

        # Reduce dimensionality using SVD (ensure valid number of components)
        n_components = min(100, tfidf_matrix.shape[1] - 1)
        if n_components < 1:
            n_components = 1

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix_sparse)

        # Use NearestNeighbors to compute similarity without large memory usage
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(tfidf_reduced)

        # Find nearest neighbors (excluding self-matching)
        distances, indices = nn.kneighbors(tfidf_reduced)

        return distances, indices

    # Connect to PostgreSQL using SQLAlchemy
def connect_db():
        engine = create_engine("postgresql+psycopg2://miniproject:31998369@localhost:6543/price_comparison")
        return engine

    # Fetch product data from athe database
def fetch_product_data():
        engine = connect_db()
        query = "SELECT product_name, price, platform, timestamp FROM product_prices3"
        df = pd.read_sql(query, engine)
        return df

    # Fuzzy Matching: Find best match for a query (using product title)
def fuzzy_match(query, product_titles):
        best_match = process.extractOne(query, product_titles)
        return best_match

    # Prepare product titles for clustering (using TF-IDF)
def cluster_products(product_titles, n_clusters=5):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(product_titles)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(tfidf_matrix)
        return kmeans.labels_

    # Visualize clusters with t-SNE
def visualize_clusters(product_titles, labels):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(product_titles)
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(tfidf_matrix.toarray())
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
        plt.title("Product Clusters")
        plt.show()

    # Main execution
if __name__ == "__main__":
        product_data = fetch_product_data()
        product_titles = product_data['product_name'].tolist()
        
        # Fuzzy Matching Example
        query = "HP Spectre x360"
        best_match = fuzzy_match(query, product_titles)
        print(f"Best match for '{query}': {best_match}")
        
        # Cosine Similarity Example
        cosine_sim = compute_cosine_similarity(product_titles)
        print("Cosine Similarity Matrix:")
        print(cosine_sim)
        
        # KMeans Clustering Example
        labels = cluster_products(product_titles, n_clusters=5)
        print(f"Cluster Labels: {labels}")
        
        # Visualize Clusters
        visualize_clusters(product_titles, labels)
