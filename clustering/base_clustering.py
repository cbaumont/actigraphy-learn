import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class BaseClustering():
    def __init__(self, file_path, n_clusters, x_column, y_column):
        self.file_path = file_path
        self.n_clusters = n_clusters
        self.x_column = x_column
        self.y_column = y_column

    def plot_kmeans(self):
        data_frame = pd.read_csv(self.file_path, sep=";")
        kmeans = KMeans(n_clusters = self.n_clusters)
        kmeans.fit(data_frame[[self.x_column]])
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        data_frame["cluster"] = labels
        # Create a scatter plot of x and y columns, colored by labels
        plt.scatter(data_frame[self.x_column], data_frame[self.y_column], c=labels)
        # Plot the centroids as red stars
        plt.scatter(centroids[:, 0], [0] * len(centroids), marker="*", c="red", s=100)
        # Show the plot
        plt.show()
