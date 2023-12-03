import pandas as pd
import matplotlib.pyplot as plt
import pyActigraphy
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
    
    def kmeans_light_data(self, light_channel):
        rawFile = pyActigraphy.io.read_raw_atr(self.file_path)
        data_frame = rawFile.light.get_channels([light_channel])
        kmeans = KMeans(n_clusters = self.n_clusters)
        kmeans.fit(data_frame[[light_channel]])
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        data_frame["cluster"] = labels
        return data_frame