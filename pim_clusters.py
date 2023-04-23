# from clustering.base_clustering import BaseClustering
from clustering import BaseClustering

fpath = "test_sample_alu.csv"
kmeans_clustering = BaseClustering(fpath, 5, "PIM", "cluster")
kmeans_clustering.plot_kmeans()