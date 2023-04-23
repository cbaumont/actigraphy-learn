from clustering import BaseClustering

fpath = "test_sample_alu.csv"
kmeans_clustering = BaseClustering(fpath, 5, "LIGHT", "cluster")
# kmeans_clustering.plot_kmeans()

kmeans_clustering_2 = BaseClustering(fpath, 5, "LIGHT", "PIM")
kmeans_clustering_2.plot_kmeans()