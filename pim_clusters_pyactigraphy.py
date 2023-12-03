# from clustering.base_clustering import BaseClustering
from clustering import BaseClustering
import matplotlib.pyplot as plt

fpath = "test_sample_atr.txt"
kmeans_clustering = BaseClustering(fpath, 5, "PIM", "cluster")
data_frame_kmeans = kmeans_clustering.kmeans_light_data('LIGHT')

