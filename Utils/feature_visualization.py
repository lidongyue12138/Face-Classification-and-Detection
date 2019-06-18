# -*- coding: utf-8 -*-
import numpy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys, os
sys.path.append("./")
from Utils.DataLoader import DataLoader

class FeatureAnalyzer:
    def __init__(self):
        pass

    def pca_analyze(self, x, n_dims=2):
        pca = PCA(n_components=n_dims)
        newX = pca.fit_transform(x)
        return newX

    def tsne_analyze(self, x, n_dims=2):
        tsne = TSNE(n_components=n_dims, learning_rate=100)
        newX = tsne.fit_transform(x)
        return newX

    def img_plot(self, path, x, y=None):
        if (y == None):
            plt.scatter(x[:, 0], x[:, 1], c="b")
            plt.show()
        else:
            plt.scatter(x[:, 0], x[:, 1], c=y)
            plt.colorbar()
            plt.show()
        plt.savefig(path)


if __name__ == "__main__":
    Data = DataLoader()
    Data.load_pickle_dataset_new()

    featureAnalyzer = FeatureAnalyzer()
    path = "./pca.png"

    x = Data.train_data_hog[:10000]
    y = list(Data.train_label[:10000])

    # featureAnalyzer.img_plot(path, x, y)

    transformed_x = featureAnalyzer.pca_analyze(x)
    featureAnalyzer.img_plot(path, transformed_x, y)

    

