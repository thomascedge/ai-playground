import numpy as np
import matplotlib.pyplot as plt
from typing import Any

def euclidian_distance(self, x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)

class KMeans():
    def __init__(self, k: int = 5, max_iters: int = 100, plot_steps: bool = False):
        """
        An unsupervised learning technique, finds a point within a dataset
        called a centroids, and finds the nearest values closest to it, 
        clustering the data around it. Each cluster has a tight nit
        relationship with regards to their attributes. The algorithm 
        categorizes data into k-groups using the Euclidean distance to find the
        distance between data point and centroid.

        K Means algorithm works as such:
        1: Choose initial centroids. These points are randomly chosen and
            represent the initial cluster centers
        2: Assign points to nearest centroids. Each point is assigned the
            nearest centroids, forming clusters
        3: Update centroids. Centroids are recalculated as the mean of the
            points in each cluster
        4: Repeat step 2-3 until convergence. Centroids stabalize and do not
            move any further, giving final cluster groups.
        """
        self.K = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters  = [[] for _ in range(self.K)] # list of sample indices per cluster
        self.centroids = [] # the centers (mean vector) for each cluster

    def predict(self, X: np.array):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimization
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters())

    def _get_cluster_labels(self, clusters):
        # each sample will get the lable of the cluseter it was assigned
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        
        return labels


    def _create_clusters(self, centroids) -> list[list]:
        # assign samples to closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # determine distance of the current sample to each centroid
        distances = [euclidian_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # assign mean value of the clusters to the centroids
        centroids = np.zeros(self.K, self.n_features)
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # check distances between old and new centroids for all centroids
        # returns zero when distances are similar between old and new centroids
        distances = [euclidian_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self) -> Any:
        fig, ax = plt.subplots(figsize=(12, 8))

        for _, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        
        for point in self.centroid:
            ax.scatter(*point, marker='x', color='black', linewidth=2)

        plt.show()


# for testing
if __name__ == "__main__":
    import pandas as pd
    from loguru import logger

    houses = pd.read_csv('austinHousingData.csv')
    properties = ['zipcode', 
                'homeType', 
                'latestPrice',
                'propertyTaxRate', 
                'numOfBedrooms', 
                'avgSchoolDistance', 
                'avgSchoolRating', 
                'MedianStudentsPerTeacher']
    
    data = houses[properties].copy()

    # Map strings to integers, removing text values
    # city = {value: key for key, value in enumerate(data['city'].unique())}
    zipcode = {value: key for key, value in enumerate(data['zipcode'].unique())}
    home_type = {value: key for key, value in enumerate(data['homeType'].unique())}

    # data['city'] = data['city'].map(city)
    data['zipcode'] = data['zipcode'].map(zipcode)
    data['homeType'] = data['homeType'].map(home_type)

    # data scaling to 1-10
    data = ((data - data.min()) / (data.max() - data.min())) * 9 + 1

    logger.info(f'Data shape: f{data.shape}.')

    k = KMeans(k=2)
    prediction = k.predict(data)
    k.plot()
