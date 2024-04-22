import os.path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
import seaborn as sns
import pandas as pd
import itertools
from tqdm import tqdm

class GCDBSCAN:
    def __init__(self, properMotionData, clusterName, extension):
        self.clusterName = clusterName
        self.extension = extension
        self.data = properMotionData
        self.clustering_data = pd.concat([properMotionData["pmra"], properMotionData["pmdec"]], axis="columns")

    def PMPlots(self):
        # Plots proper motion plots for each cluster from Gaia matches
        # fig, ax = plt.subplots()
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.95, bottom=0.1, top=0.95,
                              wspace=0.1, hspace=0.1)
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        ax.scatter(self.data['pmra'], self.data['pmdec'], s=0.1, marker="x", color="k")
        # ax.set_title("Proper Motions")
        ax.set_xlabel("pmra [mas yr$^{-1}$]")
        ax.set_ylabel("pmdec [mas yr$^{-1}$]")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.text(16,20, f"{self.clusterName}", fontsize=16)
        ax.text(16,18, "Proper Motions", fontsize=10)

        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # now determine nice limits by hand:
        binwidth = 0.4
        xymax = max(np.max(np.abs(self.data['pmra'])), np.max(np.abs(self.data['pmdec'])))
        lim = (int(xymax / binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(self.data['pmra'], bins=bins, color="k")
        ax_histy.hist(self.data['pmdec'], bins=bins, orientation='horizontal', color="k")

        fig.savefig(f"PMPlots/{self.clusterName}_{self.extension}.pdf", dpi=300, bbox_inches="tight")

    def clusters(self):
        if os.path.exists(f"DBSCANParams/{self.clusterName}.json"):
            print("-->Saved best DBSCAN parameters found")
            with open(f"DBSCANParams/{self.clusterName}.json", "r") as g:
                DBSCANParams = json.load(g)
                self.best_epsilon = DBSCANParams['best_epsilon']
                self.best_min_samples = int(DBSCANParams['best_min_samples'])
                self.best_labels = np.asarray(DBSCANParams['best_labels']).astype(int)
                self.best_score = DBSCANParams['best_score']

        else:
            print("-->Saved best DBSCAN parameters not found")
            print("---->Starting DBSCAN parameter optimization")
            # This method uses DBSCAN to further narrow down cluster members
            epsilons = np.linspace(0.1, 1, 15)  # A range of likely/acceptable epsilon values to try
            minSamples = np.arange(2, 10, step=1)  # A range of likely/acceptable min_pts values to try
            combinations = list(itertools.product(epsilons, minSamples))  # Using itertools to create an array of the two

            scores = []
            all_labels_list = []

            for i, (eps, min_samples) in tqdm(enumerate(combinations)):
                dbscan_cluster_model = DBSCAN(eps=eps, min_samples=min_samples).fit(self.clustering_data)
                labels = dbscan_cluster_model.labels_
                labels_set = set(labels)
                num_clusters = len(labels_set)

                if -1 in labels_set:  # Remove the "noise cluster" as one of the possible clusters for finding best score
                    num_clusters -= 1

                if num_clusters > 10:  # *Arbitrary* upper limit on what is a good amount of clusters, open to change
                    scores.append(-10)
                    all_labels_list.append("bad")
                    continue

                scores.append(ss(self.clustering_data, labels))
                all_labels_list.append(labels)

            best_index = np.argmax(scores)  # Find the index of the combination with the highest score
            best_parameters = combinations[best_index]  # Get the best eps and min_pts values
            best_labels = all_labels_list[best_index]  # Get the labels for this run of DBSCAN
            best_score = scores[best_index]  # Get best score (for analytical purposes)

            self.best_epsilon = best_parameters[0]
            self.best_min_samples = best_parameters[1]
            self.best_labels = best_labels
            self.best_score = best_score

            print(f"---->Best DBSCAN parameters found to be eps={self.best_epsilon}, min_pts={self.best_min_samples}, "
                  f"with a silhouette score of {self.best_score}/1")

            with open(f"DBSCANParams/{self.clusterName}.json", "w") as f:
                DBSCANParams = {'best_epsilon': self.best_epsilon, 'best_min_samples': float(self.best_min_samples),
                                'best_score': self.best_score,
                                'best_labels': list(np.asarray(self.best_labels).astype(float))}
                json.dump(DBSCANParams, f)
                print("---->Best DBSCAN parameters saved")

        self.clustering_data["labels"] = self.best_labels

    def clusterPlots(self):
        # Plot the "clustered" proper motions for each cluster
        plt.figure(figsize=(8, 6))
        p = sns.scatterplot(data=self.clustering_data, x="pmra", y="pmdec", size=0.01, marker="+",
                            hue=self.clustering_data["labels"], legend="brief", palette="deep")
        sns.move_legend(p, "upper center", bbox_to_anchor=(0.15, 0.42), fontsize="x-small", ncol=3,
                        title='Clusters')
        plt.title("Proper Motion Clustering")
        plt.xlabel("pmra [mas yr$^{-1}$]")
        plt.ylabel("pmdec [mas yr$^{-1}$]")
        plt.xlim(-20, 15)
        plt.ylim(-15, 15)
        plt.savefig(f"DBSCANPlots/{self.clusterName}_{self.extension}.pdf", bbox_inches="tight", dpi=300)

    def indexOrganizer(self, cond):
        # Set the index of oids (from Gaia) to match the Sigel index (from .phot files)
        indSiegel = self.data[f"{self.clusterName.lower()}_oid"] - 1
        indSiegel = np.asarray(indSiegel)

        # The step below makes a (reasonable) assumption that the cluster members all belong to DBSCAN cluster 0
        indLab = np.where(self.clustering_data["labels"] == 0)[0]
        notInClust = np.where(self.clustering_data["labels"] != 0)[0]
        indSiegelDB = indSiegel[indLab]
        indAll = np.intersect1d(indSiegelDB, cond)  # The cond here is the initial clean up condition (chi, sharp)
        return indSiegel, indAll, notInClust
