import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F

class CC:
    def __init__(self, num_known_classes, num_unknown_clusters, feature_dim, use_cuda=False):
        self.num_known_classes = num_known_classes
        self.num_unknown_clusters = num_unknown_clusters
        self.feature_dim = feature_dim
        self.use_cuda = use_cuda
        self.source_centroids = torch.zeros(num_known_classes, feature_dim)
        if use_cuda:
            self.source_centroids = self.source_centroids.cuda()

    def update_source_centroids(self, features, labels):
        for i in range(self.num_known_classes):
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                self.source_centroids[i] = features[class_mask].mean(dim=0)

    def ncc_clustering(self, target_features, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters)
        target_clusters = kmeans.fit_predict(target_features.cpu().detach().numpy())
        target_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
        if self.use_cuda:
            target_centroids = target_centroids.cuda()
        return target_clusters, target_centroids

    def loss_ncc(self, rep_s, y_s, rep_t, y_t_pseudo, mask_known, threshold=0.6):
        self.update_source_centroids(rep_s, y_s)

        mask_unknown = ~mask_known

        target_clusters, target_centroids = self.ncc_clustering(rep_t[mask_unknown], self.num_unknown_clusters)
        print('target_cemtroids.shape: ',target_centroids.shape,'source_centroids.shape: ',self.source_centroids.shape)
        loss_inter, _ = self.compute_inter_loss(self.source_centroids, target_centroids)

        if mask_known.sum() > 0:
            loss_intra = self.compute_intra_loss(rep_t[mask_known], y_t_pseudo[mask_known], self.source_centroids)
        else:
            loss_intra = torch.tensor(0.0)

        return loss_intra + loss_inter, mask_known, mask_unknown

    def compute_intra_loss(self, features, labels, centroids):
        loss_intra = 0.0
        for i in range(self.num_known_classes):
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                loss_intra += torch.sum((features[class_mask] - centroids[i]) ** 2)
        return loss_intra / len(features) if len(features) > 0 else torch.tensor(0.0)

    def compute_inter_loss(self, centroids_s, centroids_t):
        loss_inter = 0.0
        for i in range(len(centroids_s)):
            for j in range(len(centroids_t)):
                if i != j:
                    cos_sim = F.cosine_similarity(centroids_s[i].unsqueeze(0), centroids_t[j].unsqueeze(0), dim=1)
                    dist = 1 - cos_sim
                    loss_inter += dist
        return loss_inter / (len(centroids_s) * len(centroids_t)), centroids_t



