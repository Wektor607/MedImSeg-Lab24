import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, DistributedSampler

from utils import SamplingStrategy, ActualSequentialSampler

class CLUESampling(SamplingStrategy):
    """
    Implements CLUE: Clustering via Uncertainty-weighted Embeddings for segmentation tasks.
    """
    def __init__(self, dset, train_idx, model, device, args, batch_size, balanced=False):
        super(CLUESampling, self).__init__(dset, train_idx, model, device, args)
        self.dset = dset
        self.train_idx = train_idx
        self.model = model
        self.device = device
        self.args = args
        self.batch_size = batch_size  # Explicitly use batch_size as a parameter
        self.cluster_type = args.cluster_type
        self.T = args.clue_softmax_t

        # [NEW] Retrieve new parameters from args
        self.use_uncertainty = args.use_uncertainty
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.target_size = args.target_size  # Newly introduced parameter

        # Attributes to store embeddings and cluster centers
        self.embeddings = None
        self.cluster_centers = None

    def get_embedding(self, model, loader, device, args, with_emb=False):
        self.model.eval()
        model = model.to(self.device)

        embedding_pen = []
        embedding = []
        self.image_to_embedding_idx = []

        avg_pool = torch.nn.AvgPool2d(kernel_size=(self.kernel_size, self.kernel_size),
                                      stride=self.stride)

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(loader)):
                data = data.to(device)

                if with_emb:
                    e1, e2 = model(data, with_emb=True)
                
                while e1.shape[2] * e1.shape[3] > self.target_size:
                    e1 = avg_pool(e1)
                while e2.shape[2] * e2.shape[3] > self.target_size:
                    e2 = avg_pool(e2)

                e1 = e1.permute(0, 2, 3, 1).reshape(e1.shape[0], -1)
                e2 = e2.permute(0, 2, 3, 1).reshape(e2.shape[0], -1)
                embedding_pen.append(e2.cpu())
                embedding.append(e1.cpu())

                start_idx = batch_idx * data.size(0)
                end_idx = start_idx + data.size(0)
                self.image_to_embedding_idx.extend(range(start_idx, end_idx))

        self.image_to_embedding_idx = np.array(self.image_to_embedding_idx)
        embedding_pen = torch.cat(embedding_pen, dim=0)
        embedding = torch.cat(embedding, dim=0)
        return embedding, embedding_pen

    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        if self.args.paral:
            train_sampler = DistributedSampler(ActualSequentialSampler(self.train_idx[idxs_unlabeled]))
        else:
            train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])

        data_loader = DataLoader(
            self.dset,
            sampler=train_sampler,
            num_workers=4,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.custom_collate
        )

        tgt_emb, tgt_pen_emb = self.get_embedding(self.model, data_loader, self.device, self.args, with_emb=True)
        self.embeddings = tgt_pen_emb.cpu().numpy()

        if self.use_uncertainty:
            tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
            tgt_scores += 1e-8
            sample_weights = (-(tgt_scores * torch.log(tgt_scores)).sum(1)).cpu().numpy()
        else:
            sample_weights = np.ones(self.embeddings.shape[0])

        km = KMeans(n)
        km.fit(self.embeddings, sample_weight=sample_weights)

        self.cluster_centers = km.cluster_centers_  # Save cluster centers

        indices = np.arange(self.embeddings.shape[0])
        q_idxs = []
        used_points = set()

        for centroid in km.cluster_centers_:
            distances = np.linalg.norm(self.embeddings[indices] - centroid, axis=1)
            sorted_indices = np.argsort(distances)

            for min_dist_idx in sorted_indices:
                min_index = indices[min_dist_idx]
                if min_index not in used_points:
                    q_idxs.append(min_index)
                    used_points.add(min_index)
                    indices = np.delete(indices, min_dist_idx)
                    break

        image_idxs = self.image_to_embedding_idx[q_idxs]
        image_idxs = list(set(image_idxs))
        return idxs_unlabeled[image_idxs]

    def get_cluster_centers(self):
        if self.cluster_centers is None:
            raise ValueError("Cluster centers have not been computed. Call query() first.")
        return self.cluster_centers

    def get_embeddings(self):
        if self.embeddings is None:
            raise ValueError("Embeddings have not been computed. Call query() first.")
        return self.embeddings
