import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, DistributedSampler
import umap.umap_ as umap  # Ensure correct UMAP import

from utils import SamplingStrategy, ActualSequentialSampler

class CLUESampling(SamplingStrategy):
    """
    Implements CLUE: Clustering via Uncertainty-weighted Embeddings for segmentation tasks.
    """
    def __init__(self, dset, train_idx, model, device, args, batch_size, balanced=False):
        super(CLUESampling, self).__init__(dset, train_idx, model, device, args)
        self.dset = dset
        self.train_idx = train_idx
        self.model = model.to(device)  # Ensure model is on the correct device
        self.device = device
        self.args = args
        self.batch_size = batch_size  # Explicitly use batch_size as a parameter
        self.cluster_type = args.cluster_type
        self.T = args.clue_softmax_t

        # Retrieve new parameters from args
        self.use_uncertainty = getattr(args, "use_uncertainty", False)
        self.target_size = getattr(args, "target_size", None)
        self.umap_n_components = getattr(args, "umap_n_components", 2)
        self.umap_n_neighbors = getattr(args, "umap_n_neighbors", 15)

    def get_cluster_centers(self):
        """
        Retrieve the computed cluster centers from KMeans clustering.
        """
        if hasattr(self, 'km'):
            return self.km.cluster_centers_
        else:
            raise ValueError("[ERROR] KMeans clustering has not been run yet.")

    def get_embedding(self, model, loader, device, args, with_emb=False):
        """
        Extracts embeddings from the model for clustering.
        """
        self.model.eval()
        model = model.to(self.device)  # Move model to correct device

        embedding_pen = []
        embedding = []
        self.image_to_embedding_idx = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):
                print(f"[DEBUG] Processing batch {batch_idx+1}/{len(loader)}")

                # Ensure correct data extraction
                data = batch[0].to(self.device) if isinstance(batch, (tuple, list)) else batch.to(self.device)

                if with_emb:
                    e1, e2 = model(data, with_emb=True)

                    # Flatten spatial dimensions
                    e1 = e1.permute(0, 2, 3, 1).reshape(e1.shape[0], -1)
                    e2 = e2.permute(0, 2, 3, 1).reshape(e2.shape[0], -1)

                    embedding_pen.append(e2.cpu())
                    embedding.append(e1.cpu())

                    # Save indices
                    start_idx = batch_idx * data.size(0)
                    end_idx = start_idx + data.size(0)
                    self.image_to_embedding_idx.extend(range(start_idx, end_idx))

        self.image_to_embedding_idx = np.array(self.image_to_embedding_idx)
        embedding_pen = torch.cat(embedding_pen, dim=0).numpy()
        embedding = torch.cat(embedding, dim=0).numpy()

        # Apply UMAP dimensionality reduction
        print("[DEBUG] Applying UMAP transformation...")
        n_components = min(self.umap_n_components, embedding_pen.shape[1])
        umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=self.umap_n_neighbors)
        embedding_pen_reduced = umap_reducer.fit_transform(embedding_pen)
        print("[DEBUG] UMAP transformation complete.")

        return embedding, embedding_pen_reduced

    def query(self, n):
        """
        Perform clustering to select the most uncertain samples.
        """
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        if self.args.paral:
            train_sampler = DistributedSampler(ActualSequentialSampler(self.train_idx[idxs_unlabeled]))
        else:
            train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])

        data_loader = DataLoader(
            self.dset,
            sampler=train_sampler,
            num_workers=0,  # Avoid deadlocks
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.custom_collate
        )

        # Getting embeddings
        print("[DEBUG] Getting embeddings...")
        tgt_emb, tgt_pen_emb = self.get_embedding(self.model, data_loader, self.device, self.args, with_emb=True)
        print("[DEBUG] Embeddings retrieved.")

        # Conditionally compute sample_weights
        if self.use_uncertainty:
            tgt_scores = nn.Softmax(dim=1)(torch.tensor(tgt_emb) / self.T)
            tgt_scores += 1e-8
            sample_weights = (-(tgt_scores * torch.log(tgt_scores)).sum(1)).cpu().numpy()
        else:
            sample_weights = np.ones(tgt_pen_emb.shape[0])

        print("[DEBUG] Performing KMeans clustering...")
        self.km = KMeans(n, n_init=10, random_state=42)  # Explicitly set n_init
        self.km.fit(tgt_pen_emb, sample_weight=sample_weights)
        print("[DEBUG] KMeans clustering complete.")

        indices = np.arange(tgt_pen_emb.shape[0])
        q_idxs = []
        used_points = set()

        for centroid in self.km.cluster_centers_:
            distances = np.linalg.norm(tgt_pen_emb[indices] - centroid, axis=1)
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
