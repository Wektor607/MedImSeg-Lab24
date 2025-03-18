import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from scipy.sparse import lil_matrix  # Import for sparse optimization
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
        self.batch_size = batch_size  
        self.cluster_type = getattr(args, 'cluster_type', 'centroids')
        self.T = getattr(args, 'clue_softmax_t', 1.0)

        # Retrieve new parameters from args with default values
        self.use_uncertainty = getattr(args, 'use_uncertainty', False)
        self.target_size = getattr(args, 'target_size', 1024)
        self.isomap_n_components = getattr(args, 'isomap_n_components', 2)
        self.isomap_n_neighbors = max(10, getattr(args, 'isomap_n_neighbors', 5))  # Ensure n_neighbors >= 10
        self.km = None  
        self.last_embeddings = None  

    def get_embedding(self, model, loader, device, args, with_emb=False):
        self.model.eval()
        model = model.to(self.device)

        embedding_pen = []
        embedding = []
        self.image_to_embedding_idx = []

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(loader)):
                print(f"[DEBUG] Processing batch {batch_idx+1}/{len(loader)}")
                data = data.to(device)

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

        # Apply Isomap dimensionality reduction
        print("[DEBUG] Applying Isomap transformation...")
        isomap = Isomap(n_neighbors=min(10, self.isomap_n_neighbors), n_components=self.isomap_n_components)

        # Convert to lil_matrix for efficient sparse modifications
        embedding_pen_sparse = lil_matrix(embedding_pen)
        embedding_pen_reduced = isomap.fit_transform(embedding_pen_sparse)
        print("[DEBUG] Isomap transformation complete.")
        
        self.last_embeddings = embedding_pen_reduced  
        return embedding, embedding_pen_reduced

    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        if getattr(self.args, 'paral', False):
            train_sampler = DistributedSampler(ActualSequentialSampler(self.train_idx[idxs_unlabeled]))
        else:
            train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])

        data_loader = DataLoader(
            self.dset,
            sampler=train_sampler,
            num_workers=0,  
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
        self.km = KMeans(n, random_state=42)  # Added `random_state` for reproducibility
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

        # Remove duplicates and ensure unique indices
        image_idxs = list(set(self.image_to_embedding_idx[q_idxs]))
        return idxs_unlabeled[image_idxs]

    def get_cluster_centers(self):
        """Returns cluster centers from the last KMeans clustering operation."""
        if self.km is not None:
            return self.km.cluster_centers_
        else:
            raise AttributeError("KMeans has not been run yet. Call query() first.")

    def get_embeddings(self):
        """Returns the last computed embeddings."""
        if self.last_embeddings is not None:
            return self.last_embeddings
        else:
            raise AttributeError("Embeddings have not been computed yet. Call get_embedding() first.")
