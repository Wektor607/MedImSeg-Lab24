import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random 
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN

from torch.utils.data import DataLoader, DistributedSampler

from utils import SamplingStrategy, ActualSequentialSampler
import matplotlib.pyplot as plt
import seaborn as sns
#  python3 run_exp.py --train False --num_clusters 10 --clue_softmax_t 0.1 --adapt_num_epochs 100 --device cuda:1 --uncertainty Distance --kernel_size 5 --stride 2 --target_size 4096
class CLUESampling(SamplingStrategy):
    """
    Implements CLUE: Clustering via Uncertainty-weighted Embeddings for segmentation tasks.
    """
    def __init__(self, dset, train_idx, model, device, args, batch_size):
        super(CLUESampling, self).__init__(dset, train_idx, model, device, args)
        self.dset = dset
        self.train_idx = train_idx
        self.model = model
        self.device = device
        self.args = args
        self.batch_size = batch_size  # Explicitly use batch_size as a parameter
        self.T = args.clue_softmax_t

        # [NEW] Retrieve new parameters from args
        self.clustering = args.clustering
        self.uncertainty = args.uncertainty
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.target_size = args.target_size

        self.eps = args.eps
        self.min_samples = args.min_samples
        # target_size = 256 #16x16
        # target_size = 1024 #32x32
        # target_size = 4096 #64x64
        # target_size = 16384 #128x128

    def get_embedding(self, model, loader, device, args, with_emb=False):
        model = model.to(self.device)

        embedding_pen = []
        embedding = []
        self.image_to_embedding_idx = []

        # Use the parameters from the constructor
        avg_pool = torch.nn.AvgPool2d(kernel_size=(self.kernel_size, self.kernel_size),
                                      stride=self.stride)

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(loader)):
                data = data.to(device)

                if with_emb:
                    e1, e2 = model(data, with_emb=True)
                
                # AvgPooling repeatedly until the spatial area is <= target_size
                while e1.shape[2] * e1.shape[3] > self.target_size:
                    e1 = avg_pool(e1)
                while e2.shape[2] * e2.shape[3] > self.target_size:
                    e2 = avg_pool(e2)

                # [batch_size, h * w * num_classes]
                e1 = e1.permute(0, 2, 3, 1).reshape(e1.shape[0], -1)
                e2 = e2.permute(0, 2, 3, 1).reshape(e2.shape[0], -1)
                embedding_pen.append(e2.cpu())
                embedding.append(e1.cpu())

                # Save indices
                start_idx = batch_idx * data.size(0)
                end_idx = start_idx + data.size(0)
                self.image_to_embedding_idx.extend(range(start_idx, end_idx))

        self.image_to_embedding_idx = np.array(self.image_to_embedding_idx)
        embedding_pen = torch.cat(embedding_pen, dim=0)
        embedding = torch.cat(embedding, dim=0)
        return embedding, embedding_pen
    
    def compute_dice_score(self, pred, target, epsilon=1e-6):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + epsilon) / (union + epsilon)
        return dice

    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        print(idxs_unlabeled)
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

        # Getting embeddings
        self.model.eval()
        tgt_emb, tgt_pen_emb = self.get_embedding(self.model, data_loader, self.device, self.args, with_emb=True)
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()

        # TODO: Class with calculating uncertainty
        # Conditionally compute sample_weights
        if self.uncertainty == 'Entropy':
            tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
            tgt_scores += 1e-8
            sample_weights = (-(tgt_scores * torch.log(tgt_scores)).sum(1)).cpu().numpy()
        elif self.uncertainty == 'Distance':
            x_emb = tgt_emb[idxs_unlabeled].clone().detach()
            dists = torch.cdist(x_emb, tgt_emb)  # (B, N)
            dists += torch.randn_like(dists) * 1e-5
            sample_weights, _ = dists.min(dim=1)
        elif self.uncertainty == 'MutalInfo':
            self.model.train()  # MC Dropout
            probs_list = []
            T = 10
            for _ in range(T):
                with torch.no_grad():
                    logits, _ = self.get_embedding(self.model, data_loader, self.device, self.args, with_emb=True)  # (B, C)
                    probs  = nn.Softmax(dim=1)(logits)  # (B, C)
                    probs  += 1e-8
                probs_list.append(probs)
            print(probs_list)
            probs_stacked = torch.stack(probs_list, dim=0)  # (T, B, C)
            mean_probs = probs_stacked.mean(dim=0)  # (B, C)
            
            # H(E[p])
            ent_mean = -(mean_probs * mean_probs.log()).sum(dim=1)
            
            # E[H(p)]
            ent_stacked = -(probs_stacked * probs_stacked.log()).sum(dim=2)  # (T, B)
            ent_expected = ent_stacked.mean(dim=0)  # (B,)
            
            # MI = H(E[p]) - E[H(p])
            sample_weights = (ent_mean - ent_expected).abs()
            sample_weights = sample_weights.to(dtype=torch.float64)
            sample_weights += torch.randn_like(sample_weights) * 1e-5

            self.model.eval()
        elif self.uncertainty == 'Margin':
            probs = nn.Softmax(dim=1)(tgt_emb / self.T)   # (B, C)
            probs += 1e-8
            sorted_probs, _ = probs.sort(dim=1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            sample_weights = 1.0 - margin
            # sample_weights = margin / margin.max() 
        elif self.uncertainty == 'Uniform':
            sample_weights = np.ones(tgt_emb.shape[0])
        else:
            predictions = torch.argmax(tgt_emb, dim=1)

            targets = torch.stack([self.dset[idx]['target'][0] for idx in idxs_unlabeled])
            dice_scores = torch.tensor([self.compute_dice_score(predictions[i], targets[i]) for i in range(self.batch_size)])
            sample_weights = 1.0 - dice_scores

        # Normalize weights for visualization
        if self.uncertainty not in ['Entropy', 'Uniform']:
            sample_weights = sample_weights.cpu().numpy()
        

        if self.uncertainty == 'Margin':
            sample_weights = np.log1p(sample_weights) ** 2  # Сглаживание разности вероятностей
        # elif self.uncertainty == 'Distance':
        #     sample_weights = np.sqrt(sample_weights + 1e-5)  # Сглаживание пиков
        #     sample_weights = np.log1p(sample_weights)  # Дальнейшее выравнивание

        denom = np.max(sample_weights) - np.min(sample_weights)
        if denom > 1e-8:
            sample_weights = (sample_weights - np.min(sample_weights)) / denom
            print(np.unique(sample_weights))
        else:
            sample_weights = np.ones_like(sample_weights)

        # Save sample weights for analysis
        # np.save(f"weights_{self.uncertainty}.npy", sample_weights)

        # Visualization of sample weight distributions
        # plt.figure(figsize=(8, 5))
        # sns.histplot(sample_weights, bins=50, kde=True, alpha=0.6)
        # plt.xlabel("Weight Value", fontsize=25)
        # plt.ylabel("Density", fontsize=25)
        # plt.title(f"Distribution of Sample Weights ({self.uncertainty})", fontsize=25)
        # plt.grid(True)
        # plt.xticks(fontsize=15)
        # plt.xticks(fontsize=15)
        # plt.savefig(f"./plots/weights_distribution_{self.uncertainty}.png", dpi=300, bbox_inches='tight')

        # return 0
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(tgt_pen_emb)

        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5)
        plt.title("Projection of tgt_pen_emb")
        plt.savefig(f"./plots/tgt_pen_emb.png", dpi=300, bbox_inches='tight')
        from scipy.spatial.distance import pdist

        pairwise_dists = pdist(tgt_pen_emb, metric='euclidean')
        print(f"Mean pairwise distance: {pairwise_dists.mean()}")
        print(f"Median pairwise distance: {np.median(pairwise_dists)}")

        if self.clustering == 'KMeans':
            km = KMeans(n_clusters=n)
            km.fit(tgt_pen_emb, sample_weight=sample_weights)
            centroids = km.cluster_centers_
        elif self.clustering == 'MBKMeans':
            mbkm = MiniBatchKMeans(n_clusters=n, random_state=42)
            mbkm.fit(tgt_pen_emb, sample_weight=sample_weights)
            centroids = mbkm.cluster_centers_
        elif self.clustering == 'DBSCAN':
            epsi = self.eps
            unique_labels = set()
            while len(unique_labels) < n:
                dbscan = DBSCAN(eps=epsi, min_samples=self.min_samples)
                labels = dbscan.fit_predict(tgt_pen_emb, sample_weight=sample_weights)

                unique_labels = set(labels) - {-1}
                epsi += 10                
                print(unique_labels)
            
            while len(unique_labels) > n:
                unique_labels.remove(random.choice(list(unique_labels)))
            
            print(f"Method: {self.uncertainty}, Found clusters: {len(unique_labels)}")
            
            centroids = []
            for label in unique_labels:
                cluster_points = tgt_pen_emb[np.where(labels == label)]
                centroid_mean = cluster_points.mean(axis=0)
                print()
                closest_idx = np.argmin(np.linalg.norm(cluster_points - centroid_mean, axis=1))
                centroids.append(cluster_points[closest_idx])

            centroids = np.array(centroids)

        indices = np.arange(tgt_pen_emb.shape[0])
        q_idxs = []
        used_points = set()

        for centroid in centroids:
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
