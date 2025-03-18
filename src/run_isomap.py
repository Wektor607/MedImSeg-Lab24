import os
import sys
import numpy as np
import torch
from datetime import datetime
from omegaconf import OmegaConf
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from types import SimpleNamespace
from sklearn.manifold import Isomap  # Import Isomap
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# Import the modified CLUE sampling with Isomap
from clue_isomap import CLUESampling
sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training or loading a model.")
    parser.add_argument('--train', action='store_true', help="Enable training mode (default: False)")
    parser.add_argument('--no-train', dest='train', action='store_false', help="Disable training mode")
    parser.set_defaults(train=True)

    parser.add_argument('--num_clusters', type=int, default=5, help="Number of clusters.")
    parser.add_argument('--clue_softmax_t', type=float, default=1.0, help="Temperature for softmax.")
    parser.add_argument('--adapt_num_epochs', type=int, default=20, help="Number of epochs for fine-tuning.")
    parser.add_argument('--cluster_type', type=str, default='centroids', help="Type of clusters to use.")
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/home/chopra/lab-git/MedImSeg-Lab24/checkpoints/mnmv2-15-19_10-12-2024-v1.ckpt', 
                        help="Path to the model checkpoint.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for training.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument('--paral', type=bool, default=False, help='Enable parallelization.')
    parser.add_argument('--use_uncertainty', action='store_true', help="Enable uncertainty weights (default: False)")
    parser.add_argument('--no-use_uncertainty', dest='use_uncertainty', action='store_false', help="Disable uncertainty weights")
    parser.set_defaults(use_uncertainty=True)

    parser.add_argument('--kernel_size', type=int, default=3, help="Kernel size for pooling.")
    parser.add_argument('--stride', type=int, default=2, help="Stride for pooling.")
    parser.add_argument('--target_size', type=int, default=1024, help="Target size for pooling.")

    # Isomap Parameters
    parser.add_argument('--isomap_n_components', type=int, default=5, help="Number of Isomap components.")
    parser.add_argument('--isomap_n_neighbors', type=int, default=10, help="Number of neighbors for Isomap.")

    args = parser.parse_args()

    # Load configurations
    mnmv2_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/mnmv2.yaml')
    unet_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/monai_unet.yaml')
    trainer_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/unet_trainer.yaml')

    for num_samples in range(1, 26):
        print(f"[INFO] Running experiment with {num_samples} clusters")

        # Set up data module
        datamodule = MNMv2DataModule(
            data_dir=mnmv2_config.data_dir,
            vendor_assignment=mnmv2_config.vendor_assignment,
            batch_size=mnmv2_config.batch_size,
            binary_target=mnmv2_config.binary_target,
            non_empty_target=mnmv2_config.non_empty_target,
        )
        datamodule.setup(stage='test')

        print("[INFO] Starting CLUE sampling...")
        test_idx = np.arange(len(datamodule.mnm_test))
        clue_sampler = CLUESampling(
            dset=datamodule.mnm_test,
            train_idx=test_idx,
            model=LightningSegmentationModel.load_from_checkpoint(
                args.checkpoint_path, 
                cfg=SimpleNamespace(
                    unet_config=unet_config, 
                    lr=unet_config.get('lr', 0.001), 
                    patience=trainer_config.get('patience', 10), 
                    binary_target=mnmv2_config.binary_target, 
                    **vars(args)
                )
            ).to(args.device),
            device=args.device,
            args=args,
            batch_size=mnmv2_config.batch_size
        )

        print("[INFO] Computing nearest indices and embeddings...")
        nearest_idx = clue_sampler.query(n=num_samples)

        # Get embeddings
        embeddings = clue_sampler.get_embeddings()

        # Apply Isomap Transformation
        print("[INFO] Applying Isomap transformation...")
        isomap_transform = Isomap(n_components=args.isomap_n_components, n_neighbors=args.isomap_n_neighbors)
        reduced_embeddings = isomap_transform.fit_transform(embeddings)

        # Perform Clustering in Isomap Space
        kmeans = KMeans(n_clusters=num_samples, n_init=10, random_state=42)
        kmeans.fit(reduced_embeddings)
        transformed_centroids = kmeans.cluster_centers_  # Isomap-space centroids

        # Compute MSE between Transformed Centroids
        print("[INFO] Computing MSE between transformed centroids...")
        if num_samples == 1:
            mse = 0.0  # Single centroid MSE is always 0
        else:
            centroid_distances = euclidean_distances(transformed_centroids)
            mse = mean_squared_error(centroid_distances.flatten(), np.zeros_like(centroid_distances.flatten()))

        print(f"[INFO] Number of clusters: {num_samples}, MSE: {mse}")

        # Save MSE Results
        log_dir = "/home/chopra/lab-git/MedImSeg-Lab24/results/isomap/"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "mse_results_isomap_neww.txt")
        with open(log_file, "a") as f:
            f.write(f"Clusters: {num_samples}, MSE: {mse:.4f}\n")
        print(f"[INFO] MSE results logged to {log_file}")
