import os
import sys
import numpy as np
import torch
import time
from datetime import datetime
from torchvision import transforms
from omegaconf import OmegaConf
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from clue import CLUESampling

sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import euclidean_distances

#Plots the single image at the center of cluster

class MNMv2Subset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
        }

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_centroid_image(centroid_image, save_dir, cluster_idx):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.imshow(centroid_image.squeeze(), cmap='gray')
    plt.title(f'Centroid Image for Cluster {cluster_idx}')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f'centroid_cluster_{cluster_idx}.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training or loading a model.")
    parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True, 
                        help="Whether to train the model")
    parser.add_argument('--num_clusters', type=int, default=5, help="Number of clusters.")
    parser.add_argument('--clue_softmax_t', type=float, default=1.0, help="Temperature.")
    parser.add_argument('--adapt_num_epochs', type=int, default=20, help="Number epochs for finetuning.")
    parser.add_argument('--cluster_type', type=str, default='centroids', 
                        help="This parameter determines whether we will train our model on centroids or on the most confident data close to centroids.")
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/home/chopra/lab-git/MedImSeg-Lab24/checkpoints/mnmv2-15-19_10-12-2024-v1.ckpt', 
                        help="Path to the model checkpoint.")
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help="Device to use for training (e.g., 'cuda:0', 'cuda:1', or 'cpu').")
    parser.add_argument('--paral', type=bool, default=False, 
                        help='Enabling parallelization of the embedding, clustering, and model completion process')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='The threshold removes the images in which the model is most confident')

    # [NEW] Additional parameters for CLUE
    parser.add_argument('--use_uncertainty', type=str2bool, nargs='?', const=True, default=True, 
                        help="Whether to use uncertainty-based sample weights (True) or uniform sample weights (False)")
    parser.add_argument('--kernel_size', type=int, default=3, 
                        help="Kernel size for AvgPool2D in CLUE embedding extraction.")
    parser.add_argument('--stride', type=int, default=2, 
                        help="Stride for AvgPool2D in CLUE embedding extraction.")
    parser.add_argument('--target_size', type=int, default=1024, 
                        help="Target size (spatial area) for pooling in CLUE embedding extraction.")

    args = parser.parse_args()

    mnmv2_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/mnmv2.yaml')
    save_dir = '/home/chopra/lab-git/MedImSeg-Lab24/results/cluster_centroid_images'
    os.makedirs(save_dir, exist_ok=True)

    datamodule = MNMv2DataModule(
        data_dir=mnmv2_config.data_dir,
        vendor_assignment=mnmv2_config.vendor_assignment,
        batch_size=mnmv2_config.batch_size,
        binary_target=mnmv2_config.binary_target,
        non_empty_target=mnmv2_config.non_empty_target,
    )

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    unet_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/monai_unet.yaml')
    unet = UNet(
        spatial_dims=unet_config.spatial_dims,
        in_channels=unet_config.in_channels,
        out_channels=unet_config.out_channels,
        channels=[unet_config.n_filters_init * 2 ** i for i in range(unet_config.depth)],
        strides=[2] * (unet_config.depth - 1),
        num_res_units=4
    )

    model = LightningSegmentationModel.load_from_checkpoint(
        args.checkpoint_path,
        map_location=device,
        model=unet,
        binary_target=True if unet_config.out_channels == 1 else False,
        lr=unet_config.lr,
        patience=unet_config.patience
    )
    model = model.to(device)
    model.eval()

    datamodule.setup(stage='test')
    clue_sampler = CLUESampling(
        dset=datamodule.mnm_test,
        train_idx=np.arange(len(datamodule.mnm_test)),
        model=model,
        device=device,
        args=args,  # args should now contain 'cluster_type'
        batch_size=32
    )



    num_clusters = args.num_clusters
    nearest_idx = clue_sampler.query(n=num_clusters)
    selected_samples = [datamodule.mnm_test[idx] for idx in nearest_idx]

    embeddings = clue_sampler.get_embeddings()
    centroids = clue_sampler.get_cluster_centers()

    for cluster_idx in range(num_clusters):
        cluster_embedding = centroids[cluster_idx]
        distances_to_cluster = euclidean_distances(embeddings, [cluster_embedding])
        nearest_image_idx = np.argmin(distances_to_cluster)

        nearest_image = datamodule.mnm_test[nearest_image_idx]["input"]
        plot_centroid_image(nearest_image, save_dir, cluster_idx)
