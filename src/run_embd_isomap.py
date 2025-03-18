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
from sklearn.manifold import Isomap  # Import Isomap

from monai.networks.nets import UNet
from clue_isomap import CLUESampling  # Use the updated Isomap-based CLUE

sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel
from torch.utils.data import Dataset

class MNMv2Subset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        return {"input": self.input[idx], "target": self.target[idx]}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    # New Isomap parameters
    parser.add_argument('--isomap_n_components', type=int, default=2, help="Number of components for Isomap.")
    parser.add_argument('--isomap_n_neighbors', type=int, default=5, help="Number of neighbors for Isomap.")
    parser.add_argument('--kernel_size', type=int, default=3, 
                        help="Kernel size for AvgPool2D in CLUE embedding extraction.")
    parser.add_argument('--stride', type=int, default=2, 
                        help="Stride for AvgPool2D in CLUE embedding extraction.")
    parser.add_argument('--target_size', type=int, default=1024, 
                        help="Target size (spatial area) for pooling in CLUE embedding extraction.")

    args = parser.parse_args()

    mnmv2_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/mnmv2.yaml')
    unet_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/monai_unet.yaml')
    trainer_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/unet_trainer.yaml')

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
        datamodule = MNMv2DataModule(
            data_dir=mnmv2_config.data_dir,
            vendor_assignment=mnmv2_config.vendor_assignment,
            batch_size=mnmv2_config.batch_size,
            binary_target=mnmv2_config.binary_target,
            non_empty_target=mnmv2_config.non_empty_target,
        )

        cfg = OmegaConf.create({
            'unet_config': unet_config,
            'binary_target': True if unet_config.out_channels == 1 else False,
            'lr': unet_config.lr,
            'patience': unet_config.patience,
            'paral': args.paral,
            'threshold': args.threshold,
            'adapt_num_epochs': args.adapt_num_epochs,
            'cluster_type': args.cluster_type,
            'clue_softmax_t': args.clue_softmax_t,
            'dataset': OmegaConf.to_container(mnmv2_config),
            'batch_size': unet_config.get('batch_size', 32),
            'unet': OmegaConf.to_container(unet_config),
            'trainer': OmegaConf.to_container(trainer_config),

            # [NEW] Pass new params into cfg
            'use_uncertainty': args.use_uncertainty,
            'kernel_size': args.kernel_size,
            'stride': args.stride,
            'target_size': args.target_size,
            # Ensure Isomap parameters exist
            'isomap_n_components': getattr(args, 'isomap_n_components', 2),
            'isomap_n_neighbors': getattr(args, 'isomap_n_neighbors', 5),
        })

        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = LightningSegmentationModel.load_from_checkpoint(
            args.checkpoint_path, map_location=device, cfg=cfg
        ).to(device)

        datamodule.setup(stage='test')
        model.eval()
        trainer = L.Trainer(devices=[1])
        test_res = trainer.test(model, datamodule=datamodule)

        test_idx = np.arange(len(datamodule.mnm_test))
        clue_sampler = CLUESampling(
            dset=datamodule.mnm_test,
            train_idx=test_idx,
            model=model,
            device=device,
            args=cfg,
            batch_size=cfg.get('batch_size', 32)
        )

        if i > len(clue_sampler.dset):
            i = len(clue_sampler.dset)

        start = time.time()
        nearest_idx = clue_sampler.query(n=i)
        end = time.time()
        print("Working Time: ", end - start)

        embeddings = clue_sampler.get_embeddings()

        # Apply Isomap instead of PCA
        print("[INFO] Applying Isomap for dimensionality reduction...")
        isomap = Isomap(n_neighbors=args.isomap_n_neighbors, n_components=args.isomap_n_components)
        reduced_embeddings = isomap.fit_transform(embeddings)

        centroids = clue_sampler.get_cluster_centers()
        distances = np.linalg.norm(reduced_embeddings[:, None, :] - centroids[None, :, :], axis=-1)
        nearest_centroid_indices = np.argmin(distances, axis=1)
        nearest_centroids = centroids[nearest_centroid_indices]

        mse = mean_squared_error(reduced_embeddings, nearest_centroids)
        mse_file_path = "/home/chopra/lab-git/MedImSeg-Lab24/results/isomap/isomap_mse_embd_results.txt"
        with open(mse_file_path, "a") as f:
            f.write(f"Num_Clusters: {args.num_clusters}, MSE: {mse:.4f}\n")
        print(f"MSE logged in {mse_file_path}")

        selected_samples = [datamodule.mnm_test[idx] for idx in nearest_idx]

        datamodule.setup(stage='fit')
        selected_inputs = torch.stack([sample["input"] for sample in selected_samples])
        selected_targets = torch.stack([sample["target"] for sample in selected_samples])

        combined_data = MNMv2Subset(input=selected_inputs, target=selected_targets)
        datamodule.mnm_train = combined_data
        new_train_loader = datamodule.train_dataloader()

        model.train()
        trainer.fit(model=model, train_dataloaders=new_train_loader, val_dataloaders=datamodule.val_dataloader())

        os.makedirs('../pre-trained/finetuned_on_centroids', exist_ok=True)
        model_save_path = '../pre-trained/finetuned_on_centroids/fituned_model.pth'
        torch.save(model.state_dict(), model_save_path)

        datamodule.setup(stage='test')
        model.eval()
        test_perf = trainer.test(model, datamodule=datamodule)[0]

        with open("/home/chopra/lab-git/MedImSeg-Lab24/results/isomap/results_test_isomap.txt", "a") as f:
            f.write(f"{i}\t{test_perf['test_loss']:.4f}\t{test_perf['test_dsc']:.4f}\t{mse:.4f}\t{trainer.current_epoch:.4f}\t{end - start:.4f}\n")
