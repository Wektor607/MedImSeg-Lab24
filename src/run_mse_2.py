# import os
# import sys
# import numpy as np
# import torch
# import time
# from datetime import datetime
# from torchvision import transforms
# from omegaconf import OmegaConf
# import argparse
# import lightning as L
# from lightning.pytorch.callbacks import ModelCheckpoint
# from monai.networks.nets import UNet
# from sklearn.metrics import mean_squared_error
# from tqdm import tqdm

# from clue import CLUESampling  # Update the import path if needed
# sys.path.append('../')
# from data_utils import MNMv2DataModule
# from unet import LightningSegmentationModel
# from torch.utils.data import Dataset


# # Custom Dataset Subset
# class MNMv2Subset(Dataset):
#     def __init__(self, input, target):
#         self.input = input
#         self.target = target

#     def __len__(self):
#         return self.input.shape[0]
    
#     def __getitem__(self, idx):
#         return {
#             "input": self.input[idx], 
#             "target": self.target[idx],
#         }

# # Helper function to parse boolean arguments
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

# # Function to compute MSE between cluster distances and embedding distances
# def compute_mse(cluster_centers, embeddings):
#     cluster_distances = np.linalg.norm(cluster_centers[:, None, :] - cluster_centers[None, :, :], axis=-1)
#     embedding_distances = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)
#     mse = mean_squared_error(cluster_distances.flatten(), embedding_distances.flatten())
#     return mse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Training or loading a model.")
#     parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True, 
#                         help="Whether to train the model")
#     parser.add_argument('--num_clusters', type=int, default=5, help="Number of clusters.")
#     parser.add_argument('--clue_softmax_t', type=float, default=1.0, help="Temperature.")
#     parser.add_argument('--adapt_num_epochs', type=int, default=20, help="Number epochs for finetuning.")
#     parser.add_argument('--cluster_type', type=str, default='centroids', 
#                         help="This parameter determines whether we will train our model on centroids or on the most confident data close to centroids.")
#     parser.add_argument('--checkpoint_path', type=str, 
#                         default='/home/chopra/lab-git/MedImSeg-Lab24/checkpoints/mnmv2-15-19_10-12-2024-v1.ckpt', 
#                         help="Path to the model checkpoint.")
#     parser.add_argument('--device', type=str, default='cuda:0', 
#                         help="Device to use for training (e.g., 'cuda:0', 'cuda:1', or 'cpu').")
#     parser.add_argument('--paral', type=bool, default=False, 
#                         help='Enabling parallelization of the embedding, clustering, and model completion process')
#     parser.add_argument('--threshold', type=float, default=0.5, 
#                         help='The threshold removes the images in which the model is most confident')

#     # [NEW] Additional parameters for CLUE
#     parser.add_argument('--use_uncertainty', type=str2bool, nargs='?', const=True, default=True, 
#                         help="Whether to use uncertainty-based sample weights (True) or uniform sample weights (False)")
#     parser.add_argument('--kernel_size', type=int, default=3, 
#                         help="Kernel size for AvgPool2D in CLUE embedding extraction.")
#     parser.add_argument('--stride', type=int, default=2, 
#                         help="Stride for AvgPool2D in CLUE embedding extraction.")
#     parser.add_argument('--target_size', type=int, default=1024, 
#                         help="Target size (spatial area) for pooling in CLUE embedding extraction.")

#     args = parser.parse_args()

#     mnmv2_config   = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/mnmv2.yaml')
#     unet_config    = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/monai_unet.yaml')
#     trainer_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/unet_trainer.yaml')
#     args = parser.parse_args()

#     # Load configurations
#     mnmv2_config   = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/mnmv2.yaml')
#     unet_config    = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/monai_unet.yaml')
#     trainer_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/unet_trainer.yaml')

#     for i in [1, 10, 25, 40, 50, 60, 75, 100, 125, 150, 175, 200]:
#         datamodule = MNMv2DataModule(
#             data_dir=mnmv2_config.data_dir,
#             vendor_assignment=mnmv2_config.vendor_assignment,
#             batch_size=mnmv2_config.batch_size,
#             binary_target=mnmv2_config.binary_target,
#             non_empty_target=mnmv2_config.non_empty_target,
#         )

#         cfg = OmegaConf.create({
#             'unet_config': unet_config,
#             'binary_target': unet_config.out_channels == 1,
#             'lr': unet_config.lr,
#             'patience': unet_config.patience,
#             'threshold': args.threshold,
#             'adapt_num_epochs': args.adapt_num_epochs,
#             'cluster_type': args.cluster_type,
#             'clue_softmax_t': args.clue_softmax_t,
#             'dataset': OmegaConf.to_container(mnmv2_config),
#             'batch_size': unet_config.get('batch_size', 32),
#             'use_uncertainty': args.use_uncertainty,
#             'kernel_size': args.kernel_size,
#             'stride': args.stride,
#             'target_size': args.target_size,
#             'paral': args.paral
#         })

#         # Train or load model
#         if args.train:
#             model = LightningSegmentationModel(cfg=cfg).to(args.device)
#             trainer = L.Trainer(
#                 max_epochs=trainer_config.max_epochs,
#                 callbacks=[
#                     ModelCheckpoint(
#                         dirpath=trainer_config.model_checkpoint.dirpath,
#                         save_top_k=trainer_config.model_checkpoint.save_top_k,
#                         monitor=trainer_config.model_checkpoint.monitor,
#                     )
#                 ],
#                 precision='16-mixed',
#                 devices=1,
#             )
#             trainer.fit(model, datamodule=datamodule)
#         else:
#             # Handle loading a pre-trained model if not training
#             load_as_lightning_module = True
#             load_as_pytorch_module = False

#             if load_as_lightning_module:
#                 unet_config    = OmegaConf.load('../../MedImSeg-Lab24/configs/monai_unet.yaml')
#                 unet = UNet(
#                     spatial_dims=unet_config.spatial_dims,
#                     in_channels=unet_config.in_channels,
#                     out_channels=unet_config.out_channels,
#                     channels=[unet_config.n_filters_init * 2 ** i for i in range(unet_config.depth)],
#                     strides=[2] * (unet_config.depth - 1),
#                     num_res_units=4
#                 )
                
#                 model = LightningSegmentationModel.load_from_checkpoint(
#                     args.checkpoint_path,
#                     map_location=torch.device("cpu"),
#                     model=unet,
#                     binary_target=True if unet_config.out_channels == 1 else False,
#                     lr=unet_config.lr,
#                     patience=unet_config.patience,
#                     cfg=cfg
#                 )

#                 trainer = L.Trainer(
#                     limit_train_batches=trainer_config.limit_train_batches,
#                     max_epochs=args.adapt_num_epochs,
#                     callbacks=[
#                         ModelCheckpoint(
#                             dirpath=trainer_config.model_checkpoint.dirpath,
#                             save_top_k=trainer_config.model_checkpoint.save_top_k, 
#                             monitor=trainer_config.model_checkpoint.monitor,
#                         )
#                     ],
#                     precision='16-mixed',
#                     devices=[1]
#                 )

#             elif load_as_pytorch_module:
#                 checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
#                 model_state_dict = checkpoint['state_dict']
#                 model_state_dict = {k.replace('model.model.', 'model.'): v 
#                                     for k, v in model_state_dict.items() 
#                                     if k.startswith('model.')}
#                 model_config = checkpoint['hyper_parameters']['cfgs']

#                 print(model_config)

#                 model = UNet(
#                     spatial_dims=model_config['unet']['spatial_dims'],
#                     in_channels=model_config['unet']['in_channels'],
#                     out_channels=model_config['unet']['out_channels'],
#                     channels=[model_config['unet']['n_filters_init'] * 2 ** i for i in range(model_config['unet']['depth'])],
#                     strides=[2] * (model_config['unet']['depth'] - 1),
#                     num_res_units=4
#                 )

#                 model.load_state_dict(model_state_dict)

#         device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#         model = model.to(device)

#         # Evaluate model
#         datamodule.setup(stage='test')
#         model.eval()
#         trainer = L.Trainer(devices=1)
#         trainer.test(model, datamodule=datamodule)

#         # CLUE sampling
#         test_idx = np.arange(len(datamodule.mnm_test))
#         clue_sampler = CLUESampling(
#             dset=datamodule.mnm_test,
#             train_idx=test_idx,
#             model=model,
#             device=device,
#             args=cfg,
#             batch_size=cfg.get('batch_size', 32)  # Pass batch_size explicitly
#         )

#         # There is no need to set the number of clusters more than the number of images
#         if i > len(clue_sampler.dset):
#             i = len(clue_sampler.dset)

#         start = time.time()
#         nearest_idx = clue_sampler.query(n=i)
#         end = time.time()
#         print("Working Time: ", end - start)

#         selected_samples = [datamodule.mnm_test[idx] for idx in nearest_idx]

#         # Fine-tuning the model
#         datamodule.setup(stage='fit')
#         selected_inputs = torch.stack([sample["input"] for sample in selected_samples])
#         selected_targets = torch.stack([sample["target"] for sample in selected_samples])

#         # Example: fine-tune ONLY on the newly selected data
#         combined_data = MNMv2Subset(
#             input=selected_inputs,
#             target=selected_targets,
#         )
        
#         # If you want to fine-tune on the entire combined set, you'd do:
#         # combined_inputs = torch.cat([datamodule.mnm_train.input, selected_inputs], dim=0)
#         # combined_targets = torch.cat([datamodule.mnm_train.target, selected_targets], dim=0)
#         # combined_data = MNMv2Subset(
#         #     input=combined_inputs,
#         #     target=combined_targets,
#         # )

#         datamodule.mnm_train = combined_data
#         new_train_loader = datamodule.train_dataloader()

#         model.train()
#         trainer.fit(model=model, 
#                     train_dataloaders=new_train_loader, 
#                     val_dataloaders=datamodule.val_dataloader())

#         # Save model after fine-tuning
#         if args.cluster_type == 'centroids':
#             save_dir = '../pre-trained/finetuned_on_centroids'
#         else:
#             save_dir = '../pre-trained/finetuned_on_uncert_points'

#         os.makedirs(save_dir, exist_ok=True)

#         model_save_path = os.path.join(save_dir, f'fituned_model_on_{args.cluster_type}.pth')
#         torch.save(model.state_dict(), model_save_path)

#         # Getting results AFTER using CLUE
#         datamodule.setup(stage='test')
#         model = model.to(device)
#         model.eval()
#         test_perf = trainer.test(model, datamodule=datamodule)[0]

#         # Write results to file
#         if i == 1:
#             with open("/home/chopra/lab-git/MedImSeg-Lab24/results/uniform_weights/128/results_test_100.txt", "w") as f:
#                 f.write(f"Num_Centroids\tLoss\tDice_Score\tNum_epochs\tCentroid_time\n")    
        
#         with open("/home/chopra/lab-git/MedImSeg-Lab24/results/uniform_weights/128/results_test_100.txt", "a") as f:
#             f.write(f"{i}\t{test_perf['test_loss']:.4f}\t{test_perf['test_dsc']:.4f}\t{trainer.current_epoch:.4f}\t{end - start:.4f}\n")


#         nearest_idx = clue_sampler.query(n=i)
#         cluster_centers = clue_sampler.get_cluster_centers()
#         embeddings = clue_sampler.get_embeddings()

#         # Compute and log MSE
#         mse = compute_mse(cluster_centers, embeddings)
#         print(f"Number of clusters: {i}, MSE: {mse}")

#         log_path = "/home/chopra/lab-git/MedImSeg-Lab24/results/mse_comparison.txt"
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)
#         with open(log_path, "a") as mse_file:
#             mse_file.write(f"{i}\t{test_perf['test_loss']:.4f}\t{test_perf['test_dsc']:.4f}\t{trainer.current_epoch:.4f}\t{end - start:.4f}\t{mse:.4f}\n")





# import os
# import sys
# import numpy as np
# import torch
# from datetime import datetime
# from omegaconf import OmegaConf
# import argparse
# import lightning as L
# from lightning.pytorch.callbacks import ModelCheckpoint
# from sklearn.metrics import mean_squared_error

# from clue import CLUESampling
# sys.path.append('../')
# from data_utils import MNMv2DataModule
# from unet import LightningSegmentationModel

# def compute_mse_between_centroids(cluster_centers):
#     """
#     Compute the Mean Squared Error (MSE) between distances of cluster centroids.
#     """
#     if cluster_centers is None or len(cluster_centers) < 2:
#         return 0.0  # Set distance as 0 when there are fewer than 2 centroids
    
#     centroid_distances = np.linalg.norm(cluster_centers[:, None, :] - cluster_centers[None, :, :], axis=-1)
#     mse = mean_squared_error(centroid_distances.flatten(), np.zeros_like(centroid_distances.flatten()))
#     return mse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Training or loading a model.")
#     parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True, 
#                         help="Whether to train the model")
#     parser.add_argument('--num_clusters', type=int, default=5, help="Number of clusters.")
#     parser.add_argument('--clue_softmax_t', type=float, default=1.0, help="Temperature.")
#     parser.add_argument('--adapt_num_epochs', type=int, default=20, help="Number epochs for finetuning.")
#     parser.add_argument('--cluster_type', type=str, default='centroids', 
#                         help="This parameter determines whether we will train our model on centroids or on the most confident data close to centroids.")
#     parser.add_argument('--checkpoint_path', type=str, 
#                         default='/home/chopra/lab-git/MedImSeg-Lab24/checkpoints/mnmv2-15-19_10-12-2024-v1.ckpt', 
#                         help="Path to the model checkpoint.")
#     parser.add_argument('--device', type=str, default='cuda:0', 
#                         help="Device to use for training (e.g., 'cuda:0', 'cuda:1', or 'cpu').")
#     parser.add_argument('--paral', type=bool, default=False, 
#                         help='Enabling parallelization of the embedding, clustering, and model completion process')
#     parser.add_argument('--threshold', type=float, default=0.5, 
#                         help='The threshold removes the images in which the model is most confident')

#     # [NEW] Additional parameters for CLUE
#     parser.add_argument('--use_uncertainty', type=str2bool, nargs='?', const=True, default=True, 
#                         help="Whether to use uncertainty-based sample weights (True) or uniform sample weights (False)")
#     parser.add_argument('--kernel_size', type=int, default=3, 
#                         help="Kernel size for AvgPool2D in CLUE embedding extraction.")
#     parser.add_argument('--stride', type=int, default=2, 
#                         help="Stride for AvgPool2D in CLUE embedding extraction.")
#     parser.add_argument('--target_size', type=int, default=1024, 
#                         help="Target size (spatial area) for pooling in CLUE embedding extraction.")
#     args = parser.parse_args()

#     # Load configurations
#     mnmv2_config   = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/mnmv2.yaml')
#     unet_config    = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/monai_unet.yaml')
#     trainer_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/unet_trainer.yaml')

#     # Set up data module
#     datamodule = MNMv2DataModule(
#         data_dir=mnmv2_config.data_dir,
#         vendor_assignment=mnmv2_config.vendor_assignment,
#         batch_size=mnmv2_config.batch_size,
#         binary_target=mnmv2_config.binary_target,
#         non_empty_target=mnmv2_config.non_empty_target,
#     )
#     datamodule.setup(stage='test')

#     # Convert cfg to OmegaConf object
#     cfg = OmegaConf.create({
#             'unet_config': unet_config,
#             'binary_target': True if unet_config.out_channels == 1 else False,
#             'lr': unet_config.lr,
#             'patience': unet_config.patience,
#             'paral': args.paral,
#             'threshold': args.threshold,
#             'adapt_num_epochs': args.adapt_num_epochs,
#             'cluster_type': args.cluster_type,
#             'clue_softmax_t': args.clue_softmax_t,
#             'dataset': OmegaConf.to_container(mnmv2_config),
#             'batch_size': unet_config.get('batch_size', 32),
#             'unet': OmegaConf.to_container(unet_config),
#             'trainer': OmegaConf.to_container(trainer_config),

#             # [NEW] Pass new params into cfg
#             'use_uncertainty': args.use_uncertainty,
#             'kernel_size': args.kernel_size,
#             'stride': args.stride,
#             'target_size': args.target_size,
#         })


#     if not args.train:
#         print("[INFO] Loading pre-trained model...")
#         model = LightningSegmentationModel.load_from_checkpoint(args.checkpoint_path, cfg=cfg).to(args.device)
#     else:
#         print("[INFO] Training a new model...")
#         model = LightningSegmentationModel(cfg=cfg).to(args.device)


#     # CLUE sampling
#     print("[INFO] Starting CLUE sampling...")
#     test_idx = np.arange(len(datamodule.mnm_test))
#     clue_sampler = CLUESampling(
#         dset=datamodule.mnm_test,
#         train_idx=test_idx,
#         model=model,
#         device=args.device,
#         args=cfg,
#         batch_size=cfg['batch_size'],
#     )

#     log_dir = "/home/chopra/lab-git/MedImSeg-Lab24/results/"
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, "mse_results_centroids_new.txt")

#     for num_clusters in range(1, 26):
#         print(f"[INFO] Computing nearest indices and cluster centers for {num_clusters} clusters...")
#         nearest_idx = clue_sampler.query(n=num_clusters)
#         cluster_centers = clue_sampler.get_cluster_centers()

#         # Compute MSE
#         print("[INFO] Computing MSE between cluster centroids...")
#         mse = compute_mse_between_centroids(cluster_centers)
#         print(f"[INFO] Number of clusters: {num_clusters}, MSE: {mse}")

#         # Log results
#         with open(log_file, "a") as f:
#             f.write(f"Clusters: {num_clusters}, MSE: {mse:.4f}\n")
#     print(f"[INFO] MSE results logged to {log_file}")



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
from scipy.spatial.distance import pdist  # Added for computing pairwise Euclidean distances

from monai.networks.nets import UNet
from clue import CLUESampling  # Update the import path if needed

sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel
from torch.utils.data import Dataset

# Integrated function to compute the mean Euclidean distance between all pairs of centroids for each method
def compute_centroid_distances(centroids_dict):
    results = {}
    intra_distances = {}
    for method, centroids in centroids_dict.items():
        if len(centroids) > 1:
            dists = pdist(centroids, metric='euclidean')
            intra_distances[method] = np.mean(dists)
        else:
            intra_distances[method] = np.nan
    return intra_distances

# Dataset Subset class
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

    mnmv2_config   = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/mnmv2.yaml')
    unet_config    = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/monai_unet.yaml')
    trainer_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/unet_trainer.yaml')

    # Loop over i to select an increasing number of samples
    for i in [1,50, 75,100,125,150,175,200]:

        # init datamodule
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
            'use_uncertainty': args.use_uncertainty,
            'kernel_size': args.kernel_size,
            'stride': args.stride,
            'target_size': args.target_size,
        })

        if args.train:
            model = LightningSegmentationModel(cfg=cfg)
            
            now = datetime.now()
            filename = 'mnmv2-' + now.strftime("%H-%M_%d-%m-%Y")

            trainer = L.Trainer(
                limit_train_batches=trainer_config.limit_train_batches,
                max_epochs=trainer_config.max_epochs,
                callbacks=[
                    ModelCheckpoint(
                        dirpath=trainer_config.model_checkpoint.dirpath,
                        filename=filename,
                        save_top_k=trainer_config.model_checkpoint.save_top_k, 
                        monitor=trainer_config.model_checkpoint.monitor,
                    )
                ],
                precision='16-mixed',
                devices=[1]
            )
            trainer.fit(model, datamodule=datamodule)

        else:
            load_as_lightning_module = True
            load_as_pytorch_module = False

            if load_as_lightning_module:
                unet_config    = OmegaConf.load('../../MedImSeg-Lab24/configs/monai_unet.yaml')
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
                    map_location=torch.device("cpu"),
                    model=unet,
                    binary_target=True if unet_config.out_channels == 1 else False,
                    lr=unet_config.lr,
                    patience=unet_config.patience,
                    cfg=cfg
                )

                trainer = L.Trainer(
                    limit_train_batches=trainer_config.limit_train_batches,
                    max_epochs=args.adapt_num_epochs,
                    callbacks=[
                        ModelCheckpoint(
                            dirpath=trainer_config.model_checkpoint.dirpath,
                            save_top_k=trainer_config.model_checkpoint.save_top_k, 
                            monitor=trainer_config.model_checkpoint.monitor,
                        )
                    ],
                    precision='16-mixed',
                    devices=[1]
                )

            elif load_as_pytorch_module:
                checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
                model_state_dict = checkpoint['state_dict']
                model_state_dict = {k.replace('model.model.', 'model.'): v 
                                    for k, v in model_state_dict.items() 
                                    if k.startswith('model.')}
                model_config = checkpoint['hyper_parameters']['cfgs']

                print(model_config)

                model = UNet(
                    spatial_dims=model_config['unet']['spatial_dims'],
                    in_channels=model_config['unet']['in_channels'],
                    out_channels=model_config['unet']['out_channels'],
                    channels=[model_config['unet']['n_filters_init'] * 2 ** i for i in range(model_config['unet']['depth'])],
                    strides=[2] * (model_config['unet']['depth'] - 1),
                    num_res_units=4
                )

                model.load_state_dict(model_state_dict)
        
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Getting results BEFORE using CLUE
        datamodule.setup(stage='test')
        model.eval()
        test_res = trainer.test(model, datamodule=datamodule)

        # Getting centroids / nearest points to centroids
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
        
        # [NEW] Compute mean Euclidean distance between all pairs of centroids using compute_centroid_distances
        cluster_centers = clue_sampler.get_cluster_centers()
        centroid_distances_dict = compute_centroid_distances({'centroids': cluster_centers})
        mean_centroid_distance = centroid_distances_dict['centroids']
        
        end = time.time()
        print("Working Time: ", end - start)
        print(f"Mean Euclidean distance between cluster centers: {mean_centroid_distance:.4f}")

        selected_samples = [datamodule.mnm_test[idx] for idx in nearest_idx]

        # Calculate centroids (mean of selected samples)
        selected_inputs = torch.stack([sample["input"] for sample in selected_samples])
        centroids = torch.mean(selected_inputs, dim=0)

        # Calculate MSE between the single centroid and each selected sample
        mse_values = []
        for sample in selected_samples:
            mse = mean_squared_error(centroids.cpu().numpy().flatten(), sample["input"].cpu().numpy().flatten())
            mse_values.append(mse)
        avg_mse = np.mean(mse_values)

        # Save MSE to file
        mse_file_path = "/home/chopra/lab-git/MedImSeg-Lab24/results/MSE/mse_values_centroids.txt"
        with open(mse_file_path, "a") as f:
            f.write(f"{i}\t{avg_mse:.4f}\t{mean_centroid_distance:.4f}\n")

        # Fine-tuning the model
        datamodule.setup(stage='fit')
        selected_targets = torch.stack([sample["target"] for sample in selected_samples])

        combined_data = MNMv2Subset(
            input=selected_inputs,
            target=selected_targets,
        )

        datamodule.mnm_train = combined_data
        new_train_loader = datamodule.train_dataloader()

        model.train()
        trainer.fit(model=model, 
                    train_dataloaders=new_train_loader, 
                    val_dataloaders=datamodule.val_dataloader())

        if args.cluster_type == 'centroids':
            save_dir = '../pre-trained/finetuned_on_centroids'
        else:
            save_dir = '../pre-trained/finetuned_on_uncert_points'

        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f'fituned_model_on_{args.cluster_type}.pth')
        torch.save(model.state_dict(), model_save_path)

        # Getting results AFTER using CLUE
        datamodule.setup(stage='test')
        model = model.to(device)
        model.eval()
        test_perf = trainer.test(model, datamodule=datamodule)[0]

        results_file = "/home/chopra/lab-git/MedImSeg-Lab24/results/MSE/results_test_32_100_1_200_unifrom_centroids.txt"
        if i == 1:
            with open(results_file, "w") as f:
                f.write("Num_Centroids\tLoss\tDice_Score\tMSE_Centroid_Samples\tMSE_Cluster_Centers\tNum_epochs\tCentroid_time\n")

        with open(results_file, "a") as f:
            f.write(
                f"{i}\t"
                f"{test_perf['test_loss']:.4f}\t"
                f"{test_perf['test_dsc']:.4f}\t"
                f"{avg_mse:.4f}\t"
                f"{mean_centroid_distance:.4f}\t"
                f"{trainer.current_epoch:.4f}\t"
                f"{end - start:.4f}\n"
            )