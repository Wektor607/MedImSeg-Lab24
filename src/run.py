import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import time
import random
from datetime import datetime
from omegaconf import OmegaConf
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from monai.networks.nets import UNet
from clue import CLUESampling  # Update the import path if needed

sys.path.append('../')
from data_utils.data_utils import MNMv2DataModule
from unet import LightningSegmentationModel
from torch.utils.data import Dataset

# TODO: Add weights and remove later

class MNMv2Subset(Dataset):
    def __init__(
        self,
        input,
        target,
    ):
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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

torch.set_num_threads(1)
set_seed(42)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training or loading a model.")
    parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=False, 
                        help="Whether to train the model")
    parser.add_argument('--num_clusters', type=int, default=10, help="Number of clusters.")
    parser.add_argument('--clue_softmax_t', type=float, default=1.0, help="Temperature.")
    parser.add_argument('--adapt_num_epochs', type=int, default=20, help="Number epochs for finetuning.")
    parser.add_argument('--checkpoint_path', type=str, 
                        default='../checkpoints/mnmv2-15-19_10-12-2024-v1.ckpt', 
                        help="Path to the model checkpoint.")
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help="Device to use for training (e.g., 'cuda:0', 'cuda:1', or 'cpu').")
    parser.add_argument('--paral', type=bool, default=False, 
                        help='Enabling parallelization of the embedding, clustering, and model completion process')

    # [NEW] Additional parameters for CLUE
    parser.add_argument('--clustering', type=str, default='KMeans', 
                        help="KMeans, MBKMeans, DBSCAN")
    parser.add_argument('--uncertainty', type=str, default='Margin', 
                        help="CrossEntropy, Distance, MutalInfo, Margin, Uniform")
    parser.add_argument('--kernel_size', type=int, default=3, 
                        help="Kernel size for AvgPool2D in CLUE embedding extraction.")
    parser.add_argument('--stride', type=int, default=2, 
                        help="Stride for AvgPool2D in CLUE embedding extraction.")
    parser.add_argument('--target_size', type=int, default=1024, 
                        help="Target size (spatial area) for pooling in CLUE embedding extraction.")
    parser.add_argument('--eps', type=float, default=0.1, 
                    help="The maximum distance between two samples for them to be considered as part of the same cluster in DBSCAN.")
    parser.add_argument('--min_samples', type=int, default=3, 
                    help="The minimum number of points required to form a dense region (cluster) in DBSCAN.")


    args = parser.parse_args()

    mnmv2_config   = OmegaConf.load('../configs/mnmv2.yaml')
    unet_config    = OmegaConf.load('../configs/monai_unet.yaml')
    trainer_config = OmegaConf.load('../configs/unet_trainer.yaml')

    cfg = OmegaConf.create({
        'unet_config': unet_config,
        'binary_target': True if unet_config.out_channels == 1 else False,
        'lr': unet_config.lr,
        'patience': unet_config.patience,
        'paral': args.paral,
        'adapt_num_epochs': args.adapt_num_epochs,
        'clue_softmax_t': args.clue_softmax_t,
        'dataset': OmegaConf.to_container(mnmv2_config),
        'batch_size': unet_config.get('batch_size', 32),
        'unet': OmegaConf.to_container(unet_config),
        'trainer': OmegaConf.to_container(trainer_config),

        # [NEW] Pass new params into cfg
        'clustering': args.clustering,
        'uncertainty': args.uncertainty,
        'kernel_size': args.kernel_size,
        'stride': args.stride,
        'target_size': args.target_size,

        # DBSCAN
        'eps': float(args.eps),
        'min_samples': args.min_samples

    })
    # init datamodule
    datamodule = MNMv2DataModule(
        data_dir=mnmv2_config.data_dir,
        vendor_assignment=mnmv2_config.vendor_assignment,
        batch_size=mnmv2_config.batch_size,
        binary_target=mnmv2_config.binary_target,
        non_empty_target=mnmv2_config.non_empty_target,
    )

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
            devices=[int(args.device[-1])]  # adapt to your hardware if needed
        )
        trainer.fit(model, datamodule=datamodule)

    else:
        # Handle loading a pre-trained model if not training
        load_as_lightning_module = True
        load_as_pytorch_module = False

        if load_as_lightning_module:
            unet_config    = OmegaConf.load('../configs/monai_unet.yaml')
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
                devices=[int(args.device[-1])]
            )

        elif load_as_pytorch_module:
            checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
            model_state_dict = checkpoint['state_dict']
            model_state_dict = {k.replace('model.model.', 'model.'): v 
                                for k, v in model_state_dict.items() 
                                if k.startswith('model.')}
            model_config = checkpoint['hyper_parameters']['cfgs']

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
        batch_size=cfg.get('batch_size', 32)  # Pass batch_size explicitly
    )

    # There is no need to set the number of clusters more than the number of images
    if args.num_clusters > len(clue_sampler.dset):
        args.num_clusters = len(clue_sampler.dset)

    start = time.time()
    nearest_idx = clue_sampler.query(n=args.num_clusters)
    end = time.time()
    print("Working Time: ", end - start)

    selected_samples = [datamodule.mnm_test[idx] for idx in nearest_idx]

    # Fine-tuning the model
    datamodule.setup(stage='fit')
    selected_inputs = torch.stack([sample["input"] for sample in selected_samples])
    selected_targets = torch.stack([sample["target"] for sample in selected_samples])

    # Example: fine-tune ONLY on the newly selected data
    # combined_data = MNMv2Subset(
    #     input=selected_inputs,
    #     target=selected_targets,
    # )
    
    # If you want to fine-tune on the entire combined set, you'd do:
    combined_inputs = torch.cat([datamodule.mnm_train.input, selected_inputs], dim=0)
    combined_targets = torch.cat([datamodule.mnm_train.target, selected_targets], dim=0)
    combined_data = MNMv2Subset(
        input=combined_inputs,
        target=combined_targets,
    )

    datamodule.mnm_train = combined_data
    new_train_loader = datamodule.train_dataloader()

    model.train()
    trainer.fit(model=model, 
                train_dataloaders=new_train_loader, 
                val_dataloaders=datamodule.val_dataloader())

    # Save model after fine-tuning
    save_dir = '../pre-trained/finetuned_on_centroids'

    os.makedirs(save_dir, exist_ok=True)

    model_save_path = os.path.join(save_dir, f'fituned_model_on_centroids.pth')
    torch.save(model.state_dict(), model_save_path)

    # Getting results AFTER using CLUE
    datamodule.setup(stage='test')
    model = model.to(device)
    model.eval()
    test_perf = trainer.test(model, datamodule=datamodule)[0]

    # if test_perf['test_dsc'] > best_perf:
    #     # Write results to file
    #     if best_perf == 0:
    #         with open("/home/mikhelson/MedImSeg-Lab24/results/german_exp/dbscan.txt", "w") as f:
    #             f.write(f"eps\tmin_samples\tDice_Score\n")#Num_epochs\tCentroid_time\n")    
            
    #     with open("/home/mikhelson/MedImSeg-Lab24/results/german_exp/dbscan.txt", "a") as f:
    #         f.write(f"{eps}\t{min_samples}\t{test_perf['test_dsc']:.4f}\n") #\t{trainer.current_epoch:.4f}\t{end - start:.4f}\n") 
    #     best_perf = test_perf['test_dsc']
    # Write results to file
    # if seed == 0 and i == 'MBKMeans':
    #     with open("/home/mikhelson/MedImSeg-Lab24/results/german_exp/mbkclustering.txt", "w") as f:
    #         f.write(f"Clust_Method\tLoss\tDice_Score\tWorking time\tSeed\n")#Num_epochs\tCentroid_time\n")    
    
    # with open("/home/mikhelson/MedImSeg-Lab24/results/german_exp/mbkclustering.txt", "a") as f:
    #     f.write(f"{i}\t{test_perf['test_loss']:.4f}\t{test_perf['test_dsc']:.4f}\t{end - start:.4f}\t{seed}\n") #\t{trainer.current_epoch:.4f}\t{end - start:.4f}\n")
