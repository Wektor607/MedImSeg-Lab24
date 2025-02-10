# import os
# import sys
# import numpy as np
# import torch
# from datetime import datetime
# from torchvision import transforms
# from omegaconf import OmegaConf
# import argparse
# import lightning as L
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# from monai.networks.nets import UNet
# from clue import CLUESampling

# sys.path.append('../')
# from data_utils import MNMv2DataModule
# from unet import LightningSegmentationModel
# from torch.utils.data import Dataset

# # TODO: Add weights and remove later
# class MNMv2Subset(Dataset):
#     def __init__(
#         self,
#         input,
#         target,
#         weight
#     ):
#         self.input = input
#         self.target = target
#         self.weight = weight

#     def __len__(self):
#         return self.input.shape[0]
    
#     def __getitem__(self, idx):
#         return {
#             "input": self.input[idx], 
#             "target": self.target[idx],
#             "weight": self.weight[idx]
#         }

# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser(description="Training or loading a model.")
#     # parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True, help="Whether to train the model")
#     # parser.add_argument('--n', type=int, default=4, help="Number of clusters.")
#     # parser.add_argument('--clue_softmax_t', type=float, default=0.1, help="Temperature.")
#     # parser.add_argument('--adapt_num_epochs', type=int, default=5, help="Number epochs for finetuning.")
#     # parser.add_argument('--cluster_type', type=str, default='centroids', help="This parameter determines whether we will train our model on centroids or on the most confident data close to centroids.")
#     # parser.add_argument('--checkpoint_path', type=str, default='../../MedImSeg-Lab24/pre-trained/trained_UNets/mnmv2-15-54_09-12-2024-v1.ckpt', 
#     #                     help="Path to the model checkpoint.")
#     # parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for training (e.g., 'cuda:0', 'cuda:1', or 'cpu').")
#     # args = parser.parse_args()

#     mnmv2_config   = OmegaConf.load('../../MedImSeg-Lab24/configs/mnmv2.yaml')
#     unet_config    = OmegaConf.load('../../MedImSeg-Lab24/configs/monai_unet.yaml')
#     trainer_config = OmegaConf.load('../../MedImSeg-Lab24/configs/unet_trainer.yaml')

#     # init datamodule
#     datamodule = MNMv2DataModule(
#         data_dir=mnmv2_config.data_dir,
#         vendor_assignment=mnmv2_config.vendor_assignment,
#         batch_size=mnmv2_config.batch_size,
#         binary_target=mnmv2_config.binary_target,
#         non_empty_target=mnmv2_config.non_empty_target,
#         split_ratio=0.5
#     )

#     # cfg = OmegaConf.create({
#     #     'unet_config': unet_config,
#     #     'binary_target': True if unet_config.out_channels == 1 else False,
#     #     'lr': unet_config.lr,
#     #     'patience': unet_config.patience,
#     #     'adapt_num_epochs': args.adapt_num_epochs,
#     #     'cluster_type': args.cluster_type,
#     #     'clue_softmax_t': args.clue_softmax_t,
#     #     'dataset': OmegaConf.to_container(mnmv2_config),
#     #     'unet': OmegaConf.to_container(unet_config),
#     #     'trainer': OmegaConf.to_container(trainer_config),
#     # })

#     # if args.train:
#     #     model = LightningSegmentationModel(cfg=cfg)
        
#     #     now = datetime.now()
#     #     filename = 'mnmv2-' + now.strftime("%H-%M_%d-%m-%Y")

#     #     trainer = L.Trainer(
#     #         limit_train_batches=trainer_config.limit_train_batches,
#     #         max_epochs=trainer_config.max_epochs,
#     #         callbacks=[
#     #             EarlyStopping(
#     #                 monitor=trainer_config.early_stopping.monitor, 
#     #                 mode=trainer_config.early_stopping.mode, 
#     #                 patience=unet_config.patience * 2
#     #             ),
#     #             ModelCheckpoint(
#     #                 dirpath=trainer_config.model_checkpoint.dirpath,
#     #                 filename=filename,
#     #                 save_top_k=trainer_config.model_checkpoint.save_top_k, 
#     #                 monitor=trainer_config.model_checkpoint.monitor,
#     #             )
#     #         ],
#     #         precision='16-mixed',
#     #         gradient_clip_val=0.5,
#     #         devices=[0]
#     #     )

#     #     trainer.fit(model, datamodule=datamodule)

#     # else:
#     #     #TODO: Add argsparse
#     #     load_as_lightning_module = True #False
#     #     load_as_pytorch_module = False #True

#     #     if load_as_lightning_module:
#     #         unet_config    = OmegaConf.load('../../MedImSeg-Lab24/configs/monai_unet.yaml')
#     #         unet = UNet(
#     #             spatial_dims=unet_config.spatial_dims,
#     #             in_channels=unet_config.in_channels,
#     #             out_channels=unet_config.out_channels,
#     #             channels=[unet_config.n_filters_init * 2 ** i for i in range(unet_config.depth)],
#     #             strides=[2] * (unet_config.depth - 1),
#     #             num_res_units=4
#     #         )
            
#     #         model = LightningSegmentationModel.load_from_checkpoint(
#     #             args.checkpoint_path,
#     #             map_location=torch.device("cpu"),
#     #             model=unet,
#     #             binary_target=True if unet_config.out_channels == 1 else False,
#     #             lr=unet_config.lr,
#     #             patience=unet_config.patience,
#     #             cfg=cfg
#     #         )

#     #     elif load_as_pytorch_module:
#     #         checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
#     #         model_state_dict = checkpoint['state_dict']
#     #         model_state_dict = {k.replace('model.model.', 'model.'): v for k, v in model_state_dict.items() if k.startswith('model.')}
#     #         model_config = checkpoint['hyper_parameters']['cfgs']

#     #         print(model_config)

#     #         model = UNet(
#     #             spatial_dims=model_config['unet']['spatial_dims'],
#     #             in_channels=model_config['unet']['in_channels'],
#     #             out_channels=model_config['unet']['out_channels'],
#     #             channels=[model_config['unet']['n_filters_init'] * 2 ** i for i in range(model_config['unet']['depth'])],
#     #             strides=[2] * (model_config['unet']['depth'] - 1),
#     #             num_res_units=4
#     #         )

#     #         model.load_state_dict(model_state_dict)
    
#     # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     # model = model.to(device)

#     # Getting the most uncertainty features
#     # datamodule.setup(stage='fit')
#     # train_idx = np.arange(len(datamodule.mnm_train))
#     # clue_sampler = CLUESampling(dset=datamodule.mnm_train,
#     #                             train_idx=train_idx, 
#     #                             model=model, 
#     #                             device=device, 
#     #                             args=cfg)
#     # # Getting centroids / nearest points to centroids
#     # nearest_idx = clue_sampler.query(n=args.n)
#     # selected_samples = [datamodule.mnm_train[i] for i in nearest_idx]
    
#     # augmented_centroids = [transform(sample) for sample in selected_samples]
#     # Getting results BEFORE using CLUE
#     import matplotlib.pyplot as plt

#     # Убедитесь, что у вас есть доступ к данным из тестового загрузчика
#     datamodule.setup(stage='test')
#     test_loader = datamodule.test_dataloader()

#     # Отображение первых нескольких изображений из тестового набора
#     num_images_to_plot = 1  # Укажите, сколько изображений вы хотите отрисовать

#     # Итерация по батчам
#     for batch_idx, batch in enumerate(test_loader):
#         images, labels = batch['input'], batch['index']  # Предполагается, что даталоадер возвращает (images, labels)
        
#         # Ограничение количества изображений
#         for i in range(min(num_images_to_plot, images.size(0))):
#             img = images[i]  # Получение одного изображения
            
#             # Перевод изображения из тензора в numpy (если необходимо)
#             if img.shape[0] == 1:  # Для черно-белых изображений
#                 img = img.squeeze(0)  # Убираем канал (C, H, W) -> (H, W)
#             else:  # Для RGB-изображений
#                 img = img.permute(1, 2, 0)  # Меняем оси (C, H, W) -> (H, W, C)
            
#             img = img.cpu().numpy()  # Переводим в numpy
            
#             # Отрисовка
#             plt.figure()
#             plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
#             plt.title(f'Label: {labels[i].item()}')  # Если есть метки, добавляем их в заголовок
#             plt.axis('off')
#             plt.show()

#         break  # Останавливаемся после одного батча, если нужно только несколько изображений




import sys
import numpy as np
import torch
from datetime import datetime
from torchvision import transforms
from omegaconf import OmegaConf
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from monai.networks.nets import UNet
from clue import CLUESampling
import matplotlib.pyplot as plt

sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel
from torch.utils.data import Dataset

# TODO: Add weights and remove later
class MNMv2Subset(Dataset):
    def __init__(
        self,
        input,
        target,
        weight=None
    ):
        self.input = input
        self.target = target
        self.weight = weight

    def __len__(self):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        item = {
            "input": self.input[idx], 
            "target": self.target[idx]
        }
        if self.weight is not None:
            item["weight"] = self.weight[idx]
        return item

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def log_to_file(file_path, message):
    """Logs a message to a specified file."""
    with open(file_path, 'a') as log_file:
        log_file.write(message + '\n')

def save_image(image, label, output_dir, index):
    """Saves the visualized image to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"image_{index}_label_{label}.png")
    plt.imsave(file_path, image, cmap='gray' if len(image.shape) == 2 else None)

if __name__ == '__main__':
    # Log file initialization
    log_file_path = "/home/chopra/lab-git/MedImSeg-Lab24/src/results_log.txt"
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    log_to_file(log_file_path, "Starting the script execution.")

    mnmv2_config = OmegaConf.load('../../MedImSeg-Lab24/configs/mnmv2.yaml')
    unet_config = OmegaConf.load('../../MedImSeg-Lab24/configs/monai_unet.yaml')
    trainer_config = OmegaConf.load('../../MedImSeg-Lab24/configs/unet_trainer.yaml')

    log_to_file(log_file_path, "Loaded configuration files.")

    # Init datamodule
    datamodule = MNMv2DataModule(
        data_dir=mnmv2_config.data_dir,
        vendor_assignment=mnmv2_config.vendor_assignment,
        batch_size=mnmv2_config.batch_size,
        binary_target=mnmv2_config.binary_target,
        non_empty_target=mnmv2_config.non_empty_target,
        split_ratio=0.5
    )

    log_to_file(log_file_path, "Initialized data module.")

    # Display first few images from the test dataset
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    log_to_file(log_file_path, "Setup test data loader.")

    num_images_to_plot = 1  # Number of images to plot
    output_dir = "visualized_images"  # Directory to save visualized images
    log_to_file(log_file_path, f"Number of images to plot: {num_images_to_plot}.")

    # Filter out images with no segmentation mask from the test dataset
    filtered_inputs = []
    filtered_targets = []
    filtered_weights = [] if hasattr(next(iter(test_loader)), 'weight') else None

    for batch_idx, batch in enumerate(test_loader):
        images, targets = batch['input'], batch['target']  # Assuming dataloader returns (images, targets)
        log_to_file(log_file_path, f"Processing batch {batch_idx + 1}.")

        for i in range(images.size(0)):
            img = images[i]  # Get one image
            target = targets[i]  # Get corresponding target (segmentation mask)

            if target.sum() == 0:  # Check if the segmentation mask has no foreground
                log_to_file(log_file_path, f"Found image with no segmentation mask at batch {batch_idx + 1}, index {i}. Removing it.")
            else:
                filtered_inputs.append(img.unsqueeze(0))
                filtered_targets.append(target.unsqueeze(0))
                if filtered_weights is not None:
                    filtered_weights.append(batch['weight'][i].unsqueeze(0))

    # Combine filtered data back into tensors
    if filtered_inputs:
        filtered_inputs = torch.cat(filtered_inputs, dim=0)
        filtered_targets = torch.cat(filtered_targets, dim=0)
        if filtered_weights is not None:
            filtered_weights = torch.cat(filtered_weights, dim=0)

        log_to_file(log_file_path, f"Filtered dataset size: {filtered_inputs.size(0)} images.")
    else:
        log_to_file(log_file_path, "All images had no segmentation masks. No data left after filtering.")

    log_to_file(log_file_path, "Finished processing and filtering dataset.")