import math
import numpy as np
from enum import Enum
from typing import ( 
    List, 
    Tuple,
    Optional,
    Dict,
    Union
)
import sys
from time import sleep, time
import threading
from multiprocessing import Event, Process, Queue
import logging
from threadpoolctl import threadpool_limits
from queue import Queue as thrQueue
import random
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F, InterpolationMode
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform, 
    MirrorTransform
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform, 
    ContrastAugmentationTransform, 
    GammaTransform
)
from batchgenerators.transforms.utility_transforms import (
    RemoveLabelTransform, 
    RenameTransform, 
    NumpyToTensor
)
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.local_transforms import (
    BrightnessGradientAdditiveTransform,
    LocalGammaTransform,
    LocalSmoothingTransform,
    LocalContrastTransform
)
from batchgenerators.transforms.abstract_transforms import (
    Compose,
    AbstractTransform
)
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import producer, results_loop
import lightning as L

from dataset import *



class PMRIDataModule(L.LightningDataModule):
    
    def __init__(
        self,
        data_dir: str,
        vendor_assignment: Dict[str, str],
        batch_size: int = 32,
        train_transforms: str = 'global_transforms',
    ):
        super().__init__()
        self.data_dir = data_dir
        self.vendor_assignment = vendor_assignment
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.transforms = Transforms(patch_size=[384, 384])

    def prepare_data(self):
        # nothing to prepare for now. Data already downloaded
        pass

    def setup(self, stage=None):
        if stage == 'fit':
            pmri_full = PMRIDataset(
                data_dir=self.data_dir,
                vendor=self.vendor_assignment['train'],
            )
            self.pmri_train, self.pmri_val = pmri_full.random_split(
                val_size=0.1,
            )

        if stage == 'test':
            self.pmri_test = PMRIDataset(
                data_dir=self.data_dir,
                vendor=self.vendor_assignment['test'],
            )

        if stage == 'predict':
            self.prmi_predict = PMRIDataset(
                data_dir=self.data_dir,
                vendor=self.vendor_assignment['test'],
            )


    def train_dataloader(self):
        pmri_train_loader = MultiImageSingleViewDataLoader(
            data=self.pmri_train,
            batch_size=self.batch_size,
            return_orig=False
        )
        
        return MultiThreadedAugmenter(
            data_loader=pmri_train_loader,
            transform=self.transforms.get_transforms(self.train_transforms),
            num_processes=4,
            num_cached_per_queue = 2, 
            seeds=None
        )
    
    
    def val_dataloader(self):
        return DataLoader(
            self.pmri_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4
        )

    
    def test_dataloader(self):
        return DataLoader(
            self.pmri_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4
        )
    

    def predict_dataloader(self):
        return DataLoader(
            self.pmri_predict,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4
        )
    


class MNMv2DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        vendor_assignment: dict,
        batch_size: int = 32,
        binary_target: bool = False,
        train_transforms: str = 'global_transforms',
        non_empty_target: bool = True,
        split_ratio: float = 0.0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.vendor_assignment = vendor_assignment  # Should be a dict with keys 'train', 'val', 'test'
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.binary_target = binary_target
        self.non_empty_target = non_empty_target
        self.split_ratio = split_ratio
        self.transforms = Transforms(patch_size=[256, 256])  # Assuming similar transform object as PMRIDataModule


    def prepare_data(self):
        # Data is already prepared; nothing to do
        pass


    def setup(self, 
              stage: str = None):
        if stage == 'fit' or stage is None:
            # Load full dataset for training
            mnm_full = MNMv2Dataset(
                data_dir=self.data_dir,
                vendor=self.vendor_assignment.get('train'),
                binary_target=self.binary_target,
                non_empty_target=self.non_empty_target,
                normalize=True,  # Always normalizing
            )
            # Split using MNMv2Dataset's custom method
            if self.split_ratio > 0:
                self.mnm_train, self.mnm_train2, self.mnm_val = mnm_full.random_split(val_size=0.1, split_ratio=self.split_ratio)
            else:
                self.mnm_train, self.mnm_val = mnm_full.random_split(val_size=0.1)

        if stage == 'test' or stage is None:
            self.mnm_test = MNMv2Dataset(
                data_dir=self.data_dir,
                vendor=self.vendor_assignment.get('test'),
                binary_target=self.binary_target,
                non_empty_target=self.non_empty_target,
                normalize=True,
            )

        if stage == 'predict' or stage is None:
            self.mnm_predict = MNMv2Dataset(
                data_dir=self.data_dir,
                vendor=self.vendor_assignment.get('predict', self.vendor_assignment.get('test')),
                binary_target=self.binary_target,
                non_empty_target=self.non_empty_target,
                normalize=True,
            )


    def train_dataloader(self):
        mnm_train_loader = MultiImageSingleViewDataLoader(
            data=self.mnm_train,
            batch_size=self.batch_size,
            return_orig=False
        )
        
        return MultiThreadedAugmenter(
            data_loader=mnm_train_loader,
            transform=self.transforms.get_transforms(self.train_transforms),
            num_processes=4,
            num_cached_per_queue=2,
            seeds=None
        )


    def val_dataloader(self):
        return DataLoader(
            self.mnm_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False  # Ensuring we do not drop the last batch
        )


    def test_dataloader(self):
        return DataLoader(
            self.mnm_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False  # Ensuring we do not drop the last batch
        )


    def predict_dataloader(self):
        return DataLoader(
            self.mnm_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False  # Ensuring we do not drop the last batch
        )





class SingleImageMultiViewDataLoader(SlimDataLoaderBase):
    """Single image multi view dataloader.
    
    Adapted from batchgenerator examples:
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/example_ipynb.ipynb
    """
    def __init__(
        self, 
        data: Dataset, 
        batch_size: int, 
        return_orig: str = True
    ):
        super(SingleImageMultiViewDataLoader, self).__init__(data, batch_size)
        self.return_orig = return_orig
    
    def generate_train_batch(self):
        # select single slice from dataset
        data = self._data[random.randrange(len(self._data))]
        # split into input and target, cast to np.float for batchgenerators
        img = data['input'].numpy().astype(np.float32)
        tar = data['target'][0].numpy().astype(np.float32)
        # copy along batch dimension
        img_batched = np.tile(img, (self.batch_size, 1, 1, 1))
        tar_batched = np.tile(tar, (self.batch_size, 1, 1, 1))
        # now construct the dictionary and return it. np.float32 cast because most networks take float
        out = {'data': img_batched, 
               'seg':  tar_batched}
        
        # if the original data is also needed, activate this flag to store it where augmentations
        # cant find it.
        if self.return_orig:
            out['data_orig']   = data['input'].unsqueeze(0)
            out['target_orig'] = data['target'].unsqueeze(0)
        
        return out
    
    

class MultiImageSingleViewDataLoader(SlimDataLoaderBase):
    """Multi image single view dataloader.
    
    Adapted from batchgenerator examples:
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/example_ipynb.ipynb
    """
    def __init__(
        self, 
        data: Dataset, 
        batch_size: int, 
        return_orig: str = True
    ):
        super(MultiImageSingleViewDataLoader, self).__init__(data, batch_size)
        # data is now stored in self._data.
        self.return_orig = return_orig
    
    def generate_train_batch(self):
        # get random subset from dataset of batch size length
        sample = torch.randint(0, len(self._data), size=(self.batch_size,))
        data   = self._data[sample]
        # split into input and target
        img    = data['input']
        tar    = data['target']
        
        #construct the dictionary and return it. np.float32 cast because most networks take float
        out = {'data': img.numpy().astype(np.float32), 
               'seg':  tar.numpy().astype(np.float32)}
        
        # if the original data is also needed, activate this flag to store it where augmentations
        # cant find it.
        if self.return_orig:
            out['data_orig']   = img
            out['target_orig'] = tar
        
        return out
    
    

class Transforms(object):
    """
    A container for organizing and accessing different sets of image transformation operations.

    This class defines four categories of transformations: 'io_transforms', 
    'global_nonspatial_transforms', 'global_transforms', and 'local_transforms'. Each category 
    contains a list of specific transform objects designed for image preprocessing in machine learning tasks.

    The 'io_transforms' are basic input/output operations, like renaming and format conversion.
    'global_nonspatial_transforms' apply transformations like noise addition or resolution simulation 
    that do not depend alter spatial information. 'global_transforms' include spatial transformations 
    like rotation and scaling. 'local_transforms' focus on localized changes to an image, such as adding 
    brightness gradients or local smoothing and are also non-spatial.

    Attributes:
        transforms (dict): A dictionary where keys are transform categories and values are lists of transform objects.

    Methods:
        get_transforms(arg: str): Retrieves a composed transform pipeline based on the specified category.

    Usage:
        >>> transforms = Transforms()
        >>> global_transforms = transforms.get_transforms('global_transforms')
    """
    
    def __init__(
        self,
        patch_size = [256, 256]
    ) -> None:
        
        io_transforms = [
            RemoveLabelTransform(
                output_key = 'seg', 
                input_key = 'seg',
                replace_with = 0,
                remove_label = -1
            ),
            RenameTransform(
                delete_old = True,
                out_key = 'target',
                in_key = 'seg'
            ),
            NumpyToTensor(
                keys = ['data', 'target'], 
                cast_to = 'float')    
        ]
       
        global_nonspatial_transforms = [
            SimulateLowResolutionTransform(
                order_upsample = 3, 
                order_downsample = 0, 
                channels = None, 
                per_channel = True, 
                p_per_channel = 0.5, 
                p_per_sample = 0.25, 
                data_key = 'data',
                zoom_range = (0.5, 1), 
                ignore_axes = None
            ),
            GaussianNoiseTransform(
                p_per_sample = 0.1, 
                data_key = 'data', 
                noise_variance = (0, 0.1), 
                p_per_channel = 1, 
                per_channel = False
            ),
        ] 
       
        global_transforms = [
            SpatialTransform(
                independent_scale_for_each_axis = False, 
                p_rot_per_sample = 0.2, 
                p_scale_per_sample = 0.2, 
                p_el_per_sample = 0.2, 
                data_key = 'data', 
                label_key = 'seg', 
                patch_size = np.array(patch_size), 
                patch_center_dist_from_border = None, 
                do_elastic_deform = False, 
                alpha = (0.0, 200.0), 
                sigma = (9.0, 13.0), 
                do_rotation = True, 
                angle_x = (-3.141592653589793, 3.141592653589793), 
                angle_y = (-0.0, 0.0), 
                angle_z = (-0.0, 0.0), 
                do_scale = True,
                scale = (0.7, 1.4), 
                border_mode_data = 'constant',
                border_cval_data = 0, 
                order_data = 3, 
                border_mode_seg = 'constant',
                border_cval_seg = -1, 
                order_seg = 1,
                random_crop = False,
                p_rot_per_axis = 1, 
                p_independent_scale_per_axis = 1
            ),
            GaussianBlurTransform(
                p_per_sample = 0.2, 
                different_sigma_per_channel = True, 
                p_per_channel = 0.5, 
                data_key = 'data', 
                blur_sigma = (0.5, 1.0), 
                different_sigma_per_axis = False, 
                p_isotropic = 0
            ),
            BrightnessMultiplicativeTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                multiplier_range = (0.75, 1.25), 
                per_channel = True
            ),
            ContrastAugmentationTransform(
                p_per_sample = 0.15, 
                data_key = 'data', 
                contrast_range = (0.75, 1.25), 
                preserve_range = True, 
                per_channel = True, 
                p_per_channel = 1
            ),
            GammaTransform(
                p_per_sample = 0.1,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = True
            ),
            GammaTransform(
                p_per_sample = 0.3,
                retain_stats = True, 
                per_channel = True, 
                data_key = 'data', 
                gamma_range = (0.7, 1.5), 
                invert_image = False
            ),
            MirrorTransform(
                p_per_sample = 1, 
                data_key = 'data', 
                label_key = 'seg', 
                axes = (0, 1)
            ),
        ]       
        
        
        local_transforms = [
            BrightnessGradientAdditiveTransform(
                scale=200, 
                max_strength=4, 
                p_per_sample=0.2, 
                p_per_channel=1
            ),
            LocalGammaTransform(
                scale=200, 
                gamma=(2, 5), 
                p_per_sample=0.2,
                p_per_channel=1
            ),
            LocalSmoothingTransform(
                scale=200,
                smoothing_strength=(0.5, 1),
                p_per_sample=0.2,
                p_per_channel=1
            ),
            LocalContrastTransform(
                scale=200,
                new_contrast=(1, 3),
                p_per_sample=0.2,
                p_per_channel=1
            ),
        ]
        
        self.transforms = {
            'io_transforms': io_transforms,
            'global_nonspatial_transforms': global_nonspatial_transforms + io_transforms,
            'global_transforms': global_transforms + io_transforms,
            'local_transforms': global_nonspatial_transforms + local_transforms + io_transforms,
            'local_val_transforms': local_transforms + io_transforms,
            'all_transforms': local_transforms + global_transforms + io_transforms,
        }


    def get_transforms(
        self, 
        arg: str
    ) -> AbstractTransform:
        if arg in self.transforms:
            return Compose(self.transforms[arg])

        elif 'global_without' in arg:
            print(arg.split('_')[-1])
            missing_transform = arg.split('_')[-1]
            return Compose(list(
                filter(
                    lambda x: x.__class__.__name__ != missing_transform,
                    self.transforms['global_transforms']
                )
            ))



def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Zoom":
        scale = (100 - magnitude) / 100
        img = F.resized_crop(
            img,
            top=int(img.shape[-2] // 2 - (img.shape[-2] * scale) // 2),
            left=int(img.shape[-1] // 2 - (img.shape[-1] * scale) // 2),
            height=int(img.shape[-2] * scale),
            width=int(img.shape[-1] * scale),
            size=img.shape[-2:],
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img



class RandAugmentWithLabels(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_max_ops (int): Maximum number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_max_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_max_ops = num_max_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(
        self, num_bins: int, image_size: List[int]
    ) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins),
                True,
            ),
            "TranslateY": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins),
                True,
            ),
            "Rotate": (torch.linspace(0.0, 45.0, num_bins), True),
            "Zoom": (torch.linspace(0.0, 50.0, num_bins), False),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_trace = []

        # for _ in range(torch.randint(0, self.num_max_ops, (1,))):
        for _ in range(random.randint(0, self.num_max_ops)):
            op_meta = self._augmentation_space(
                self.num_magnitude_bins, F.get_image_size(img)
            )
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = (
                float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            )
            magnitude = torch.rand((1,)).item() * magnitude
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(
                img, op_name, magnitude, interpolation=self.interpolation, fill=fill
            )
            if op_name in [
                "Identity",
                "ShearX",
                "ShearY",
                "TranslateX",
                "TranslateY",
                "Rotate",
                "Zoom",
            ]:
                op_trace.append((op_name, magnitude, InterpolationMode.NEAREST, fill))
                # op_trace.append((op_name, magnitude, self.interpolation, fill))
                # print("label transformed")
                # lbl = _apply_op(lbl, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img, op_trace

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_max_ops={self.num_max_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

        

def volume_collate(batch: List[dict]) -> dict:
    return batch[0]


     
def slice_selection(
    dataset: Dataset, 
    indices: Tensor,
    n_cases: int = 10,
    verbose: bool = False
) -> Tensor:
    
    slices = dataset.__getitem__(indices)['input']
    if n_cases > len(slices):
        if verbose:
            print(f"Number of cases ({n_cases}) is larger than number of data points ({len(slices)}). Setting n_classes = len(slices)")
        n_cases = len(slices)

    
    kmeans_in = slices.reshape(len(indices), -1)
    kmeans = KMeans(
        n_clusters=n_cases,
        init='k-means++',
        n_init=1,
        algorithm='elkan',
        ).fit(kmeans_in)
    idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, kmeans_in)
    return indices[idx]



def dataset_from_indices(
    dataset: Dataset, 
    indices: Tensor
) -> DataLoader:
    
    data = dataset.__getitem__(indices)
    
    class CustomDataset(Dataset):
        
        def __init__(self, input: Tensor, labels: Tensor, 
                     voxel_dim: Tensor):
            self.input = input
            self.labels = labels
            self.voxel_dim = voxel_dim
            
        def __getitem__(self, idx):
            return {'input': self.input[idx],
                    'target': self.labels[idx],
                    'voxel_dim': self.voxel_dim[idx]}
        
        def __len__(self):
            return self.input.size(0)
        
    return CustomDataset(*data.values())



@torch.no_grad()
def get_subset(
    dataset: Dataset,
    model: nn.Module,
    criterion: nn.Module,
    device: str = 'cuda:0',
    fraction: float = 0.1, 
    n_cases: int = 10, 
    part: str = "tail",
    batch_size: int = 1,
    verbose: bool = False,
    cache_path: Optional[str] = None
) -> Dataset:
    """Selects a subset of the otherwise very large CC-359 Dataset, which
        - is in the bottom/top fraction w.r.t. to a criterion and model
        - contains n_cases, drawn to be divers w.r.t. to the input space and
          defined as k_means cluster centers, one for each case.
    """
    # TODO: cache subset indices for subsequent runs. Cache based on all
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            indices_selection = pickle.load(f)
        print(f"A cached subset of the {cache_path}")
        return dataset_from_indices(dataset, indices_selection)
    
    # factors that influence selection, i.e. model, criterion, function params etc
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
#     if n_cases < len(dataset):
#         print("n_cases > len(dataset). Setting n_cases = len(dataset)")
#         n_cases = len(dataset)
    
    # collect evaluation per slice and cache
    print(f'{dataset.folder}:')
    start = time()
    assert criterion.reduction == 'none'
    model.eval()
    loss_list = []
    for batch in dataloader:
        input_  = batch['input'].to(device)
        target  = batch['target'].to(device)
        net_out = model(input_)
        loss    = criterion(net_out, target).view(input_.shape[0], -1).mean(1).cpu()
        loss_list.append(loss)
        
    loss_tensor = torch.cat(loss_list)
    assert len(loss_tensor.shape) == 1
    print(f'It took {(time() - start):.2f} seconds to collect all dice scores')
    start = time()
    # get indices from loss function values
    indices = torch.argsort(loss_tensor, descending=True)
    # set some params for slicing
    len_    = len(dataset)
    devisor = int(1 / fraction)
    # Chunk dataset for either tail or head part based on func args
    if part == 'tail':
        indices = indices[:len_ // devisor]
    elif part == 'head':
        indices = indices[-len_ // devisor:]
    print(f'It took {(time() - start):.2f} seconds to sort them')
    # select slices within fraction of dataset based on kmeans
    indices_selection = slice_selection(
        dataset,
        indices, 
        n_cases=n_cases,
        verbose=verbose
    )
    print(f'It took {(time() - start):.2f} seconds to determine center clusters.\n')
    # build dataset from selected slices and return
    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(indices_selection, f)
        print(f"The saved Subset in the cache: {cache_path}")
        
    subset = dataset_from_indices(dataset, indices_selection)

    return subset



class MultiThreadedAugmenter(object):
    """ 
    Adapted from batchgenerators, see https://github.com/MIC-DKFZ/batchgenerators/
    Changed user_api from blas to openmp in class method _start in threadpool_limits.
    Otherwise it doesn't work with docker!
    
    Makes your pipeline multi threaded. Yeah!
    If seeded we guarantee that batches are retunred in the same order and with the same augmentation every time this
    is run. This is realized internally by using une queue per worker and querying the queues one ofter the other.
    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure
        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)
        num_processes (int): number of processes
        num_cached_per_queue (int): number of batches cached per process (each process has its own
        multiprocessing.Queue). We found 2 to be ideal.
        seeds (list of int): one seed for each worker. Must have len(num_processes).
        If None then seeds = range(num_processes)
        pin_memory (bool): set to True if all torch tensors in data_dict are to be pinned. Pytorch only.
        timeout (int): How long do we wait for the background workers to do stuff? If timeout seconds have passed and
        self.__get_next_item still has not gotten an item from the workers we will perform a check whether all
        background workers are still alive. If all are alive we wait, if not we set the abort flag.
        wait_time (float): set this to be lower than the time you need per iteration. Don't set this to 0,
        that will come with a performance penalty. Default is 0.02 which will be fine for 50 iterations/s
    """

    def __init__(self, data_loader, transform, num_processes, num_cached_per_queue=2, seeds=None, pin_memory=False,
                 timeout=10, wait_time=0.02):
        self.timeout = timeout
        self.pin_memory = pin_memory
        self.transform = transform
        if seeds is not None:
            assert len(seeds) == num_processes
        else:
            seeds = [None] * num_processes
        self.seeds = seeds
        self.generator = data_loader
        self.num_processes = num_processes
        self.num_cached_per_queue = num_cached_per_queue
        self._queues = []
        self._processes = []
        self._end_ctr = 0
        self._queue_ctr = 0
        self.pin_memory_thread = None
        self.pin_memory_queue = None
        self.abort_event = Event()
        self.wait_time = wait_time
        self.was_initialized = False

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __get_next_item(self):
        item = None

        while item is None:
            if self.abort_event.is_set():
                self._finish()
                raise RuntimeError("One or more background workers are no longer alive. Exiting. Please check the "
                                   "print statements above for the actual error message")

            if not self.pin_memory_queue.empty():
                item = self.pin_memory_queue.get()
            else:
                sleep(self.wait_time)

        return item

    def __next__(self):
        if not self.was_initialized:
            self._start()

        try:
            item = self.__get_next_item()

            while isinstance(item, str) and (item == "end"):
                self._end_ctr += 1
                if self._end_ctr == self.num_processes:
                    self._end_ctr = 0
                    self._queue_ctr = 0
                    logging.debug("MultiThreadedGenerator: finished data generation")
                    raise StopIteration

                item = self.__get_next_item()

            return item

        except KeyboardInterrupt:
            logging.error("MultiThreadedGenerator: caught exception: {}".format(sys.exc_info()))
            self.abort_event.set()
            self._finish()
            raise KeyboardInterrupt

    def _start(self):
        if not self.was_initialized:
            self._finish()
            self.abort_event.clear()

            logging.debug("starting workers")
            self._queue_ctr = 0
            self._end_ctr = 0

            if hasattr(self.generator, 'was_initialized'):
                self.generator.was_initialized = False

            with threadpool_limits(limits=1, user_api="openmp"):
                for i in range(self.num_processes):
                    self._queues.append(Queue(self.num_cached_per_queue))
                    self._processes.append(Process(target=producer, args=(
                        self._queues[i], self.generator, self.transform, i, self.seeds[i], self.abort_event)))
                    self._processes[-1].daemon = True
                    self._processes[-1].start()

            if torch is not None and torch.cuda.is_available():
                gpu = torch.cuda.current_device()
            else:
                gpu = None

            # more caching = more performance. But don't cache too much or your RAM will hate you
            self.pin_memory_queue = thrQueue(max(3, self.num_cached_per_queue * self.num_processes // 2))

            self.pin_memory_thread = threading.Thread(target=results_loop, args=(
                self._queues, self.pin_memory_queue, self.abort_event, self.pin_memory, gpu, self.wait_time,
                self._processes))

            self.pin_memory_thread.daemon = True
            self.pin_memory_thread.start()

            self.was_initialized = True
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but it has already been "
                          "initialized previously")

    def _finish(self, timeout=10):
        self.abort_event.set()

        start = time()
        while self.pin_memory_thread is not None and self.pin_memory_thread.is_alive() and start + timeout > time():
            
            sleep(0.2)

        if len(self._processes) != 0:
            logging.debug("MultiThreadedGenerator: shutting down workers...")
            [i.terminate() for i in self._processes]

            for i, p in enumerate(self._processes):
                self._queues[i].close()
                self._queues[i].join_thread()

            self._queues = []
            self._processes = []
            self._queue = None
            self._end_ctr = 0
            self._queue_ctr = 0

            del self.pin_memory_queue
        self.was_initialized = False

    def restart(self):
        self._finish()
        self._start()

    def __del__(self):
        logging.debug("MultiThreadedGenerator: destructor was called")
        self._finish()
    
