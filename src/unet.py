"""
- PAPER
https://arxiv.org/pdf/2008.07357.pdf
- CODE
https://github.com/kechua/DART20/blob/master/damri/model/unet.py
"""
from __future__ import annotations

import warnings
from tqdm import tqdm
from collections.abc import Sequence
from typing import (
    Dict,
    Tuple,
    List,
    Union
)
from omegaconf import OmegaConf
import torch
from torch import nn
import lightning as L
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.layers.factories import Act, Norm
from monai.utils import (
    look_up_option, 
    SkipMode
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.blocks import ResidualUnit, Convolution



__all__ = ["UNet", "Unet"]


class LightningSegmentationModel(L.LightningModule):
    def __init__(self, cfg: OmegaConf = None):
        super().__init__()
        # this would save the model as hyperparameter, not desired!
        self.cfg = cfg
        self.model = self.get_unet(cfg.unet_config)
        
        self.final_train_loss = None
        self.final_train_acc = None
        self.final_val_loss = None
        self.final_val_acc = None
        
        self.save_hyperparameters({
            'cfg': cfg,
        })
        
        self.lr = cfg.lr
        self.patience = cfg.patience
        self.cfg = cfg
        self.loss = DiceCELoss(
            softmax=False if cfg.binary_target else True,
            sigmoid=True if cfg.binary_target else False,
            to_onehot_y=False if cfg.binary_target else True,
        )
        self.dsc = DiceMetric(include_background=False, reduction="none")
        self.IoU = MeanIoU(include_background=False, reduction="none")

    def get_unet(self, unet_config):
        return UNet(
            spatial_dims=unet_config.spatial_dims,
            in_channels=unet_config.in_channels,
            out_channels=unet_config.out_channels,
            channels=[unet_config.n_filters_init * 2 ** i for i in range(unet_config.depth)],
            strides=[2] * (unet_config.depth - 1),
            num_res_units=4
        )
    
    def forward(self, inputs, with_emb=False):        
        if with_emb:
            # For CLUE we need embedings from the penultimate layer
            emb = self.model.model[:-1](inputs)
            pred = self.model.model[-1](emb)
            return pred, emb
        else:
            return self.model(inputs)


    def test_model(self, data_loader, device):
        self.model.eval()

        total_loss = 0
        total_dsc = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch['input'] = batch['input'].to(device)
                batch['target'] = batch['target'].to(device)

                result = self.test_step(batch, batch_idx)
                
                total_loss += result['loss'].item()
                total_dsc += result['dsc'].item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_dsc = total_dsc / num_batches

        # self.log_dict({
        #     'avg_test_loss': avg_loss,
        #     'avg_test_dsc': avg_dsc,
        # })

        print(f"Test Results - Average Loss: {avg_loss:.4f}, Average Dice Score: {avg_dsc:.4f}")

        return {
            'avg_loss': avg_loss,
            'avg_dsc': avg_dsc
        }


    def training_step(self, batch, batch_idx):
        input = batch['data']
        target = batch['target']
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0.5) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean()

        self.log_dict({
            'train_loss': loss,
            'train_dsc': dsc,
        })
        return {
            'loss': loss,
            'dsc': dsc
        }
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean()

        self.log_dict({
            'val_loss': loss,
            'val_dsc': dsc,
        })
        return {
            'loss': loss,
        }
    
    def on_train_epoch_end(self):
        # Сохраняем последние значения train_loss и train_acc
        self.final_train_loss = self.trainer.callback_metrics["train_loss"]
        self.final_train_acc = self.trainer.callback_metrics["train_dsc"]
    
    def on_validation_epoch_end(self):
        # Сохраняем последние значения val_loss и val_acc
        self.final_val_loss = self.trainer.callback_metrics["val_loss"]
        self.final_val_acc = self.trainer.callback_metrics["val_dsc"]

    def test_step(self, batch, batch_idx=None, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean()


        self.log_dict({
            'test_loss': loss,
            'test_dsc': dsc,
        })
        return {
            'loss': loss,
            'dsc': dsc
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        outputs = self(input)
        return outputs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience),
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }
    
    # def on_train_epoch_end(self):
    #     avg_loss = torch.stack(self.train_losses).mean()
    #     avg_dsc = torch.stack(self.train_dscs).mean()

    #     self.log("avg_train_loss", avg_loss, prog_bar=True)
    #     self.log("avg_train_dsc", avg_dsc, prog_bar=True)

    # def on_validation_epoch_end(self):
    #     avg_loss = self.trainer.callback_metrics.get("val_loss", torch.tensor(0.0))
    #     avg_dsc = self.trainer.callback_metrics.get("val_dsc", torch.tensor(0.0))

    #     self.log("avg_val_loss", avg_loss, prog_bar=True)
    #     self.log("avg_val_dsc", avg_dsc, prog_bar=True)


# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




class UNet(nn.Module):
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        class Swivel(nn.Module):
            def __init__(self):
                super(Swivel, self).__init__()
                self.swivel = nn.Identity()
            def forward(self, x):
                return self.swivel(x)

        return nn.Sequential(
            self._get_down_layer(in_channels, out_channels, 1, False),
            Swivel()
        )

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Convolution | nn.Sequential

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


Unet = UNet


class SkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::

        --+--submodule--o--
          |_____________|

    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(self, submodule, dim: int = 1, mode: str | SkipMode = "cat") -> None:
        """

        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
            mode: ``"cat"``, ``"add"``, ``"mul"``. defaults to ``"cat"``.
        """
        super().__init__()
        self.submodule = submodule
        self.dim = dim
        self.mode = look_up_option(mode, SkipMode).value
        self.swivel = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)
        x = self.swivel(x)
        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(x, y)
        if self.mode == "mul":
            return torch.mul(x, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")

