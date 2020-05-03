"""
Simplified code from https://github.com/mprzewie/gmms_inpainting/tree/master/inpainting/datasets
"""

import dataclasses as dc
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from torchvision.datasets import MNIST, CelebA
from torchvision import transforms as tr

# annotations in mask
KNOWN = 1  # known data
UNKNOWN_LOSS = 0  # unknown (hidden) data, from inpainting which we calculate loss
UNKNOWN_NO_LOSS = -1  # truly unknown data, which are assumed to be missing from the dataset.


@dc.dataclass(frozen=True)
class RandomRectangleMaskConfig:
    """A config for generating random masks"""

    value: int
    height: int
    width: int
    height_ampl: int = 0
    width_ampl: int = 0

    def generate_on_mask(
            self, mask: np.ndarray, copy: bool = True, seed: int = None
    ) -> np.ndarray:
        """
        Args:
            mask: [h, w]
            copy:
        """
        if seed is not None:
            np.random.seed(seed)
        mask = mask.copy() if copy else mask
        m_height = self.height + np.random.randint(
            -self.height_ampl, self.height_ampl + 1
        )
        m_width = self.width + np.random.randint(-self.width_ampl, self.width_ampl + 1)
        tot_height, tot_width = mask.shape[1:3]

        m_y = (
            np.random.randint(0, tot_height - m_height) if m_height < tot_height else 0
        )
        m_x = np.random.randint(0, tot_width - m_width) if m_width < tot_width else 0
        mask[..., m_y: m_y + m_height, m_x: m_x + m_width] = self.value
        return mask


def random_mask_fn(
        mask_configs: Sequence[RandomRectangleMaskConfig], deterministic: bool = True
):
    def tensor_to_tensor_with_random_mask(image_tensor: torch.Tensor):
        mask = np.ones_like(image_tensor.numpy())
        for mc in mask_configs:
            mask = mc.generate_on_mask(
                mask,
                seed=mc.value + int((image_tensor * 255).sum().item())
                if deterministic
                else None,
            )
        return image_tensor, torch.tensor(mask).float()

    return tensor_to_tensor_with_random_mask


DEFAULT_MNIST_MASK_CONFIGS = (
    RandomRectangleMaskConfig(UNKNOWN_LOSS, 14, 14, 0, 0),
    # RandomRectangleMaskConfig(UNKNOWN_NO_LOSS, 8, 8, 2, 2),
)


def mnist_train_val_datasets(
        save_path: Path = Path("data"),
        mask_configs: Sequence[RandomRectangleMaskConfig] = DEFAULT_MNIST_MASK_CONFIGS,
        ds_type: MNIST = MNIST,
        resize_size: Tuple[int, int] = (28, 28),
        with_mask: bool = True

) -> Tuple[MNIST, MNIST]:
    base_transform = tr.Compose([tr.Resize(resize_size), tr.ToTensor()])

    train_transform = [
        base_transform
    ]


    if with_mask:
        train_transform.extend([
            tr.Lambda(
                random_mask_fn(mask_configs=mask_configs, deterministic=True)
            ),
            tr.Lambda(
                lambda x_j: (x_j[0].reshape(-1), x_j[1].reshape(-1))
            )
        ])
    else:
        train_transform.append(
            tr.Lambda(
                lambda x: x.reshape(-1)
            )
        )

    train_transform = tr.Compose(
        train_transform
    )

    val_transform = [base_transform]

    if with_mask:
        val_transform.extend([
            tr.Lambda(
                random_mask_fn(
                    mask_configs=[
                        m
                        for m in mask_configs
                        if m.value == UNKNOWN_LOSS or m.value == KNOWN
                    ],  # only the mask which will be inpainted
                    deterministic=True,
                )
            ),
            tr.Lambda(
                lambda x_j: (x_j[0].reshape(-1), x_j[1].reshape(-1))
            )
        ])
    else:
        val_transform.append(
            tr.Lambda(
                lambda x: x.reshape(-1)
            )
        )
    

    val_transform = tr.Compose(val_transform)

    ds_train = ds_type(save_path, train=True, download=True, transform=train_transform)
    ds_val = ds_type(save_path, train=False, download=True, transform=val_transform)

    return ds_train, ds_val


DEFAULT_CELEBA_MASK_CONFIGS = (
    RandomRectangleMaskConfig(UNKNOWN_LOSS, 32, 32, 0, 0),
    # RandomRectangleMaskConfig(UNKNOWN_NO_LOSS, 15, 15, 0, 0),
)


def celeba_train_val_datasets(
        save_path: Path = Path("/home/mprzewiezlikowski/uj/.data/"),
        mask_configs: Sequence[RandomRectangleMaskConfig] = DEFAULT_CELEBA_MASK_CONFIGS,
        resize_size: Tuple[int, int] = (120, 120),
        crop_size: Tuple[int, int] = (64, 64),
        with_mask: bool = True
) -> Tuple[CelebA, CelebA]:
    base_transform = tr.Compose(
        [
            tr.Lambda(lambda im: im.convert("RGB")),
            tr.Resize(resize_size),
            tr.CenterCrop(crop_size),
            tr.ToTensor(),
        ]
    )

    train_transform = [
        base_transform
    ]


    if with_mask:
        train_transform.extend([
            tr.Lambda(
                random_mask_fn(mask_configs=mask_configs, deterministic=True)
            ),
            tr.Lambda(
                lambda x_j: (x_j[0].reshape(-1), x_j[1].reshape(-1))
            )
        ])
    else:
        train_transform.append(
            tr.Lambda(
                lambda x: x.reshape(-1)
            )
        )

    train_transform = tr.Compose(
        train_transform
    )

    val_transform = [base_transform]

    if with_mask:
        val_transform.extend([
            tr.Lambda(
                random_mask_fn(
                    mask_configs=[
                        m
                        for m in mask_configs
                        if m.value == UNKNOWN_LOSS or m.value == KNOWN
                    ],  # only the mask which will be inpainted
                    deterministic=True,
                )
            ),
            tr.Lambda(
                lambda x_j: (x_j[0].reshape(-1), x_j[1].reshape(-1))
            )
        ])
    else:
        val_transform.append(
            tr.Lambda(
                lambda x: x.reshape(-1)
            )
        )
    

    val_transform = tr.Compose(val_transform)

    ds_train = CelebA(
        save_path, split="train", download=True, transform=train_transform
    )
    ds_val = CelebA(save_path, split="valid", download=True, transform=val_transform)

    return ds_train, ds_val
