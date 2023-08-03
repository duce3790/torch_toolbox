import os
import os.path
import shutil
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


class IMAGENET(VisionDataset):
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        # self.train = train  # training set or test set

        # if download:
        #     self.download()

        # if not self._check_exists():
        #     raise RuntimeError(
        #         "Dataset not found." + " You can use download=True to download it"
        #     )

        # if self.train:
        #     data_file = self.training_file
        # else:
        #     data_file = self.test_file
        # self.data, self.targets, self.users = torch.load(
        #     os.path.join(self.processed_folder, data_file)
        # )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], FEMNIST.__relabel_class(self.targets[index])

        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="F")

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return img, target, {"user": self.users[index]}

    def __len__(self) -> int:
        return len(self.data)


