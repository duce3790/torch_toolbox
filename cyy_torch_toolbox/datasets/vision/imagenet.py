import os
import os.path
import shutil
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg

class TINYIMAGENET(VisionDataset):
    base_folder = 'tiny-imagenet-200/'
    def __init__(
        self,
        root: str,
        train: bool = True,split='train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.dataset_path = os.path.join(self.root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        # self.data = make_dataset(self.dataset_path, self.split, class_to_idx)
        self.data = make_dataset(self.dataset_path, train, class_to_idx)
        self.targets = [s[1] for s in self.data]

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)


    def __getitem__(self, index: int):
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
        
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
    
    def __len__(self) -> int:
        return len(self.data)

def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


# def make_dataset(root, dirname, class_to_idx):
def make_dataset(root, train, class_to_idx):
    images = []
    # dir_path = os.path.join(root, dirname)
    if train:
        dir_path = os.path.join(root, "train")
    else:
        dir_path = os.path.join(root, "val")

    # if dirname == 'train':
    if train:
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())
        print(data_info)

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        # print("cls_map",cls_map)
        # print("cls_map",class_to_idx)

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            for imgname_ in sorted(os.listdir(path)):
                path__ = os.path.join(path, imgname_)
                # print(class_to_idx[cls_map[imgname_]])
                item = (path__, class_to_idx[cls_map[imgname_]])
                images.append(item)
            # item = (path, class_to_idx[cls_map[imgname]])
            # images.append(item)

    return images
