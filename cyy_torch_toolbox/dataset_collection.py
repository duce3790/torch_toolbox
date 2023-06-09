import copy
import json
import os
import threading
from typing import Callable

import torch
import torchvision
from cyy_naive_lib.fs.ssd import is_ssd
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reflection import get_kwarg_names
from cyy_naive_lib.storage import get_cached_data

from cyy_torch_toolbox.dataset import (DictDataset,
                                       convert_iterable_dataset_to_map,
                                       sub_dataset)
from cyy_torch_toolbox.dataset_repository import get_dataset_constructors
from cyy_torch_toolbox.dataset_transform.transforms import (Transforms,
                                                            replace_target)
from cyy_torch_toolbox.dataset_transform.transforms_factory import \
    add_transforms
from cyy_torch_toolbox.dataset_util import (DatasetUtil, TextDatasetUtil,
                                            VisionDatasetUtil)
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       TransformType)


class DatasetCollection:
    def __init__(
        self,
        training_dataset: torch.utils.data.Dataset,
        validation_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        dataset_type: DatasetType,
        name: str,
    ):
        self.__name = name
        self._datasets: dict[MachineLearningPhase, torch.utils.data.Dataset] = {}
        self._datasets[MachineLearningPhase.Training] = training_dataset
        self._datasets[MachineLearningPhase.Validation] = validation_dataset
        if test_dataset is not None:
            self._datasets[MachineLearningPhase.Test] = test_dataset
        self.__dataset_type = dataset_type
        self.__transforms: dict[MachineLearningPhase, Transforms] = {}
        for phase in MachineLearningPhase:
            self.__transforms[phase] = Transforms()
        self.__old_transforms: dict[MachineLearningPhase, Transforms] = {}

        # duce add
        self._datasets[MachineLearningPhase.Test] = validation_dataset
        self._datasets[MachineLearningPhase.Validation] = training_dataset


    @property
    def dataset_type(self):
        return self.__dataset_type

    def transform_dataset(
        self, phase: MachineLearningPhase, transformer: Callable
    ) -> None:
        dataset = self.get_dataset(phase)
        dataset_util = self.get_dataset_util(phase)
        self._datasets[phase] = transformer(dataset, dataset_util, phase)

    def foreach_dataset(self):
        for phase in MachineLearningPhase:
            if self.has_dataset(phase=phase):
                yield self.get_dataset(phase=phase)

    def transform_all_datasets(self, transformer: Callable) -> None:
        for phase in MachineLearningPhase:
            self.transform_dataset(phase, transformer)

    def has_dataset(self, phase: MachineLearningPhase) -> bool:
        return phase in self._datasets

    def remove_dataset(self, phase: MachineLearningPhase) -> None:
        get_logger().debug("remove dataset %s", phase)
        self._datasets.pop(phase, None)

    def get_dataset(self, phase: MachineLearningPhase) -> torch.utils.data.Dataset:
        return self._datasets[phase]

    def get_training_dataset(self) -> torch.utils.data.Dataset:
        return self.get_dataset(MachineLearningPhase.Training)

    def get_transforms(self, phase, original: bool = False) -> Transforms:
        if original and phase in self.__old_transforms:
            return self.__old_transforms[phase]
        return self.__transforms[phase]

    def get_original_dataset(
        self, phase: MachineLearningPhase
    ) -> torch.utils.data.Dataset:
        dataset = self.get_dataset(phase)
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def get_dataset_util(
        self, phase: MachineLearningPhase = MachineLearningPhase.Test
    ) -> DatasetUtil:
        match self.dataset_type:
            case DatasetType.Vision:
                class_name = VisionDatasetUtil
            case DatasetType.Text:
                class_name = TextDatasetUtil
            case _:
                class_name = DatasetUtil
        return class_name(
            dataset=self.get_dataset(phase),
            transforms=self.__transforms[phase],
            name=self.name,
        )

    def clear_transform(self, key, phases=None):
        for phase in MachineLearningPhase:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].clear(key)

    def append_transform(self, transform, key, phases=None):
        for phase in MachineLearningPhase:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].append(key, transform)

    def transforms_cached(self, phase):
        return phase in self.__old_transforms

    def cache_transforms(self, phase: MachineLearningPhase, device=None) -> None:
        if self.transforms_cached(phase=phase):
            return
        assert self.has_dataset(phase=phase)
        dataset = self.get_dataset(phase=phase)
        transforms = self.get_transforms(phase=phase)
        transformed_dataset, new_transforms = transforms.cache_transforms(
            dataset, device
        )
        self._datasets[phase] = DictDataset(transformed_dataset)
        self.__old_transforms[phase] = self.__transforms[phase]
        self.__transforms[phase] = new_transforms
        get_logger().debug("new training transforms are %s", new_transforms)

    @property
    def name(self) -> str:
        return self.__name

    def transform_text(self, phase, text):
        return self.get_transforms(phase).transform_text(text)

    __dataset_root_dir: str = os.path.join(os.path.expanduser("~"), "pytorch_dataset")
    lock = threading.RLock()

    @classmethod
    def get_dataset_root_dir(cls):
        with cls.lock:
            return os.getenv("pytorch_dataset_root_dir", cls.__dataset_root_dir)

    @classmethod
    def set_dataset_root_dir(cls, root_dir: str):
        with cls.lock:
            cls.__dataset_root_dir = root_dir

    @classmethod
    def __get_dataset_dir(cls, name: str):
        dataset_dir = os.path.join(cls.get_dataset_root_dir(), name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        try:
            if not is_ssd(dataset_dir):
                get_logger().warning(
                    "dataset %s is not on a SSD disk: %s", name, dataset_dir
                )
        except BaseException:
            pass
        return dataset_dir

    @classmethod
    def __get_dataset_cache_dir(
        cls,
        name: str,
        phase: MachineLearningPhase = None,
    ) -> str:
        cache_dir = os.path.join(cls.__get_dataset_dir(name), ".cache")
        if phase is not None:
            cache_dir = os.path.join(cache_dir, str(phase))
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @classmethod
    def create(
        cls,
        name: str,
        dataset_type: DatasetType,
        dataset_constructor,
        dataset_kwargs: dict = None,
    ) -> tuple:
        constructor_kwargs = get_kwarg_names(dataset_constructor)
        dataset_kwargs_fun = cls.__prepare_dataset_kwargs(
            name, constructor_kwargs, dataset_kwargs
        )
        training_dataset = None
        validation_dataset = None
        test_dataset = None

        for phase in MachineLearningPhase:
            while True:
                try:
                    processed_dataset_kwargs = dataset_kwargs_fun(
                        phase=phase, dataset_type=dataset_type
                    )
                    if processed_dataset_kwargs is None:
                        break
                    dataset = dataset_constructor(**processed_dataset_kwargs)
                    dataset = convert_iterable_dataset_to_map(dataset)
                    if phase == MachineLearningPhase.Training:
                        training_dataset = dataset
                    elif phase == MachineLearningPhase.Validation:
                        validation_dataset = dataset
                    else:
                        test_dataset = dataset
                    break
                except Exception as e:
                    get_logger().debug("has exception %s", e)
                    if "of splits is not supported for dataset" in str(e):
                        break
                    if "for argument split. Valid values are" in str(e):
                        break
                    if "Unknown split" in str(e):
                        break
                    raise e

        assert not (validation_dataset is None and test_dataset is None)
        if validation_dataset is None:
            validation_dataset = test_dataset
            test_dataset = None
        return (training_dataset, validation_dataset, test_dataset, dataset_type, name)

    def is_classification_dataset(self) -> bool:
        first_target = self.get_dataset_util(
            phase=MachineLearningPhase.Training
        ).get_sample_label(0)
        match first_target:
            case int():
                return True
        return False

    @classmethod
    def __prepare_dataset_kwargs(
        cls,
        name: str,
        constructor_kwargs: set,
        dataset_kwargs: dict | None = None,
    ) -> Callable:
        if dataset_kwargs is None:
            dataset_kwargs = {}
            new_dataset_kwargs = {}
        else:
            new_dataset_kwargs = copy.deepcopy(dataset_kwargs)
        if "root" not in new_dataset_kwargs:
            new_dataset_kwargs["root"] = cls.__get_dataset_dir(name)
        if "download" not in new_dataset_kwargs:
            new_dataset_kwargs["download"] = True

        def get_dataset_kwargs_per_phase(
            dataset_type: DatasetType, phase: MachineLearningPhase
        ) -> dict | None:
            if "train" in constructor_kwargs:
                # Some dataset only have train and test parts
                if phase == MachineLearningPhase.Validation:
                    return None
                new_dataset_kwargs["train"] = phase == MachineLearningPhase.Training
            elif "split" in constructor_kwargs:
                if phase == MachineLearningPhase.Training:
                    new_dataset_kwargs["split"] = dataset_kwargs.get(
                        "train_split", "train"
                    )
                elif phase == MachineLearningPhase.Validation:
                    if "val_split" in dataset_kwargs:
                        new_dataset_kwargs["split"] = dataset_kwargs["val_split"]
                    else:
                        if dataset_type == DatasetType.Text:
                            new_dataset_kwargs["split"] = "valid"
                        else:
                            new_dataset_kwargs["split"] = "val"
                else:
                    new_dataset_kwargs["split"] = dataset_kwargs.get(
                        "test_split", "test"
                    )
            elif "subset" in constructor_kwargs:
                if phase == MachineLearningPhase.Training:
                    new_dataset_kwargs["subset"] = "training"
                elif phase == MachineLearningPhase.Validation:
                    new_dataset_kwargs["subset"] = "validation"
                else:
                    new_dataset_kwargs["subset"] = "testing"
            discarded_dataset_kwargs = set()
            for k in new_dataset_kwargs:
                if k not in constructor_kwargs:
                    discarded_dataset_kwargs.add(k)
            if discarded_dataset_kwargs:
                get_logger().warning(
                    "discarded_dataset_kwargs %s", discarded_dataset_kwargs
                )
                for k in discarded_dataset_kwargs:
                    new_dataset_kwargs.pop(k)
            return new_dataset_kwargs

        return get_dataset_kwargs_per_phase

    def _split_validation(self) -> None:
        assert not self.has_dataset(phase=MachineLearningPhase.Test)
        get_logger().debug("split validation dataset for %s", self.name)
        datasets = None
        dataset_util = self.get_dataset_util(phase=MachineLearningPhase.Validation)

        def computation_fun():
            nonlocal datasets
            sub_dataset_indices_list = dataset_util.iid_split_indices([1, 1])
            datasets = dataset_util.split_by_indices(sub_dataset_indices_list)
            return sub_dataset_indices_list

        split_index_lists = self.get_cached_data(
            file="split_index_lists.pk", computation_fun=computation_fun
        )
        if datasets is None:
            datasets = dataset_util.split_by_indices(split_index_lists)
        self._datasets[MachineLearningPhase.Validation] = datasets[0]
        self._datasets[MachineLearningPhase.Test] = datasets[1]

    def get_cached_data(self, file: str, computation_fun: Callable) -> dict:
        with DatasetCollection.lock:
            cache_dir = DatasetCollection.__get_dataset_cache_dir(self.name)
            return get_cached_data(os.path.join(cache_dir, file), computation_fun)


class ClassificationDatasetCollection(DatasetCollection):
    @classmethod
    def create(cls, **kwargs):
        dataset_kwargs = kwargs.get("dataset_kwargs", {})
        if not dataset_kwargs:
            dataset_kwargs = {}
        model_config = kwargs.pop("model_config", None)
        dc: ClassificationDatasetCollection = cls(*DatasetCollection.create(**kwargs))
        add_transforms(dc, dataset_kwargs, model_config)
        if not dc.has_dataset(MachineLearningPhase.Test):
            dc._split_validation()
        return dc

    def get_labels(self, use_cache: bool = True) -> set:
        def computation_fun():
            if self.name.lower() == "imagenet":
                return range(1000)
            return self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_labels()

        if not use_cache:
            return computation_fun()

        return self.get_cached_data("labels.pk", computation_fun)

    def get_label_names(self) -> dict:
        def computation_fun():
            label_names = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_label_names()
            if not label_names:
                raise NotImplementedError(f"failed to get label names for {self.name}")
            return label_names

        return self.get_cached_data("label_names.pk", computation_fun)

    def get_raw_data(self, phase: MachineLearningPhase, index: int) -> tuple:
        if self.dataset_type == DatasetType.Vision:
            dataset_util = self.get_dataset_util(phase)
            return (
                dataset_util.get_sample_image(index),
                dataset_util.get_sample_label(index),
            )
        if self.dataset_type == DatasetType.Text:
            dataset_util = self.get_dataset_util(phase)
            return (
                dataset_util.get_sample_text(index),
                dataset_util.get_sample_label(index),
            )
        raise RuntimeError("Unimplemented Code")

    def generate_raw_data(self, phase: MachineLearningPhase):
        if self.dataset_type == DatasetType.Vision:
            dataset_util = self.get_dataset_util(phase)
            return (
                (
                    dataset_util.get_sample_image(i),
                    dataset_util.get_sample_label(i),
                )
                for i in range(len(dataset_util))
            )
        raise RuntimeError("Unimplemented Code")

    @classmethod
    def get_label(cls, label_name, label_names):
        reversed_label_names = {v: k for k, v in label_names.items()}
        return reversed_label_names[label_name]

    def adapt_to_model(self, model, model_kwargs):
        """add more transformers for model"""
        if self.dataset_type == DatasetType.Vision:
            input_size = getattr(model.__class__, "input_size", None)
            if input_size is not None:
                get_logger().debug("resize input to %s", input_size)
                self.append_transform(
                    torchvision.transforms.Resize(input_size), key=TransformType.Input
                )
        get_logger().debug(
            "use transformers for training => \n %s",
            str(self.get_transforms(MachineLearningPhase.Training)),
        )


def create_dataset_collection(
    cls, name: str, dataset_kwargs: dict = None, model_config=None
):
    with cls.lock:
        all_dataset_constructors = set()
        for dataset_type in DatasetType:
            dataset_constructor = get_dataset_constructors(
                dataset_type,
                cache_dir=os.path.join(
                    DatasetCollection.get_dataset_root_dir(), ".cache"
                ),
            )
            if name in dataset_constructor:
                return cls.create(
                    name=name,
                    dataset_type=dataset_type,
                    dataset_constructor=dataset_constructor[name],
                    dataset_kwargs=dataset_kwargs,
                    model_config=model_config,
                )
            if name.lower() in {k.lower() for k in dataset_constructor.keys()}:
                get_logger().warning(
                    "there is a similar name in dataset type %s", str(dataset_type)
                )

            all_dataset_constructors |= dataset_constructor.keys()
        get_logger().error(
            "supported datasets are %s", sorted(all_dataset_constructors)
        )
        raise NotImplementedError(name)


class DatasetCollectionConfig:
    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name
        self.dataset_kwargs = {}
        self.training_dataset_percentage = None
        self.training_dataset_indices_path = None
        self.training_dataset_label_map_path = None
        self.training_dataset_label_map = None
        self.training_dataset_label_noise_percentage = None

    def create_dataset_collection(self, save_dir=None, model_config=None):
        if self.dataset_name is None:
            raise RuntimeError("dataset_name is None")

        dc = create_dataset_collection(
            ClassificationDatasetCollection,
            name=self.dataset_name,
            dataset_kwargs=self.dataset_kwargs,
            model_config=model_config,
        )
        if not dc.is_classification_dataset():
            dc = create_dataset_collection(
                DatasetCollection, self.dataset_name, self.dataset_kwargs
            )

        self.__transform_training_dataset(dc=dc, save_dir=save_dir)
        return dc

    def __transform_training_dataset(self, dc, save_dir=None):
        subset_indices = None
        dataset_util = dc.get_dataset_util(phase=MachineLearningPhase.Training)
        training_dataset = dc.get_dataset(phase=MachineLearningPhase.Training)
        if self.training_dataset_percentage is not None:
            subset_dict = dataset_util.iid_sample(self.training_dataset_percentage)
            subset_indices = sum(subset_dict.values(), [])
            with open(
                os.path.join(save_dir, "training_dataset_indices.json"),
                mode="wt",
                encoding="utf-8",
            ) as f:
                json.dump(subset_indices, f)

        if self.training_dataset_indices_path is not None:
            assert subset_indices is None
            get_logger().info(
                "use training_dataset_indices_path %s",
                self.training_dataset_indices_path,
            )
            with open(self.training_dataset_indices_path, "r", encoding="utf-8") as f:
                subset_indices = json.load(f)
        if subset_indices is not None:
            training_dataset = sub_dataset(training_dataset, subset_indices)
            dc.transform_dataset(
                phase=MachineLearningPhase.Training,
                transformer=lambda _, __: training_dataset,
            )
        dataset_util = dc.get_dataset_util(phase=MachineLearningPhase.Training)
        label_map = None
        if self.training_dataset_label_noise_percentage:
            label_map = dataset_util.randomize_subset_label(
                self.training_dataset_label_noise_percentage
            )
            with open(
                os.path.join(
                    save_dir,
                    "training_dataset_label_map.json",
                ),
                mode="wt",
                encoding="utf-8",
            ) as f:
                json.dump(label_map, f)

        if self.training_dataset_label_map_path is not None:
            assert label_map is not None
            get_logger().info(
                "use training_dataset_label_map_path %s",
                self.training_dataset_label_map_path,
            )
            with open(self.training_dataset_label_map_path, "r", encoding="utf-8") as f:
                self.training_dataset_label_map = json.load(f)

        if self.training_dataset_label_map is not None:
            dc.append_transform(
                transform=replace_target(self.training_dataset_label_map),
                key=TransformType.Target,
                phases=[MachineLearningPhase.Training],
            )
