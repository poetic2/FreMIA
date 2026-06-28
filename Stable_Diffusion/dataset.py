import io
import os
import random
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from datasets import Dataset, load_from_disk
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import CocoDetection


IMAGE_COLUMN = "image"
CAPTION_COLUMN = "text"
RESOLUTION = 512


def tokenize_captions(examples, tokenizer, is_train=True):
    captions = _select_captions(examples[CAPTION_COLUMN], is_train)
    return _tokenize(tokenizer, captions)


def tokenize_captions_multi(examples, tokenizer, is_train=True):
    captions = _select_captions(examples[CAPTION_COLUMN], is_train)
    return _tokenize_caption_parts(tokenizer, captions)


def preprocess_train(examples, tokenizer, multi_caption=False):
    transform = _image_transform()
    images = [image.convert("RGB") for image in examples[IMAGE_COLUMN]]
    examples["pixel_values"] = [transform(image) for image in images]

    if multi_caption:
        (
            examples["input_ids"],
            examples["input_ids_1"],
            examples["input_ids_2"],
            examples["input_ids_3"],
            examples["input_ids_null"],
        ) = tokenize_captions_multi(examples, tokenizer)
    else:
        examples["input_ids"] = tokenize_captions(examples, tokenizer)

    return examples


def collate_fn(examples):
    return _collate(examples, multi_caption=False)


def collate_fn_clid(examples):
    return _collate(examples, multi_caption=True)


def load_pokemon_datasets(dataset_root, num_samples=415, tokenizer=None):
    return _load_pokemon_datasets(dataset_root, num_samples, tokenizer, multi_caption=False)


def clid_load_pokemon_datasets(dataset_root, num_samples=415, tokenizer=None):
    return _load_pokemon_datasets(dataset_root, num_samples, tokenizer, multi_caption=True)


def load_coco_datasets(dataset_root, num_samples=100, tokenizer=None):
    return _load_coco_datasets(dataset_root, num_samples, tokenizer, multi_caption=False)


def clid_load_coco_datasets(dataset_root, num_samples=100, tokenizer=None):
    return _load_coco_datasets(dataset_root, num_samples, tokenizer, multi_caption=True)


def load_flickr_datasets(dataset_root, num_samples=100, tokenizer=None):
    return _load_flickr_like_datasets(dataset_root, "Flickr", num_samples, tokenizer, multi_caption=False)


def clid_load_flickr_datasets(dataset_root, num_samples=100, tokenizer=None):
    return _load_flickr_like_datasets(dataset_root, "Flickr", num_samples, tokenizer, multi_caption=True)


def clid_load_ti_datasets(dataset_root, num_samples=100, tokenizer=None):
    return _load_flickr_like_datasets(dataset_root, "text-to-image-2m", num_samples, tokenizer, multi_caption=True)


class CocoCaptionsDict(CocoDetection):
    def __init__(
        self,
        split,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        tokenizer=None,
        multi_caption=False,
    ) -> None:
        assert split in ["train", "val"]
        ann_file = os.path.join(root, "annotations/captions_val2017.json")
        conf = OmegaConf.load(os.path.join(root, "coco_split.yaml"))
        image_root = os.path.join(root, "val2017")

        self.split = split
        self._train_ids = conf["train"]
        self._val_ids = conf["test"]
        self.tokenizer = tokenizer
        self.multi_caption = multi_caption

        super().__init__(image_root, ann_file, transform, target_transform, transforms)
        selected_ids = self._train_ids if split == "train" else self._val_ids
        self.ids = [self.ids[i] for i in selected_ids]
        self._init_tokenized_captions()

    def _init_tokenized_captions(self):
        captions = []
        for image_id in self.ids:
            caption = [ann["caption"] for ann in super()._load_target(image_id)]
            captions.append(caption[0] if isinstance(caption, (list, np.ndarray)) else caption)

        if self.multi_caption:
            (
                self.input_ids,
                self.input_ids_1,
                self.input_ids_2,
                self.input_ids_3,
                self.input_ids_null,
            ) = _tokenize_caption_parts(self.tokenizer, captions)
        else:
            self.input_ids = _tokenize(self.tokenizer, captions)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = self._load_image(image_id)

        if self.transforms is not None:
            image, _ = self.transforms(image, None)

        item = {"pixel_values": image, "input_ids": self.input_ids[index]}
        if self.multi_caption:
            item.update(
                {
                    "input_ids_1": self.input_ids_1[index],
                    "input_ids_2": self.input_ids_2[index],
                    "input_ids_3": self.input_ids_3[index],
                    "input_ids_null": self.input_ids_null[index],
                }
            )
        else:
            item["caption"] = super()._load_target(image_id)

        return item


def _load_pokemon_datasets(dataset_root, num_samples, tokenizer, multi_caption):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")

    dataset = load_from_disk(os.path.join(dataset_root, "pokemon"))

    def preprocess_with_tokenizer(examples):
        return preprocess_train(examples, tokenizer, multi_caption=multi_caption)

    train_dataset = dataset["train"].select(range(num_samples)).with_transform(preprocess_with_tokenizer)
    test_dataset = dataset["test"].select(range(num_samples)).with_transform(preprocess_with_tokenizer)
    collate = collate_fn_clid if multi_caption else collate_fn

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=1, collate_fn=collate)
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def _load_coco_datasets(dataset_root, num_samples, tokenizer, multi_caption):
    train_dataset = CocoCaptionsDict(
        split="train",
        transform=_image_transform(),
        tokenizer=tokenizer,
        root=os.path.join(dataset_root, "coco2017val"),
        multi_caption=multi_caption,
    )
    test_dataset = CocoCaptionsDict(
        split="val",
        transform=_image_transform(),
        tokenizer=tokenizer,
        root=os.path.join(dataset_root, "coco2017val"),
        multi_caption=multi_caption,
    )

    if num_samples is not None:
        train_dataset = Subset(train_dataset, list(range(min(num_samples, len(train_dataset)))))
        test_dataset = Subset(test_dataset, list(range(min(num_samples, len(test_dataset)))))

    collate = collate_fn_clid if multi_caption else collate_fn
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, collate_fn=collate, batch_size=1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, collate_fn=collate, batch_size=1)
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def _load_flickr_like_datasets(dataset_root, folder_name, num_samples, tokenizer, multi_caption):
    data = _read_parquet_image_caption_dataset(os.path.join(dataset_root, folder_name))

    def preprocess_with_tokenizer(examples):
        return preprocess_train(examples, tokenizer, multi_caption=multi_caption)

    train_dataset = Dataset.from_dict({"image": data["train_images"], "text": data["train_texts"]})
    test_dataset = Dataset.from_dict({"image": data["test_images"], "text": data["test_texts"]})
    train_dataset = train_dataset.select(range(min(num_samples, len(train_dataset)))).with_transform(
        preprocess_with_tokenizer
    )
    test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset)))).with_transform(
        preprocess_with_tokenizer
    )

    collate = collate_fn_clid if multi_caption else collate_fn
    shuffle = not multi_caption
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, batch_size=1, collate_fn=collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=shuffle, batch_size=1, collate_fn=collate)
    return train_dataloader, test_dataloader


def _read_parquet_image_caption_dataset(folder_path):
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(folder_path, file)))

    combined_df = pd.concat(dfs, ignore_index=True)
    data = {"train_images": [], "train_texts": [], "test_images": [], "test_texts": []}
    for _, row in combined_df.iterrows():
        image = Image.open(io.BytesIO(row["image"]["bytes"]))
        if row["split"] == "train":
            data["train_images"].append(image)
            data["train_texts"].append(row["caption"])
        elif row["split"] == "test":
            data["test_images"].append(image)
            data["test_texts"].append(row["caption"])
    return data


def _collate(examples, multi_caption):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    batch = {
        "pixel_values": pixel_values,
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
    }

    if multi_caption:
        batch.update(
            {
                "input_ids_1": torch.stack([example["input_ids_1"] for example in examples]),
                "input_ids_2": torch.stack([example["input_ids_2"] for example in examples]),
                "input_ids_3": torch.stack([example["input_ids_3"] for example in examples]),
                "input_ids_null": torch.stack([example["input_ids_null"] for example in examples]),
            }
        )

    return batch


def _select_captions(raw_captions, is_train):
    captions = []
    for caption in raw_captions:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(f"Caption column `{CAPTION_COLUMN}` should contain strings or lists of strings.")
    return captions or ["None"]


def _tokenize(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids


def _tokenize_caption_parts(tokenizer, captions):
    return (
        _tokenize(tokenizer, captions),
        _tokenize(tokenizer, [caption[: int(len(caption) / 3)] for caption in captions]),
        _tokenize(tokenizer, [caption[int(len(caption) / 3) : int(2 * len(caption) / 3)] for caption in captions]),
        _tokenize(tokenizer, [caption[int(2 * len(caption) / 3) :] for caption in captions]),
        _tokenize(tokenizer, ["" for _ in captions]),
    )


def _image_transform():
    return transforms.Compose(
        [
            transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
