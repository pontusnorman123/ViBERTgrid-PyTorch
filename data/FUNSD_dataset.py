import os ##FUNSD_Dataset.py
import json

from tqdm import tqdm
from typing import Tuple, Optional, Callable

import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import distributed, Dataset, DataLoader, BatchSampler
from transformers import BertTokenizer


FUNSD_CLASS_INDEX = {"COMPANY": 0, "DATE": 1, "ADDRESS": 2, "TOTAL": 3, "TAX": 4, "PRODUCT": 5, "others":6}


class FUNSDDataset(Dataset):
    """The FUNSD dataset

    MEAN = [0.94802886 0.94802886 0.94802886]
    STD = [0.18400319 0.18400319 0.18400319]

    Parameters
    ----------
    root : str
        root directory of the dataset, which contains 'train' and 'validate' folders
    train : bool, optional
        set True if loading training set, else False, by default True
    tokenizer : Optional[Callable], optional
        tokenizer used for corpus generation,
        from huggingface/transformer library,
        e.g transformer.BertTokenizer,
        by default None

    Returns
    -------
    imgs: tuple[torch.Tensor]
        tuple of original images,
        each with shape [3, H, W]
        need further transforms for forward propagation
    class_labels: tuple[torch.Tensor]
        tuple of class labels, same in shape with images,
        each with shape [num_classes, H, W].
        if the pixel at (x, y) belongs to class [3], then the value at
        channel [3] (in other words, at coor (3, x, y)) is one and zero
        at other channels.
        need further transforms for forward propagation
    pos_neg_labels: tuple[torch.Tensor]
        tuple of pos_neg labels, same in shape with images,
        each with shape [3, H, W].
        channel [0] = 1 if the pixel belongs to background
        channel [1] = 1 if the pixel belongs to key text region
        channel [2] = 1 if the pixel belongs to non-key text region
        need further transforms for forward propagation
    ocr_coors: torch.Tensor
        coordinates of each token
    ocr_corpus: torch.Tensor
        corpus generated from the given OCR result, padding performed.
    mask: torch.Tensor
        BERT-like model requires input with constant length (typically 512),
        if len(corpus) < constant_length, padding will be performed.
        mask indicates where the padding steps are. len(mask) = constant_length,
        mask[step] = 0 at padding steps, 1 otherwise.

    """

    def __init__(
        self, root: str, train: bool, tokenizer: Optional[Callable] = None
    ) -> None:
        super().__init__()

        assert os.path.exists(root), f"the given root path {root} does not exists"
        assert tokenizer is not None, "no tokenizer given"

        self.root = root
        self.train = train
        self.tokenizer = tokenizer
        self.max_length = 0
        self.transform_img = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

        self.filename_list = []
        if self.train:
            self.subset_root = os.path.join(root, "training_data")
            for file in os.listdir(os.path.join(self.subset_root, "_label_csv")):
                self.filename_list.append(file.replace(".csv", ""))
        else:
            self.subset_root = os.path.join(root, "testing_data")
            for file in os.listdir(os.path.join(self.subset_root, "_label_csv")):
                self.filename_list.append(file.replace(".csv", ""))

    def __len__(self) -> int:
        return len(self.filename_list)

    def __getitem__(self, index):
        dir_img = os.path.join(
            self.subset_root, "images", (self.filename_list[index] + ".png")
        )
        dir_csv_label = os.path.join(
            self.subset_root, "_label_csv", (self.filename_list[index] + ".csv")
        )

        image = Image.open(dir_img)
        if len(image.split()) != 3:
            image = image.convert("RGB")

        ocr_coor = []
        ocr_text = []
        seg_classes = []
        csv_label: pd.DataFrame = pd.read_csv(dir_csv_label)
        for _, row in csv_label.iterrows():
            if row["text"] == "" or row["text"] == " ":
                continue

            ocr_text.append(str(row["text"]))
            ocr_coor.append([row["left"], row["top"], row["right"], row["bot"]])
            seg_classes.append(row["data_class"])

        ocr_tokens = []
        seg_indices = []
        ocr_coor_ = []
        seg_classes_ = []
        ocr_text_filter = []
        seg_index = 0
        seg_index_filter = 0
        for text in ocr_text:
            if len(text) == 0 or text.isspace():
                seg_index += 1
                continue
            curr_tokens = self.tokenizer.tokenize(text)
            if len(curr_tokens) == 0:
                seg_index += 1
                continue
            ocr_text_filter.append(text)
            ocr_coor_.append(ocr_coor[seg_index])
            seg_classes_.append(seg_classes[seg_index])
            for i in range(len(curr_tokens)):
                ocr_tokens.append(curr_tokens[i])
                seg_indices.append(seg_index_filter)
            seg_index += 1
            seg_index_filter += 1

        ocr_corpus = self.tokenizer.convert_tokens_to_ids(ocr_tokens)

        if self.train == True:
            return (
                self.transform_img(image),
                torch.tensor(seg_indices, dtype=torch.int),
                torch.tensor(seg_classes_, dtype=torch.int),
                torch.tensor(ocr_coor_, dtype=torch.long),
                torch.tensor(ocr_corpus, dtype=torch.long),
            )
        else:
            return (
                self.transform_img(image),
                torch.tensor(seg_indices, dtype=torch.int),
                torch.tensor(seg_classes_, dtype=torch.int),
                torch.tensor(ocr_coor_, dtype=torch.long),
                torch.tensor(ocr_corpus, dtype=torch.long),
                ocr_text_filter,
            )

    def _ViBERTgrid_coll_func(self, samples):
        imgs = []
        seg_indices = []
        token_classes = []
        ocr_coors = []
        ocr_corpus = []
        ocr_text = []
        for item in samples:
            imgs.append(item[0])
            seg_indices.append(item[1])
            token_classes.append(item[2])
            ocr_coors.append(item[3])
            ocr_corpus.append(item[4])
            if self.train == False:
                ocr_text.append(item[5])

        # pad sequence to generate mini-batch
        ocr_corpus = pad_sequence(ocr_corpus, batch_first=True)
        # add mask to indicate valid corpus
        mask = torch.zeros(ocr_corpus.shape, dtype=torch.long)
        mask = mask.masked_fill_((ocr_corpus != 0), 1)

        if self.train == True:
            return (
                tuple(imgs),
                tuple(seg_indices),
                tuple(token_classes),
                tuple(ocr_coors),
                ocr_corpus,
                mask.int(),
            )
        else:
            return (
                tuple(imgs),
                tuple(seg_indices),
                tuple(token_classes),
                tuple(ocr_coors),
                ocr_corpus,
                mask.int(),
                tuple(ocr_text),
                None,
            )


def load_train_dataset(
    root: str,
    batch_size: int,
    num_workers: int = 0,
    tokenizer: Optional[Callable] = None,
    return_mean_std: bool = False,
) -> Tuple[DataLoader]:
    """load FUNSD train dataset

    Parameters
    ----------
    root : str
        root of dataset
    batch_size : int
        batch size
    num_workers : int, optional
        number of workers in dataloader, by default 0
    tokenizer : optional
        tokenizer
    return_mean_std : bool
        if True, return mean and std of train set, by default False

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    image_mean : numpy.ndarray
    image_std : numpy.ndarray

    """

    FUNSD_train_dataset = FUNSDDataset(root=root, train=True, tokenizer=tokenizer)
    FUNSD_val_dataset = FUNSDDataset(root=root, train=False, tokenizer=tokenizer)

    train_loader = DataLoader(
        FUNSD_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=FUNSD_train_dataset._ViBERTgrid_coll_func,
    )

    val_loader = DataLoader(
        FUNSD_val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=FUNSD_val_dataset._ViBERTgrid_coll_func,
    )

    if return_mean_std:
        print("calculating mean and std")
        image_mean = torch.zeros(3)
        image_std = torch.zeros(3)
        for image_list, _, _, _, _, _ in tqdm(train_loader):
            for batch_index in range(batch_size):
                if batch_index >= len(image_list):
                    continue
                curr_img = image_list[batch_index]
                for d in range(3):
                    image_mean[d] += curr_img[d, :, :].mean()
                    image_std[d] += curr_img[d, :, :].std()
        image_mean.div_(len(FUNSD_train_dataset))
        image_std.div_(len(FUNSD_train_dataset))

        return train_loader, val_loader, image_mean.numpy(), image_std.numpy()

    return train_loader, val_loader


def load_train_dataset_multi_gpu(
    root: str,
    batch_size: int,
    num_workers: int = 0,
    tokenizer: Optional[Callable] = None,
) -> Tuple[DataLoader]:
    """load FUNSD train dataset in multi-gpu scene

    Parameters
    ----------
    root : str
        root of dataset
    batch_size : int
        batch size
    num_workers : int, optional
        number of workers in dataloader, by default 0
    tokenizer : optional
        tokenizer

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader

    """

    FUNSD_train_dataset = FUNSDDataset(root, train=True, tokenizer=tokenizer)
    FUNSD_val_dataset = FUNSDDataset(root, train=False, tokenizer=tokenizer)

    train_sampler = distributed.DistributedSampler(FUNSD_train_dataset)
    val_sampler = distributed.DistributedSampler(FUNSD_val_dataset)

    train_batch_sampler = BatchSampler(
        train_sampler, batch_size=batch_size, drop_last=True
    )

    train_loader = DataLoader(
        FUNSD_train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        collate_fn=FUNSD_train_dataset._ViBERTgrid_coll_func,
    )

    val_loader = DataLoader(
        FUNSD_val_dataset,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=FUNSD_val_dataset._ViBERTgrid_coll_func,
    )

    return train_loader, val_loader, train_sampler


def load_test_data(
    root: str,
    num_workers: int = 0,
    tokenizer: Optional[Callable] = None,
):
    FUNSD_test_dataset = FUNSDDataset(root=root, train=False, tokenizer=tokenizer)
    test_loader = DataLoader(
        FUNSD_test_dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=FUNSD_test_dataset._ViBERTgrid_coll_func,
        shuffle=True,
    )

    return test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--model")
    args = parser.parse_args()

    dir_processed = args.root
    model_version = args.model
    print("loading bert pretrained")
    tokenizer = BertTokenizer.from_pretrained(model_version)
    train_loader, val_loader, image_mean, image_std = load_train_dataset(
        # train_loader, val_loader = load_train_dataset(
        dir_processed,
        batch_size=4,
        num_workers=0,
        tokenizer=tokenizer,
        return_mean_std=True,
    )

    print(image_mean, image_std)

    # for train_batch in tqdm(train_loader):
    #     img, class_label, pos_neg, coor, corpus, mask = train_batch
    #     print("TEST")