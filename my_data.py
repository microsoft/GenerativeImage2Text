import os
from argparse import ArgumentParser
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor, InterpolationMode)
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import numpy as np
from .train import prep_forward_data
import torchvision.datasets as datasets
import random
from .data_layer.builder import collate_fn
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class CHPDatasetBase(Dataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data = None
        # self.process_data()
    
    def process_data(self):
        print("Processing Dataset!")
        data = []
        for x in tqdm(self.dataset):
            for caption in x[1]:
                data.append((x[0], caption))
        
        self.data = data
        print("Processing Dataset Done!")
        
    def __len__(self):
        return len(self.dataset)


class CHPDataset(CHPDatasetBase):
    def __init__(self, dataset, tokenizer: BertTokenizer) -> None:
        super().__init__(dataset, tokenizer)

    def __getitem__(self, index):
        row = self.dataset[index][0],self.dataset[index][1][0]
        batch = row[0], row[1]

        return batch


class CHPTestDataset(CHPDatasetBase):
    def __init__(self, dataset, tokenizer: BertTokenizer) -> None:
        super().__init__(dataset, tokenizer)

    def __getitem__(self, index):
        row = self.dataset[index][0],self.dataset[index][1][random.randint(0, len(self.dataset[index][1]))] 
        batch = prep_forward_data([row[0]], [row[1]])

        return batch



class CHPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: BertTokenizer,
        batch_size: int,
        batch_size_test: int,
        dataloader_num_workers: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.save_hyperparameters(ignore='tokenizer')

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group(
            'CHPDataModule'
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            default=32
        )
        parser.add_argument(
            '--batch_size_test',
            type=int,
            default=1
        )
        parser.add_argument(
            '--dataloader_num_workers',
            type=int,
            default=8
        )
        return parent_parser

    def setup(self, stage: Optional[str] = None):
        dataset = datasets.CocoCaptions(annFile = "dataset/annotations/captions_val2014.json", root = "dataset/val2014")
        self.train_dataset = CHPDataset(
            dataset,
            self.tokenizer,
        )

        self.val_dataset = self.train_dataset


        self.test_dataset = CHPTestDataset(
            dataset,
            self.tokenizer,
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate,
            pin_memory=True,
            num_workers=self.dataloader_num_workers
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         collate_fn=self.collate,
    #         pin_memory=True,
    #         num_workers=self.dataloader_num_workers
    #     )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            pin_memory=True,
            num_workers=self.dataloader_num_workers
        )

    def collate(self, batch):
        images = []
        captions = []
        for i in batch:
            images.append(i[0])
            captions.append(i[1])
        
        return prep_forward_data(self.tokenizer, images, captions)
