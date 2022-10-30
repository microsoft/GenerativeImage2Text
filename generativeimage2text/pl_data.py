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


class CHPDatasetBase(Dataset):
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, max_length: Optional[int], crop_size: int) -> None:
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.image_path = os.path.join(os.path.dirname(csv_path), 'images')
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.image_transform = Compose([
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(crop_size),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def load_image(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        return self.image_transform(image)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # tokenize reference description
        target_encoding = self.tokenizer(
            row['description'],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors='pt'
        )
        caption_tokens = target_encoding['input_ids'].squeeze()
        need_predict = torch.ones_like(caption_tokens)
        need_predict[0] = 0

        # load image
        image_path = os.path.join(self.image_path, row['image_name'])
        image = self.load_image(image_path)

        return {
            'sample_id': row['sample_id'],
            'caption_tokens': caption_tokens,
            'need_predict': need_predict,
            'image': image,
        }

    def __len__(self):
        return self.data.shape[0]


class CHPDataset(CHPDatasetBase):
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, max_length: Optional[int], crop_size: int) -> None:
        super().__init__(csv_path, tokenizer, max_length, crop_size)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # tokenize reference description
        target_encoding = self.tokenizer(
            row['description'],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors='pt'
        )
        caption_tokens = target_encoding['input_ids'].squeeze()
        need_predict = torch.ones_like(caption_tokens)
        need_predict[0] = 0

        # load image
        image_path = os.path.join(self.image_path, row['image_name'])
        image = self.load_image(image_path)

        return {
            'sample_id': row['sample_id'],
            'caption_tokens': caption_tokens,
            'need_predict': need_predict,
            'image': image,
        }

class CHPTestDataset(CHPDatasetBase):
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, max_length: Optional[int], crop_size: int) -> None:
        super().__init__(csv_path, tokenizer, max_length, crop_size)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # load image
        image_path = os.path.join(self.image_path, row['image_name'])
        image = self.load_image(image_path)

        return {
            'sample_id': row['sample_id'],
            'reference': row['description'],
            'image': image,
        }

class CHPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        batch_size: int,
        batch_size_test: int,
        max_length: Optional[int],
        crop_size: int,
        dataloader_num_workers: int
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.crop_size = crop_size
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
            default=16
        )
        parser.add_argument(
            '--batch_size_test',
            type=int,
            default=1
        )
        parser.add_argument(
            '--data_path',
            type=str,
            required=True
        )
        parser.add_argument(
            '--max_length',
            type=int,
            default=None
        )
        parser.add_argument(
            '--dataloader_num_workers',
            type=int,
            default=0
        )
        parser.add_argument(
            '--crop_size',
            type=int,
            default=224
        )
        return parent_parser

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = CHPDataset(
            os.path.join(self.data_path, 'train.csv'),
            self.tokenizer,
            self.max_length,
            self.crop_size
        )

        self.val_dataset = CHPDataset(
            os.path.join(self.data_path, 'val.csv'),
            self.tokenizer,
            self.max_length,
            self.crop_size
        )

        self.test_dataset = CHPTestDataset(
            os.path.join(self.data_path, 'test.csv'),
            self.tokenizer,
            self.max_length,
            self.crop_size
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

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            pin_memory=True,
            num_workers=self.dataloader_num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            pin_memory=True,
            num_workers=self.dataloader_num_workers
        )

    def collate(self, batch):
        need_padding = ['caption_tokens', 'need_predict']
        collated = {}
        for key in need_padding:
            if key not in batch[0]:
                continue
            items = [sample.pop(key) for sample in batch]
            collated[key] = pad_sequence(
                items, batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
        collated.update(default_collate(batch))
        return collated
