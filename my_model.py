from argparse import ArgumentParser
from collections import defaultdict
from statistics import mean
from typing import Dict

import pytorch_lightning as pl
import torch
from azfuse import File
from transformers import BertTokenizer

from .model import get_git_model
from .torch_common import load_state_dict, torch_load
from .tsv_io import load_from_yaml_file

import warnings
warnings.filterwarnings("ignore")


class TestImageCaptioningModel(pl.LightningModule):
    def __init__(self, model_name: str, tokenizer_name: str, learning_rate: float) -> None:
        super().__init__()
        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name,
            do_lower_case=True
        )
        # GIT model
        git_param = {}
        # if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        #     git_param = load_from_yaml_file(
        #         f'aux_data/models/{model_name}/parameter.yaml')
        self.git_model = get_git_model(self.tokenizer, git_param)
        # pretrained = f'output/{model_name}/snapshot/model.pt'
        # checkpoint = torch.load(pretrained)  #torch_load(pretrained)['model']
        # load_state_dict(self.git_model, checkpoint)
        self.test_results = None

        self.save_hyperparameters()

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group(
            'PoseImageCaptioningModel'
        )
        parser.add_argument(
            '--model_name',
            type=str,
            default='GIT_LARGE_COCO'
        )
        parser.add_argument(
            '--tokenizer_name',
            type=str,
            default='bert-base-uncased'
        )
        parser.add_argument(
            '--learning_rate',
            type=float,
            default=2.5e-6
        )
        return parent_parser

    def forward(self, inputs, infer: bool = False) -> Dict[str, torch.Tensor]:
        return self.git_model(inputs)

    def training_step(self, batch, batch_idx):
        loss_dict = self(batch)
        loss = sum(loss_dict.values())

        self.log_dict(
            {'train_loss': loss},
            on_epoch=True,
            on_step=True,
            logger=True
        )

        return loss

#     def validation_step(self, batch, batch_idx):
#         loss_dict = self(batch)
#         loss = sum(loss_dict.values())

#         self.log_dict(
#             {'val_loss': loss},
#             batch_size=len(batch['sample_id']),
#             on_epoch=True,
#             on_step=True,
#             logger=True
#         )


#     def test_step(self, batch, batch_idx):
#         loss_dict = self(batch)
#         loss = sum(loss_dict.values())

#         self.log_dict(
#             {'test_loss': loss},
#             batch_size=len(batch['sample_id']),
#             on_epoch=True,
#             on_step=True,
#             logger=True
#         )

#     def test_epoch_end(self, outputs):
#         loss_dict = self(batch)
#         loss = sum(loss_dict.values())

#         self.log_dict(
#             {'test_loss': loss},
#             batch_size=len(batch['sample_id']),
#             on_epoch=True,
#             on_step=True,
#             logger=True
#         )


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
