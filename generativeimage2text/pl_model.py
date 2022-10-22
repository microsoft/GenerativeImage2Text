from argparse import ArgumentParser
from statistics import mean
from typing import Dict

import pytorch_lightning as pl
import torch
from azfuse import File
from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score
from transformers import BertTokenizer

from .model import get_git_model
from .torch_common import load_state_dict, torch_load
from .tsv_io import load_from_yaml_file


class PoseImageCaptioningModel(pl.LightningModule):
    def __init__(self, model_name: str, tokenizer_name: str, learning_rate: float) -> None:
        super().__init__()
        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name,
            do_lower_case=True
        )
        # GIT model
        git_param = {}
        if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
            git_param = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')
        self.git_model = get_git_model(self.tokenizer, git_param)
        pretrained = f'output/{model_name}/snapshot/model.pt'
        checkpoint = torch_load(pretrained)['model']
        load_state_dict(self.git_model, checkpoint)

        self.save_hyperparameters()

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group(
            'PoseImageCaptioningModel'
        )
        parser.add_argument(
            '--model_name',
            type=str,
            default='GIT_BASE_COCO'
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

    def forward(self, inputs: Dict[str, torch.Tensor], infer: bool = False) -> Dict[str, torch.Tensor]:
        return self.git_model(inputs, infer)

    def training_step(self, batch, batch_idx):
        loss_dict = self(batch)
        loss = sum(loss_dict.values())

        self.log_dict(
            {'train_loss': loss},
            batch_size=batch['image'].size(0),
            on_epoch=True,
            on_step=True,
            logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict = self(batch)
        loss = sum(loss_dict.values())

        self.log_dict(
            {'val_loss': loss},
            batch_size=batch['image'].size(0),
            on_epoch=True,
            on_step=True,
            logger=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        image = batch['image']
        references = batch['reference']
        result = self({'image': image}, infer=True)
        predictions = self.tokenizer.batch_decode(
            result['predictions'],
            skip_special_tokens=True
        )

        bleu_scores = [
            bleu_score.sentence_bleu(
                [word_tokenize(reference)],
                word_tokenize(prediction),
                smoothing_function=bleu_score.SmoothingFunction().method4
            )
            for reference, prediction in zip(references, predictions)
        ]
        mean_bleu = mean(bleu_scores)

        # Please help add other metrics here:
        # cider*, bertscore, bleurt
        
        # * Note since cider needs the entire corpus to calculate the
        # score, we need to do that in self.test_epoch_end (which has
        # access to the list of references and predictions we return here).

        self.log_dict(
            {'test_bleu': mean_bleu},
            batch_size=batch['image'].size(0),
            on_epoch=True,
            on_step=True,
            logger=True
        )

        return {'references': references, 'predictions': predictions}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
