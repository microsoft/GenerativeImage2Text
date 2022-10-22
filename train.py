import argparse
from generativeimage2text.pl_model import PoseImageCaptioningModel
from generativeimage2text.pl_data import CHPDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main(args):
    seed_everything(args.seed)

    model = PoseImageCaptioningModel(
        args.model_name,
        args.tokenizer_name,
        learning_rate=args.learning_rate
    )

    data_module = CHPDataModule(
        data_path=args.data_path,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        batch_size_test=args.batch_size_test,
        max_length=args.max_length,
        crop_size=args.crop_size,
        dataloader_num_workers=args.dataloader_num_workers
    )

    loss_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_epoch',
        filename='{epoch}-{val_loss_epoch:.6f}',
        save_top_k=args.early_stop_patience,
        mode='min',
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss_epoch',
        patience=args.early_stop_patience,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[
            early_stop_callback,
            loss_checkpoint_callback,
            lr_monitor
        ]
    )

    if args.do_train:
        trainer.fit(model, data_module)

    if args.do_test:
        ckpt_path = 'best' if args.do_train else None
        trainer.test(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--skip_train', dest='do_train',
                        action='store_false', default=True)
    parser.add_argument('--skip_test', dest='do_test',
                        action='store_false', default=True)
    parser = Trainer.add_argparse_args(parser)
    parser = PoseImageCaptioningModel.add_argparse_args(parser)
    parser = CHPDataModule.add_argparse_args(parser)
    main(parser.parse_args())
