import time
import torch
from transformer.transformer import Transformer
from optimizer import ScheduledOptim
from torch.nn import functional as F
from datasets import load_dataset

import wandb
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
)
from tokenizers.processors import TemplateProcessing
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class Tok:
    def __init__(self, dataset, max_length=512, vocab_lenght=25000):
        self.dataset = dataset
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.enable_truncation(max_length=max_length)
        tokenizer.enable_padding(length=max_length, pad_token="<PAD>")
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_lenght,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<PAD>", "<BOS>", "<EOS>"]
        )

        tokenizer.train_from_iterator(
            self.batch_iterator(), length=25000, trainer=trainer
        )
        self.tokenizer = tokenizer

        if not os.path.exists("tokenizer"):
            os.mkdir("tokenizer")
        self.tokenizer.save("tokenizer/tokenizer_11.m")

    def batch_iterator(self, batch_size=1000):
        for i in range(0, len(self.dataset), batch_size):
            yield ['<BOS>' + k["en"] + '<EOS>' for k in self.dataset[i : i + batch_size]["translation"]]
            yield ['<BOS>' + k["en"] + '<EOS>' for k in self.dataset[i : i + batch_size]["translation"]]


class DS(Dataset):
    def __init__(self):
        self.dataset = load_dataset("opus_books", "en-es")
        self.dataset = self.dataset["train"]
        self.tokenizer = Tok(self.dataset).tokenizer

    def __len__(self):
        # return self.dataset.num_rows
        return 1000

    def __getitem__(self, idx):
        en = self.dataset[idx]["translation"]["en"]
        es = self.dataset[idx]["translation"]["es"]

        tgt, src = self.tokenizer.encode(en), self.tokenizer.encode(es)
        tgt, src = torch.tensor(tgt.ids), torch.tensor(src.ids)
        tgt = F.pad(tgt, (0, 1), "constant", 0)
        return src, tgt


class TT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Transformer(25000, 25000).cuda()
        self.step = 0

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def compute_loss(self, ans, tgt_out, log=False):
        return F.cross_entropy(ans.transpose(1, 2), tgt_out)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        out = self.forward(src, tgt_in)
        loss = self.compute_loss(out, tgt_out)
        self.log("train/loss", loss)
        self.step += 1

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = ScheduledOptim(optimizer, 1, self.model.d_model, 4000)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def get_checkpoint_callback(save_path="model", name="model_checkpoint_11"):
    saving_path = os.path.join(save_path, name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(saving_path)
    return ModelCheckpoint(dirpath=saving_path)


def initialize_wandb():
    wandb.login()
    wandb.init(project="attention", notes="Train Test")


if __name__ == "__main__":

    initialize_wandb()
    model = TT()
    checkpoint_callback = get_checkpoint_callback()

    wandb_logger = WandbLogger()

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")

    trainer = pl.Trainer(
        gpus=-1,
        auto_select_gpus=True,
        accelerator="dp",
        logger=WandbLogger(),
        callbacks=[checkpoint_callback],
        max_epochs=100,
    )

    ds = DS()
    train_dataloader = DataLoader(
        ds, batch_size=32, shuffle=True, num_workers=os.cpu_count()
    )

    trainer.fit(model, train_dataloader)
