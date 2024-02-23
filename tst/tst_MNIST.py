import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def tst_mnist_pl():
    # define any number of nn.Modules (or use your current ones)
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    # define the LightningModule

    # init the autoencoder
    autoencoder = LitAutoEncoder(encoder, decoder)

    # setup data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
    train_loader = utils.data.DataLoader(train_set)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=10)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # load checkpoint
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=9-step=1000.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # choose your trained nn.Module
    encoder = autoencoder.encoder
    encoder.eval()

    # embed 4 fake images!
    fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
    embeddings = encoder(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

    # train 1TB+ parameter models with Deepspeed/fsdp
    trainer = L.Trainer(
        devices=4,
        accelerator="auto",
        precision=16
    )


if __name__ == "__main__":
    tst_mnist_pl()
