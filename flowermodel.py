import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
import pytorch_lightning as pl
import torchmetrics

class FlowerPowerNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(512, 17)
        self.acc = torchmetrics.Accuracy()
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        probs = F.softmax(y_hat, dim=1)

        acc = self.acc(probs, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        probs = F.softmax(y_hat, dim=1)

        acc = self.acc(probs, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer