import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from src.evaluation import KNNEvaluator
from torch.optim.lr_scheduler import _LRScheduler
from src.model.utils.functions import cancel_gradients_last_layer

class TeacherStudentSSLModule(pl.LightningModule):
    
    def __init__(
        self, 
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
        last_layer_frozen: int = 2,
    ) -> None:
        """Teacher-Student Self-Supervised PL Module

        Args:
            model (nn.Module): self-supervised framework
            criterion (nn.Module): loss criterion
            optimizer (Optimizer): optimizer
            lr_scheduler (_LRScheduler, optional): learning rate scheduler. Defaults to None.
            last_layer_frozen (int, optional): if last layer is to freeze. Defaults to 2.
        """
        
        super().__init__()        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.last_layer_frozen = last_layer_frozen
        
    def forward(self, views):
        return self.model(views)
        
    def training_step(self, batch, batch_idx):
        
        x, views = batch
        outputs = self(views)
        loss = self.criterion(outputs)
        
        cancel_gradients_last_layer(
            epoch=self.current_epoch,
            model=self.model,
            frozen_epochs=self.last_layer_frozen
        )
        # EMA update        
        self.model.update_teacher()
        
        self.log("loss/train", loss, sync_dist=True, prog_bar=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
                
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, views = batch
        outputs = self(views)
        loss = self.criterion(outputs)
        
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("loss/val", avg_loss, sync_dist=True, prog_bar=True)
        
    def training_epoch_end(self, outputs):
        pass
        
    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]