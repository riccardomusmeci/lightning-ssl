from src.loss import SSLLOSS
import pytorch_lightning as pl
from src.io.io import load_config
from src.transform import Transform
from src.optimizer import Optimizer
from src.scheduler import LRScheduler
from src.utils import Callbacks, Logger
from src.datamodule import SSLDataModule
from src.model import SSLModel, TeacherStudentSSLModule

def train(args):
    
    pl.seed_everything(args.seed)
    
    config = load_config(args.config)
    
    # Data Module
    datamodule = SSLDataModule(
        data_dir=args.data_dir,
        train_transform=Transform(
            framework=config["framework"],
            train=True,
            **config["transform"]
        ),
        val_transform=Transform(
            framework=config["framework"],
            train=False,
            **config["transform"]
        ),
        **config["datamodule"]
    )
    
    # setting up model, loss, optimizer, lr_scheduler
    model = SSLModel(
        framework=config["framework"],
        img_size=config["transform"]["img_size"],
        **config["model"]
    )
    
    criterion = SSLLOSS(
        framework=config["framework"],
        **config["loss"]
    )
    
    # adapt lr to rule (base_lr * batch_size / 256)
    config["optimizer"]["lr"] *= config["datamodule"]["batch_size"] / 256.
    optimizer = Optimizer(
        model=model, 
        **config["optimizer"]
    )
    lr_scheduler = LRScheduler(
        optimizer=optimizer, 
        **config["lr_scheduler"]
    )
    
    ssl_module = TeacherStudentSSLModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    logger = Logger(output_dir=args.checkpoint_dir)
    callbacks = Callbacks(
        output_dir=args.checkpoint_dir,
        **config["callbacks"]
    )
    
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_from,
        **config["trainer"]
    )
    
    trainer.fit(model=ssl_module, datamodule=datamodule)
    
    