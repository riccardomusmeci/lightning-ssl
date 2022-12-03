import timm
import torch.nn as nn
from typing import Union
from src.model.byol import BYOL
from src.model.dino import DINO
from src.model.vit import create_vit

def ssl_model(
    framework: str,
    backbone: str,
    img_size: int,
    pretrained: bool = True,
    hidden_dim: int = 4096,
    proj_dim: int = 256,
    out_dim: int = 65568,
    num_layers: int = 3,
    use_bn: bool = False,
    use_gelu: bool = False,
    drop_p: float = 0.,
    init_weights: bool = True,
    norm_last_layer: bool = True,
    beta: float = 0.996,
) -> Union[BYOL, DINO]:
    
    if framework == "byol":
        return BYOL(
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            use_gelu=use_gelu,
            drop_p=drop_p,
            init_weights=init_weights,
            beta=beta
        )
    if framework == "dino":
        return DINO(
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            use_gelu=use_gelu,
            drop_p=drop_p,
            init_weights=init_weights,
            norm_last_layer=norm_last_layer,
            beta=beta
        )
    
    print(f"{framework} not supported.")
    quit()