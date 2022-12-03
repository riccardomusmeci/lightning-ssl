import torch
import torch.nn as nn
from src.model.layers import MLP
from src.model.utils import get_out_features, create_model

class Encoder(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        img_size: int,
        pretrained: bool = True,
        hidden_dim: int = 4096,
        proj_dim: int = 256,
        num_layers: int = 2,
        use_bn: bool = True,
        use_gelu: bool = False,
        drop_p: float = 0.,
        init_weights: bool = True,
        dino: bool = False,
        dino_out_dim: int = 65568,
        norm_last_layer: bool = False
    ) -> None:
        """Encoder initializer

        Args:
            backbone (str): backbone architecture
            img_size (int): input image size
            pretrained (bool, optional): load pretrained weights. Defaults to True.
            hidden_dim (int, optional): MLP hidden dim. Defaults to 4096.
            proj_dim (int, optional): MLP projector output dim. Defaults to 256.
            num_layers (int, optional): MLP number of linear layers. Defaults to 2.
            use_bn (bool, optional): use batch norm in MLP. Defaults to True.
            use_gelu (bool, optional): use GELU in MLP. Defaults to False.
            drop_p (float, optional): dropout prob in MLP. Defaults to 0..
            init_weights (bool, optional): whether to init weights in MLP. Defaults to True.
            dino (bool, optional): if DINO model and so put last layer. Defaults to False.
            dino_out_dim (int, optional): DINO output dim. Defaults to 65568.
            norm_last_layer (bool, optional): whether to normalize last layer. Defaults to False.
        """
        
        super().__init__()
        
        self.dino = dino
        
        self.backbone = create_model(
            backbone=backbone,
            pretrained=pretrained,
            img_size=img_size
        )
        backbone_out = get_out_features(backbone)
        
        self.projector = MLP(
            in_features=backbone_out,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            use_gelu=use_gelu,
            drop_p=drop_p,
            init_weights=init_weights
        )
        
        if dino:
            self.last_layer = nn.utils.weight_norm(nn.Linear(in_features=proj_dim, out_features=dino_out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.backbone(x)
        x = self.projector(x)
        if hasattr(self, 'last_layer'):
            x = nn.functional.normalize(x, dim=-1, p=2)             
            x = self.last_layer(x)
        return x
        