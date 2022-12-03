import torch
import torch.nn as nn
from typing import Tuple
from src.model.layers import MLP
from src.model.layers import Encoder

class BYOL(nn.Module):
    
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
        beta: float = 0.996,
    ) -> None:
        """BYOL initializer

        Args:
            backbone (str): backbone architecture
            img_size (int): input image size.
            pretrained (bool, optional): load pretrained weights. Defaults to True.
            hidden_dim (int, optional): encoder hidden dim. Defaults to 4096.
            proj_dim (int, optional): encoder porjector output dim. Defaults to 256.
            num_layers (int, optional): encoder num of linear layers. Defaults to 3.
            use_bn (bool, optional): use batch norm in encoder. Defaults to False.
            use_gelu (bool, optional): use gelu in encoder. Defaults to False.
            drop_p (float, optional): dropout in encoder. Defaults to 0..
            init_weights (bool, optional): init weights in encoder. Defaults to True.
            beta (float, optional): EMA update weight. Defaults to 0.996.
        """
        super().__init__()
        
        # g_theta 
        self.student = Encoder(
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
        )
        
        # q_theta -> follows g_theta output
        self.q = MLP(
            in_features=proj_dim,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            use_gelu=use_gelu,
            drop_p=drop_p,
            init_weights=init_weights
        )
        
        # f_eps -> second branch (target) -> (no gradient updates for this branch, only EMA)
        self.teacher = Encoder(
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim
        )
        
        # init f weights (target)
        self._init_teacher_weights()
        
        # EMA weight
        self.beta = beta
        
    def _init_teacher_weights(self):
        """inits f() weights with g() weights. Also sets the gradient of f() to False.
        """
        for params_student, params_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            params_teacher.data.copy_(params_student.data) # copying g params
            params_teacher.requires_grad = False # no gradient updates
            
    @torch.no_grad()
    def update_teacher(self):
        """EMA update of the target network
        """
        for params_student, params_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            params_teacher.data = self.beta * params_teacher.data + (1 - self.beta) * params_student.data

    def embeds(self, x) -> torch.Tensor:
        return self.student.backbone(x)
    
    def forward(self, x: Tuple[torch.Tensor]) -> Tuple:
        """BYOL forward pass

        Args:
            x (Tuple[torch.Tensor]) -> (view_1, view_2)

        Returns:
            Tuple: logits_1, logits_2 (g+q branch logits), targ_1, targ_2 (f branch target val)
        """
        x1, x2 = x
        logits_1, logits_2 = self.q(self.student(x1)), self.q(self.student(x2))
        with torch.no_grad():
            targ_1, targ_2 = self.teacher(x1), self.teacher(x2)
        return logits_1, logits_2, targ_1, targ_2
    
    