import timm
import torch.nn as nn
from typing import Dict
from src.model.vit import create_vit

def create_model(
    backbone: str, 
    pretrained: bool = True,
    img_size: int = None,
    num_classes: int = 0
) -> nn.Module:
    """creates model's backbone

    Args:
        backbone (str): backbone name
        pretrained (bool, optional): pretrained. Defaults to True.
        img_size (int, optional): input image size. Defaults to 224.
        num_classes (int, optional): number of output classes. Defaults to 0.

    Returns:
        nn.Module: backbone model
    """
    
    if backbone.startswith("custom_"):
        model_info=backbone.split("_")
        img_size = int(model_info[-1]) if img_size is None else img_size
        return create_vit(
            vit_base=model_info[1],
            model_size=model_info[2],
            pretrained=pretrained,
            patch_size=int(model_info[3].replace("patch", "")),
            img_size=img_size,
            num_classes=num_classes
        )
    else:
        return timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes
        )

def get_out_features(
    model_name: str
) -> int:
    """returns the out features dim of a model

    Args:
        model_name (str): model name for timm

    Returns:
        int: model out features dim
    """
    if model_name.startswith("custom_"):
        model_info = model_name.split("_")
        if model_info[2] == "tiny": return 192
        if model_info[2] == "small": return 384
        if model_info[2] == "base": return 768
    else:
        model = timm.create_model(
            model_name=model_name, 
            pretrained=False
        )
        layers = list(model.children())
        return layers[-1].in_features

def load_state_dict_ssl(
    model: nn.Module, 
    ssl_state_dict: Dict, 
    initials: str = "model.student.backbone."
) -> nn.Module:
    """loads weights from self-supervised model into model based on initials params (e.g. "model.student.backbone." are the layers to consider for TeacherStudentSSL models)

    Args:
        model (nn.Module): model
        ssl_state_dict (Dict): self supervised model state dict
        initials (str, optional): layers' initial to consider for loading weights to model. Defaults to "model.student.backbone.".

    Returns:
        nn.Module: model with loaded weights
    """
    count = 0
    for k, v in ssl_state_dict.items():
        if k.startswith(initials):
            _k = k.replace(initials, "")
            if _k in model.state_dict().keys():
                model.state_dict()[_k].copy_(v)
                count += 1
    print(f"> Loaded weights into model for {count}/{len(list(model.state_dict().keys()))} layers.")
    if count == 0:
        print("[WARNING] Unable to load weights properly.")
    return model