from typing import Union
from src.loss.byol import BYOLLoss
from src.loss.dino import DINOLoss

def ssl_loss(
    framework: str,
    **kwargs
) -> Union[DINOLoss, BYOLLoss]:
    """returns self supervised loss

    Args:
        framework (str): framework name

    Returns:
        Union[DINOLoss, BYOLLoss]: self supervised loss
    """
    
    if framework == "byol":
        return BYOLLoss(**kwargs)

    if framework == "dino":
        return DINOLoss(**kwargs)

    print(f"{framework} not supported.")
    quit()