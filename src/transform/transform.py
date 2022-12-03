from typing import Callable, Union
from src.transform import SSLTransform, DINOTransform

def transform(
    framework: str,
    train: bool, 
    img_size: Union[int, list, tuple], 
    local_crop_size: Union[int, list, tuple] = 96,
    global_crops_scale: tuple = (0.4, 1), 
    local_crops_scale: tuple =(0.05, .4), 
    n_local_crops: int = 8,
    mean: list = [0.485, 0.456, 0.406], 
    std: list = [0.229, 0.224, 0.225],
    crop_resize_p: float = 0.5,
    brightness: float = 0.4, 
    contrast: float = 0.4, 
    saturation: float = 0.2, 
    hue: float = 0.1,
    color_jitter_p: float = .5,
    grayscale_p: float = 0.2,
    h_flip_p: float = .5,
    kernel: tuple = (5, 5),
    sigma: tuple = (.1, 2),
    gaussian_blur_p: float = 0.1,
    solarization_p: float = 0.2,
    solarize_t: int = 170,
) -> Union[SSLTransform, DINOTransform]:
    """retunrs train image transformations class

    Args:
        framework (str): self-supervised framework name (DINO/BYOL)

    Returns:
        Union[SSLTransform, DINOTransform]: train transformation
    """
    
    if framework == "dino":
        return DINOTransform(
            img_size=img_size,
            local_crop_size=local_crop_size,
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            mean=mean,
            std=std,
            crop_resize_p=crop_resize_p,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            color_jitter_p=color_jitter_p,
            grayscale_p=grayscale_p,
            h_flip_p=h_flip_p,
            kernel=kernel,
            sigma=sigma,
            solarization_p=solarization_p,
            solarize_t=solarize_t
        )
    
    if framework == "byol":
        return SSLTransform(
            train=train,
            img_size=img_size,
            mean=mean,
            std=std,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            color_jitter_p=color_jitter_p,
            grayscale_p=grayscale_p,
            h_flip_p=h_flip_p,
            kernel=kernel,
            sigma=sigma,
            gaussian_blur_p=gaussian_blur_p
        )    
    
    print(f"{framework} not supported.")
    quit()

