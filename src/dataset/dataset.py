import os
import random
from src.io.io import read_rgb
from torch.utils.data import Dataset
from typing import Callable, Tuple, List

class SSLDataset(Dataset):
    
    EXTENSIONS = (
        "jpg",
        "jpeg",
        "png",
        "ppm",
        "bmp",
        "pgm",
        "tif",
        "tiff",
    )
    
    def __init__(
        self,
        root_dir: str, 
        split: str,
        with_folders: bool = True,
        max_samples: int = None,
        random_samples: bool = False,
        transform: Callable = None
    ) -> None:
        """Self Supervised Dataset support

        Args:
            root_dir (str): data dir
            split (str): one of train, val and test.
            with_folders (bool, optional): if dataset has folders in it. Defaults to True. 
            max_samples_per_class (int, optional): max number of samples for the dataset. Defaults to None.
            random_samples (bool, optional): if selecting randomnly the max samples. Defaults to False.
            transform (Callable, optional): self-sup transform function. Defaults to None.
        """
        
        assert split in ["train", "val", "test"], print(f"Split must be one of train, val, test. Not {split}.")
        
        self.data_dir = os.path.join(root_dir, split)
        assert os.path.exists(self.data_dir), print(f"{self.data_dir} does not exists.")
        print(f"-"*40)
        print(f"> Loading dataset from {self.data_dir}")
        self.img_paths = self._load_samples(
            data_dir=self.data_dir,
            with_folders=with_folders,
            max_samples=max_samples,
            random_samples=random_samples
        )
        print(f"> Dataset size: {len(self.img_paths)} samples.")
        print(f"-"*40)
        self.transform = transform
    
    def _load_samples(
        self,
        data_dir: str,
        with_folders: bool = True,
        max_samples: int = None,
        random_samples: bool = False,
    ) -> List[str]:
        """loads samples from data_dir also when folders are in it

        Args:
            data_dir (str): dataset root dir
            with_folders (bool, optional): if dataset root dir has folders. Defaults to True.
            max_samples (int, optional): max number of samples. Defaults to None.
            random_samples (bool, optional): if selecting random samples if max_samples is not None. Defaults to False.

        Returns:
            List[str]: list of image paths
        """
        
        paths = [ os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.split(".")[-1].lower() in self.EXTENSIONS ]
        print(f"\t> Found {len(paths)} images in main folder.")
        if with_folders:
            folders = [ f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) ]
            print(f"\t> Loading images also from subfolders: {folders}")
            for f in folders:
                f_dir = os.path.join(data_dir, f)
                f_paths = [ os.path.join(f_dir, img) for img in os.listdir(f_dir) if img.split(".")[-1].lower() in self.EXTENSIONS ]
                paths.extend(f_paths)
        
        if max_samples is not None:
            if len(paths) > max_samples:
                old_len = len(paths)
                paths = paths[:max_samples] if not random_samples else random.sample(paths, k=max_samples)
                print(f"\t> Reducing number of samples from {old_len} to {len(paths)}.")
        return paths
     
    def __getitem__(self, index) -> Tuple:
        
        img_path = self.img_paths[index]
        #label = self.targets[index]
            
        img = read_rgb(img_path)
        views = None
        if self.transform:
            img, views = self.transform(img)
        
        return img, views
    
    def __len__(self) -> int:
        return len(self.img_paths)
        