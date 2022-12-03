import torch

def collate_fn(batch):
    
    imgs = torch.stack([torch.tensor(x[0]) for x in batch], dim=0)
    views = [ [] for _ in range(len(batch[0][1]))]
    for x in batch:
        for i, view in enumerate(x[1]):
            views[i].append(torch.tensor(view))
    views = [torch.stack(view) for view in views]
    #labels = torch.stack([torch.tensor(x[2]) for x in batch], dim=0)
    
    return imgs, views