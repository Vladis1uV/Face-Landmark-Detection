from torch.utils.data import DataLoader, random_split
from .dataset import FaceLandmarksDataset
from .transforms import Transforms

def build_dataloaders(
        root_dir,
        batch_size,
        val_split=0.1,
        num_workers=4,
):
    dataset = FaceLandmarksDataset(root_dir=root_dir, transform=Transforms())

    len_valid = int(val_split * len(dataset))
    len_train = int(len(dataset)) - len_valid

    train_dataset, valid_dataset = random_split(dataset, [len_train, len_valid])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, valid_dataloader