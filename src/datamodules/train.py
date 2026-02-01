import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.resnet18 import Network
from pathlib import Path
from build import build_dataloaders


ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
ckpt_dir.mkdir(exist_ok=True)

train_loader, valid_loader = build_dataloaders(
    root_dir="data/",
    batch_size=64,
    val_split=0.1,
    num_workers=4 
)

torch.autograd.set_detect_anomaly(True)
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network()
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss_min = np.inf
num_epochs = 10

for epoch in range(1, num_epochs+1):

    loss_train = 0
    loss_valid = 0
    
    model.train()
    for images, landmarks in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        
        images.to(device)
        landmarks = landmarks.view(landmarks.size(0), -1).to(device)

        predictions = model(images)

        optimizer.zero_grad()

        loss_train_step = criterion(predictions, landmarks)
        loss_train_step.backward()

        optimizer.step()

        loss_train += loss_train_step.item()
    
    loss_train /= len(train_loader)

    model.eval()
    with torch.no_grad():
        for images, landmarks in tqdm(valid_loader, desc=f"Epoch {epoch} [valid]"):

            images.to(device)
            landmarks = landmarks.view(landmarks.size(0), -1).to(device)

            predictions = model(images)
            loss_valid_step = criterion(predictions, landmarks)

            loss_valid += loss_valid_step.item()

    loss_valid /= len(valid_loader)

    writer.add_scalar("Loss/train", loss_train, epoch)
    writer.add_scalar("Loss/valid", loss_valid, epoch)

if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(model.state_dict(), ckpt_dir / "best.pth")
        print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
        print('Model Saved\n')