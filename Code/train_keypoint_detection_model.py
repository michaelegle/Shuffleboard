import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models
import os

class ShuffleboardDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.img_dir}/{row['image']}").convert("RGB")
        img = self.transform(img)
        # 16 values: x1,y1,x2,y2,...,x8,y8 already normalized 0-1
        keypoints = torch.tensor(row[1:].values.astype(float), dtype=torch.float32)
        return img, keypoints

class KeypointCNN(nn.Module):
    def __init__(self, num_keypoints=8):
        super().__init__()
        # pretrained ResNet as backbone
        backbone = models.resnet34(weights="IMAGENET1K_V1")
        # drop the classification head
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # regression head outputs x,y per keypoint
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_keypoints * 2),
            nn.Sigmoid()  # keeps outputs in 0-1 range
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

from torch.utils.data import DataLoader

def train(model, train_loader, val_loader, epochs=50, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, kps in train_loader:
            imgs, kps = imgs.to(device), kps.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, kps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, kps in val_loader:
                imgs, kps = imgs.to(device), kps.to(device)
                val_loss += criterion(model(imgs), kps).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")

# kick it off
model = KeypointCNN(num_keypoints=8)

print(os.getcwd())

train_ds = ShuffleboardDataset("../Data/keypoint_detection/labels/train.csv", "../Training Images")
val_ds   = ShuffleboardDataset("../Data/keypoint_detection/labels/val.csv", "../Training Images")
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16)
train(model, train_loader, val_loader)

torch.save(model.state_dict(), "../Models/keypoint_detection/keypoint_model.pth")
