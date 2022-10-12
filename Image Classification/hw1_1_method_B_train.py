import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class DLCV_HW1_Q1_Dataset(Dataset):
    def __init__(self, img_folder, tfm):
        super(DLCV_HW1_Q1_Dataset, self).__init__()
        self.img_folder = img_folder
        self.tfm = tfm
        self.img_name_list = os.listdir(img_folder)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img = Image.open(os.path.join(self.img_folder, img_name))
        img = self.tfm(img)

        try:
            label  = int(img_name.split('_')[0])
        except:
            label = -1

        return img, label

test_tfm = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
])

train_tfm = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
])

batch_size = 128

train_dataset = DLCV_HW1_Q1_Dataset('/content/hw1_data/p1_data/train_50', train_tfm)
valid_dataset = DLCV_HW1_Q1_Dataset('/content/hw1_data/p1_data/val_50', test_tfm)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

lr = 0.00001

model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(2048, 50)
model = model.to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

n_epochs = 50
best_acc = 0

for epoch in range(n_epochs):
    model.train()
    train_loss, train_acc = [], []
    for image, label in train_dataloader:
        pred = model(image.to('cuda'))
        loss = loss_fn(pred, label.to('cuda'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (pred.argmax(dim=-1) == label.cuda()).float().mean()
        train_loss.append(loss.item())
        train_acc.append(acc)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)
    print(f"[ Train | {epoch+1:03d}/{n_epochs} ] loss = {train_loss:.03f}, acc = {train_acc:.03f}")

    model.eval()
    valid_loss, valid_acc = [], []
    for image, label in valid_dataloader:
        with torch.no_grad():
            pred = model(image.to('cuda'))
            loss = loss_fn(pred, label.to('cuda'))
        acc = (pred.argmax(dim=-1) == label.cuda()).float().mean()
        valid_loss.append(loss.item())
        valid_acc.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_acc) / len(valid_acc)
    print(f"[ Valid | {epoch+1:03d}/{n_epochs} ] loss = {valid_loss:.03f}, acc = {valid_acc:.03f}")

    if valid_acc >= best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), './best_3.ckpt')
        print(f"save model with {best_acc:.3f} val_acc...")