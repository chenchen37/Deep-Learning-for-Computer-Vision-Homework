import os
import torch
import torch.nn as nn
import imageio
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


class DLCV_HW1_Q2_Dataset(Dataset):
    def __init__(self, img_folder, tfm):
        super(DLCV_HW1_Q2_Dataset, self).__init__()
        self.img_folder = img_folder
        self.tfm = tfm
        self.ID_list = [file[:4] for file in os.listdir(img_folder) if file.endswith('.jpg')]


    def __len__(self):
        return len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        sat_img = Image.open(os.path.join(self.img_folder, ID+'_sat.jpg'))
        sat_img = self.tfm(sat_img)

        mask = torch.ones(512, 512)
        mask[:] = 6
        mask_img = imageio.v2.imread(os.path.join(self.img_folder, ID+'_mask.png'))
        mask_img = (mask_img >= 128).astype(int)
        mask_img = 4 * mask_img[:, :, 0] + 2 * mask_img[:, :, 1] + mask_img[:, :, 2]
        mask[mask_img == 3] = 0  # (Cyan: 011) Urban land 
        mask[mask_img == 6] = 1  # (Yellow: 110) Agriculture land 
        mask[mask_img == 5] = 2  # (Purple: 101) Rangeland 
        mask[mask_img == 2] = 3  # (Green: 010) Forest land 
        mask[mask_img == 1] = 4  # (Blue: 001) Water 
        mask[mask_img == 7] = 5  # (White: 111) Barren land 
        mask[mask_img == 0] = 6  # (Black: 000) Unknown 

        return sat_img, mask.type(torch.LongTensor), ID


test_tfm = transforms.Compose([
    transforms.ToTensor(),
])

train_tfm = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
])

batch_size = 3

train_dataset = DLCV_HW1_Q2_Dataset('./hw1_data/p2_data/train', train_tfm)
valid_dataset = DLCV_HW1_Q2_Dataset('./hw1_data/p2_data/validation', train_tfm)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

lr = 1e-5
weight_decay = 1e-3

model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
model.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
model = model.to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

n_epochs = 25
best_loss = 10
best_epoch = 0

for epoch in range(n_epochs):
    model.train()
    train_running_loss = []
    for sat, mask, id in train_dataloader:
        sat = sat.to('cuda')
        mask = mask.to('cuda')
        pred = model(sat)['out']
        loss = loss_fn(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_running_loss.append(loss.item())
    train_loss = sum(train_running_loss) / len(train_running_loss)
    
    model.eval()
    valid_running_loss = []
    for sat, mask, id in valid_dataloader:
        sat = sat.to('cuda')
        mask = mask.to('cuda')
        with torch.no_grad():
            pred = model(sat)['out']
            loss = loss_fn(pred, mask)
            valid_running_loss.append(loss.item())
    valid_loss = sum(valid_running_loss) / len(valid_running_loss)

    if valid_loss <= best_loss:
        print(f"[ {epoch+1:02d}/{n_epochs} ] train loss = {train_loss:.03f}, valid loss = {valid_loss:.03f} <- best")
        best_loss = valid_loss
        best_epoch = epoch + 1
        na = str(epoch+1)
        torch.save(model.state_dict(), 'deeplabv3_resnet101_ceLoss_blur_lr1r5_epoch'+na+'.ckpt')
    else:
        print(f"[ {epoch+1:02d}/{n_epochs} ] train loss = {train_loss:.03f}, valid loss = {valid_loss:.03f}")
        na = str(epoch+1)
        torch.save(model.state_dict(), 'deeplabv3_resnet101_ceLoss_blur_lr1r5_epoch'+na+'.ckpt')

print(f"Best epoch: {best_epoch}, {best_loss:.03f}")