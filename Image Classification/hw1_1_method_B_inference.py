import os
import csv
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
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

        return img, label, img_name

def results_gen(model, loader, csv_output_path):
    model.eval()
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for image, label, img_name in loader:
            with torch.no_grad():
                pred = model(image.to('cuda'))
                pred = pred.argmax(dim=-1)
                for i in range(image.size(0)):
                    writer.writerow([img_name[i], pred[i].item()])
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_dir', help='testing images directory', type=str)
    parser.add_argument('-o', '--out_path', help='path of output csv file', type=str)
    args = parser.parse_args()

    batch_size = 128

    test_tfm = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
    ])

    valid_dataset = DLCV_HW1_Q1_Dataset(args.test_dir, test_tfm)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = models.resnext101_32x8d()
    model.fc = nn.Linear(2048, 50)
    model.load_state_dict(torch.load('./best_3.ckpt'))
    model = model.to('cuda')

    results_gen(model, valid_dataloader, args.out_path)

