import os
import torch
import torch.nn as nn
import imageio
from PIL import Image
import torch.nn.functional as F
import torchvision
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

# ref: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self, alpha=None, gamma: float = 0., reduction: str = 'mean', ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, pred, mask):
        if pred.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = pred.shape[1]
            pred = pred.permute(0, *range(2, pred.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            mask = mask.view(-1)

        unignored_mask = mask != self.ignore_index
        mask = mask[unignored_mask]
        if len(mask) == 0:
            return torch.tensor(0.)
        pred = pred[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(pred, dim=-1)
        ce = self.nll_loss(log_p, mask)

        # get true class column from each row
        all_rows = torch.arange(len(pred))
        log_pt = log_p[all_rows, mask]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

test_tfm = transforms.Compose([
    transforms.ToTensor(),
])

train_tfm = transforms.Compose([
    transforms.ToTensor(),
])


#Reference: https://github.com/wuchangsheng951/satellite_segmentation/blob/master/models/unet/unet.py?fbclid=IwAR30lU5vbCWc0BJPjBJ8WC2IY4B3MbwkeOSl5AGWEHOpOqijs6O0txLQhx4
class ConvRelu(nn.Module):
    """3x3 convolution followed by ReLU activation building block.
    """

    def __init__(self, num_in, num_out):
        """Creates a `ConvReLU` building block.
        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """

        return nn.functional.relu(self.block(x), inplace=True)


class DecoderBlock(nn.Module):
    """Decoder building block upsampling resolution by a factor of two.
    """

    def __init__(self, num_in, num_out):
        """Creates a `DecoderBlock` building block.
        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """

        return self.block(nn.functional.upsample(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """The "U-Net" architecture for semantic segmentation, adapted by changing the encoder to a ResNet feature extractor.
       Also known as AlbuNet due to its inventor Alexander Buslaev.
    """

    def __init__(self, num_classes=7, num_filters=32, pretrained=True):
        """Creates an `UNet` instance for semantic segmentation.
        Args:
          num_classes: number of classes to predict.
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        # Todo: make input channels configurable, not hard-coded to three channels for RGB

        self.resnet = models.resnet50(pretrained=pretrained)

        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.center = DecoderBlock(2048, num_filters * 8)

        self.dec0 = DecoderBlock(2048 + num_filters * 8, num_filters * 8)
        self.dec1 = DecoderBlock(1024 + num_filters * 8, num_filters * 8)
        self.dec2 = DecoderBlock(512 + num_filters * 8, num_filters * 2)
        self.dec3 = DecoderBlock(256 + num_filters * 2, num_filters * 2 * 2)
        self.dec4 = DecoderBlock(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """
        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        center = self.center(nn.functional.max_pool2d(enc4, kernel_size=2, stride=2))

        dec0 = self.dec0(torch.cat([enc4, center], dim=1))
        dec1 = self.dec1(torch.cat([enc3, dec0], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)

        return self.final(dec5)


batch_size = 3

train_dataset = DLCV_HW1_Q2_Dataset('./hw1_data/p2_data/train', train_tfm)
valid_dataset = DLCV_HW1_Q2_Dataset('./hw1_data/p2_data/validation', test_tfm)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

device = 'cuda'
lr = 1e-4
weight_decay = 1e-4

model = UNet()
model = model.to(device)
loss_fn = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

n_epochs = 50
best_loss = 10
best_epoch = 0

for epoch in range(n_epochs):
    model.train()
    train_running_loss = []
    for sat, mask, id in train_dataloader:
        sat = sat.to(device)
        mask = mask.to(device)
        pred = model(sat)
        loss = loss_fn(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_running_loss.append(loss.item())
    train_loss = sum(train_running_loss) / len(train_running_loss)
    
    model.eval()
    valid_running_loss = []
    for sat, mask, id in valid_dataloader:
        sat = sat.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            pred = model(sat)
            loss = loss_fn(pred, mask)
            valid_running_loss.append(loss.item())
    valid_loss = sum(valid_running_loss) / len(valid_running_loss)

    if valid_loss <= best_loss:
        print(f"[ {epoch+1:02d}/{n_epochs} ] train loss = {train_loss:.03f}, valid loss = {valid_loss:.03f} <- best")
        best_loss = valid_loss
        best_epoch = epoch + 1
        na = str(epoch+1)
        torch.save(model.state_dict(), 'res_unet_focalLoss_lr1e4_epoch'+na+'.ckpt')
    else:
        print(f"[ {epoch+1:02d}/{n_epochs} ] train loss = {train_loss:.03f}, valid loss = {valid_loss:.03f}")
        na = str(epoch+1)
        torch.save(model.state_dict(), 'res_unet_focalLoss_lr1e4_epoch'+na+'.ckpt')

print(f"best epoch: {best_epoch}, {best_loss:.03f}")