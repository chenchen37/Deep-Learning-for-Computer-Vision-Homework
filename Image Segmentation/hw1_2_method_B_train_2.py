import os
import torch
import torch.nn as nn
import imageio
from PIL import Image
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


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
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
])

def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)


# Reference: https://github.com/ternaus/TernausNet/blob/master/ternausnet/models.py
class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Interpolate(nn.Module):
    def __init__(
        self,
        size: int = None,
        scale_factor: int = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class DecoderBlockV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        is_deconv: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet16(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        num_filters: int = 32,
        pretrained: bool = True,
        is_deconv: bool = False,
    ):
        """
        Args:
            num_classes:
            num_filters:
            pretrained:
                False - no pre-trained network used
                True - encoder pre-trained with VGG16
            is_deconv:
                False: bilinear interpolation is used in decoder
                True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu
        )

        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu
        )

        self.conv3 = nn.Sequential(
            self.encoder[10],
            self.relu,
            self.encoder[12],
            self.relu,
            self.encoder[14],
            self.relu,
        )

        self.conv4 = nn.Sequential(
            self.encoder[17],
            self.relu,
            self.encoder[19],
            self.relu,
            self.encoder[21],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[24],
            self.relu,
            self.encoder[26],
            self.relu,
            self.encoder[28],
            self.relu,
        )

        self.center = DecoderBlockV2(
            512, num_filters * 8 * 2, num_filters * 8, is_deconv
        )

        self.dec5 = DecoderBlockV2(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
        )
        self.dec4 = DecoderBlockV2(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
        )
        self.dec3 = DecoderBlockV2(
            256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv
        )
        self.dec2 = DecoderBlockV2(
            128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv
        )
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


batch_size = 3

train_dataset = DLCV_HW1_Q2_Dataset('./hw1_data/p2_data/train', train_tfm)
valid_dataset = DLCV_HW1_Q2_Dataset('./hw1_data/p2_data/validation', test_tfm)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

device = 'cuda'
lr = 1e-5
weight_decay = 1e-4

model = UNet16()
model = model.to(device)
loss_fn = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

n_epochs = 25
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
        torch.save(model.state_dict(), 'unet16_focalLoss_gb_lr1e5_epoch'+na+'.ckpt')
    else:
        print(f"[ {epoch+1:02d}/{n_epochs} ] train loss = {train_loss:.03f}, valid loss = {valid_loss:.03f}")
        na = str(epoch+1)
        torch.save(model.state_dict(), 'unet16_focalLoss_gb_lr1e5_epoch'+na+'.ckpt')

print(f"best epoch: {best_epoch}, {best_loss:.03f}")