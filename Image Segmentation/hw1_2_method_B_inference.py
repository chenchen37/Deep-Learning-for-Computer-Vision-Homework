import os
import torch
import torch.nn as nn
import numpy as np
import imageio
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import argparse

class DLCV_HW1_Q2_Dataset(Dataset):
    def __init__(self, img_folder, tfm):
        super(DLCV_HW1_Q2_Dataset, self).__init__()
        self.img_folder = img_folder
        self.tfm = tfm
        self.ID_list = [file for file in os.listdir(img_folder) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        sat_img = Image.open(os.path.join(self.img_folder, ID))
        sat_img = self.tfm(sat_img)

        ID = ID.split('.')[0]

        return sat_img, int(ID)



def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)


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

        self.encoder = models.vgg16(pretrained=pretrained).features

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

class ConvRelu_ResNet(nn.Module):
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


class DecoderBlock_ResNet(nn.Module):
    """Decoder building block upsampling resolution by a factor of two.
    """

    def __init__(self, num_in, num_out):
        """Creates a `DecoderBlock` building block.
        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = ConvRelu_ResNet(num_in, num_out)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """

        return self.block(nn.functional.upsample(x, scale_factor=2, mode="nearest"))


class UNet_Res(nn.Module):
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

        self.center = DecoderBlock_ResNet(2048, num_filters * 8)

        self.dec0 = DecoderBlock_ResNet(2048 + num_filters * 8, num_filters * 8)
        self.dec1 = DecoderBlock_ResNet(1024 + num_filters * 8, num_filters * 8)
        self.dec2 = DecoderBlock_ResNet(512 + num_filters * 8, num_filters * 2)
        self.dec3 = DecoderBlock_ResNet(256 + num_filters * 2, num_filters * 2 * 2)
        self.dec4 = DecoderBlock_ResNet(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu_ResNet(num_filters, num_filters)

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_img_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    args = parser.parse_args()

    test_tfm = transforms.Compose([
    transforms.ToTensor()
    ])

    batch_size = 2
    device = 'cuda'

    valid_dataset = DLCV_HW1_Q2_Dataset(args.test_img_dir, test_tfm)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model1 = UNet16(is_deconv=False)
    model1.load_state_dict(torch.load('./unet16_ceLoss_gb_lr1e5_epoch14.ckpt'))
    model1 = model1.to(device)

    model2 = UNet16(is_deconv=False)
    model2.load_state_dict(torch.load('./unet16_focalLoss_gb_lr1e5_epoch17.ckpt'))
    model2 = model2.to(device)

    model3 = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model3.classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    model3.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    model3.load_state_dict(torch.load('./seg_epoch6.ckpt'))
    model3 = model3.to(device)

    model5 = UNet_Res()
    model5.load_state_dict(torch.load('./res_unet_focalLoss_lr1e4_epoch15.ckpt'))
    model5 = model5.to(device)

    model7 = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model7.classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    model7.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    model7.load_state_dict(torch.load('./deeplabv3_resnet101_focalLoss_lr1e5_epoch20.ckpt'))
    model7 = model7.to(device)

    model8 = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model8.classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    model8.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    model8.load_state_dict(torch.load('./deeplabv3_resnet101_ceLoss_lr1e5_epoch32.ckpt'))
    model8 = model8.to(device)

    model9 = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model9.classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    model9.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    model9.load_state_dict(torch.load('./deeplabv3_resnet101_ceLoss_blur_lr1r5_epoch20.ckpt'))
    model9 = model9.to(device)

    model1.eval()
    model2.eval()
    model3.eval()
    model5.eval()
    model7.eval()
    model8.eval()
    model9.eval()

    for sat, id in valid_dataloader:
        sat = sat.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            pred = model1(sat)
            pred2 = model2(sat)
            pred3 = model3(sat)['out']
            pred5 = model5(sat)
            pred7 = model7(sat)['out']
            pred8 = model8(sat)['out']
            pred9 = model9(sat)['out']

            pred = F.softmax(pred, dim=1)
            pred2 = F.softmax(pred2, dim=1)
            pred3 = F.softmax(pred3, dim=1)
            pred5 = F.softmax(pred5, dim=1)
            pred7 = F.softmax(pred7, dim=1)
            pred8 = F.softmax(pred8, dim=1)
            pred9 = F.softmax(pred9, dim=1)

            pred = pred + pred2 + pred3 + pred5 + 1.6*pred7 + 1.6*pred8 + 1.6*pred9

            pred = torch.argmax(pred, dim=1)
            pred = pred.detach().cpu().numpy()
        
        for aa in range(sat.size(0)):
            save_mask = np.zeros((512, 512, 3))
            for i in range(7):
                if i == 0:
                    tar = np.array([0, 255, 255])
                elif i == 1:
                    tar = np.array([255, 255, 0])
                elif i == 2:
                    tar = np.array([255, 0, 255])
                elif i == 3:
                    tar = np.array([0, 255, 0])
                elif i == 4:
                    tar = np.array([0, 0, 255])
                elif i == 5:
                    tar = np.array([255, 255, 255])
                elif i == 6:
                    tar = np.array([0, 0, 0])

                idx1, idx2 = np.where(pred[aa]==i)
                for j in range(idx1.shape[0]):
                    save_mask[idx1[j], idx2[j], :] = tar

            imageio.imsave(os.path.join(args.output_dir, str(id[aa])+'.png'), np.uint8(save_mask))

