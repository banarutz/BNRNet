import torch.nn as nn
import torch.nn.functional as F

class VGG16 (nn.Module):
    def __init__(self):
        super().__init__()

    # def conv_relu(in_channels, out_channels, ksize, stride):
    #     block = nn.Sequential (nn.Conv2d(in_channels = 1, out_channels= 64, kernel_size = 3, stride = 1),
    #         nn.ReLU())
    #     return block
    pass

class VGGBlock1 (nn.Module):
    """
    For the 2 input blocks, only 2 Conv2d layers.
    """
    def __init__(self, in_ch, out_ch, k_size, stride, simple):
        super().__init__()

        if simple == False:

            self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same'),
                nn.ReLU(),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same'),
                nn.ReLU(),
                nn.BatchNorm2d(),
                nn.MaxPool2d(kernel_size= 2)
            )
        
        elif simple == True:

             self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same'),
                nn.ReLU(),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same'),
                nn.ReLU(), 
                nn.BatchNorm2d()
            )

    def forward(self, x):
        return self.block(x)

class VGGBlock2 (nn.Module):
    """
    This block has 3 Conv2d.
    """
    def __init__(self, in_ch, out_ch, k_size, stride, simple):
        super().__init__()

        if simple == False:

            self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same', bias = True),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same', bias = True),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same', bias = True),
                nn.ReLU(True),
                nn.BatchNorm2d(),
                nn.MaxPool2d(kernel_size= 2)
            )
        
        elif simple == True:

             self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same', bias = True),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same', bias = True),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same', bias = True),
                nn.ReLU(True), 
                nn.BatchNorm2d()
            )

    def forward(self, x):
        return self.block(x)


class VGGBlock3 (nn.Module):
    """
    Last but one block.
    """
    def __init__(self, in_ch, out_ch, k_size, stride, simple):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same', bias = True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same', bias = True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = stride, padding = 'same', bias = True),
            nn.ReLU(True),
            nn.BatchNorm2d(),
            nn.MaxPool2d(kernel_size= 2)
            )

    def forward(self, x):
        return self.block(x)


class VGGNet (nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        self.block1 = VGGBlock1 (1, 64, 3, 1, True)
        self.block2 = VGGBlock1 (64, 128, 3, 1, True)
        self.block3 = VGGBlock2 (128, 256, 3, 1, True)
        self.block4 = VGGBlock2 (256, 512, 3, 1, False)
        self.block5 = VGGBlock2 (512, 512, 3, 1, True)
        self.block6 = VGGBlock2 (512, 512, 3, 1, True)
        self.block7 = VGGBlock2 (512, 512, 3, 1, True)
        self.block8 = VGGBlock3 (512, 256, 3, 1, True)
        self.block9 = nn.Sequential (
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.softmax = nn.Softmax (dim = 1)

    def forward (self, input):
        l = self.block1(input)
        l = self.block2(l)
        l = self.block3(l)
        l = self.block4(l)
        l = self.block5(l)
        l = self.block6(l)
        l = self.block7(l)
        l = self.block8(l)
        l = self.block9(l)

        out = self.softmax (self.model_out(l))

        return self.upsample(out)

        
        

        


