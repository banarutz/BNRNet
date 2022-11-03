import torch.nn as nn
import torch.nn.functional as F

class VGGBlock1 (nn.Module):
    """
    For the 2 input blocks, only 2 Conv2d layers.
    """
    def __init__(self, in_ch, out_ch, k_size, simple):
        super().__init__()

        if simple == False:

            self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 1, bias = True),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = 2, padding = 1, bias = True),
                nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                # nn.MaxPool2d(kernel_size= 2),
            )
        
        elif simple == True:

             self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 'same'),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 'same'),
                nn.ReLU(True), 
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return self.block(x)

class VGGBlock2 (nn.Module):
    """
    This block has 3 Conv2d.
    """
    def __init__(self, in_ch, out_ch, k_size, simple):
        super().__init__()

        if simple == False:

            self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 1, bias = True),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 1, bias = True),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 1, bias = True),
                nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                # nn.MaxPool2d(kernel_size= 2),
            )
        
        elif simple == True:

             self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 1, bias = True),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 1, bias = True),
                nn.ReLU(True),
                nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = 2, padding = 1, bias = True),
                nn.ReLU(True), 
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return self.block(x)


class VGGBlock3 (nn.Module):
    """
    Last but one block.
    """
    def __init__(self, in_ch, out_ch, k_size, simple):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 1, bias = True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 1, bias = True),
            nn.ReLU(True),
            nn.Conv2d(in_channels = out_ch, out_channels= out_ch, kernel_size = k_size, stride = 1, padding = 1, bias = True),
            nn.ReLU(True),
            nn.BatchNorm2d(out_ch),
            # nn.MaxPool2d(kernel_size= 2),
            )

    def forward(self, x):
        return self.block(x)


class VGGNet (nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = VGGBlock1 (1, 64, 3, True)
        self.block2 = VGGBlock1 (64, 128, 3, True)
        self.block3 = VGGBlock2 (128, 256, 3, True)
        self.block4 = VGGBlock2 (256, 512, 3, False)
        self.block5 = VGGBlock2 (512, 512, 3, True)
        self.block6 = VGGBlock2 (512, 512, 3, False)
        self.block7 = VGGBlock2 (512, 512, 3, False)
        self.block8 = VGGBlock3 (512, 256, 3, True)
        self.block9 = nn.Conv2d(256, 313, kernel_size=1, stride=1, padding='same', bias=True)
        self.upsample = nn.Upsample((218, 178), mode='bilinear')
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
        out = self.model_out (self.softmax(l))
        out = self.upsample (out)
        return out


class ECCVGenerator(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(input_l)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        print(conv8_3.shape)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.upsample4(out_reg)
        

        


