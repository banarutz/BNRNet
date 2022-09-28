from imports import *

####################


class doubleConvBlock (nn.Module):

    def __init__ (self, in_channel, out_channel, mid_channel = None):
        super().__init__()

        if not mid_channel:
            mid_channel = out_channel

        self.doubleConv = nn.Sequential(

            nn.Conv2d (in_channel, mid_channel, kernel_size= 3, padding = 1, bias = False),
            nn.BatchNorm2d (mid_channel),
            nn.ReLU (inplace=True),
            nn.Conv2d (mid_channel, out_channel, kernel_size= 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU (inplace=True)
# De ce e padding = 1?
        )

    def forward(self, x):
        return self.doubleConv(x)


class Down (nn.Module):

    def __init__ (self, in_channel, out_channel):
        super().__init__()

        self.down = nn.Sequential (
            
            nn.MaxPool2d(kernel_size= 2),
            doubleConvBlock (in_channel, out_channel)
    
        )

    def forward (self, x):
        return self.down(x)



class Up (nn.Module):

    def __init__ (self, in_channel, out_channel, upsample = 'conv'):

        """
        Up module, upsample argument takes:
                'conv' - for upsampling with Conv2dTranspose;
                'bilinear' - for upsampling with Upsample layer
        """
        super().__init__()

        if out_channel != 2 * in_channel:
            raise ValueError("Out channel must be 2 * in_channel")

        if upsample == 'bilinear': 

            self.up = nn.Upsample(scale_factor= 2, mode = 'bilinear', align_corners= True),
            self.conv = doubleConvBlock (in_channel, out_channel)
            

        elif upsample == 'conv':

            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size = 2, stride = 2),
            self.conv = doubleConvBlock (in_channel, out_channel)
             
    def forward(self, x1, x2):

        x1 = self.up (x1)
    # IN implementarea de pe git, aici mai avea ceva padding ??? why?
        x = torch.cat ([x2, x1], dim = 1)

        return self.conv(x)

class out_conv (nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channels= 3, kernel_size= 1)
    
    def forward(self, x):
        return self.conv(x)


class UNet (nn.Module):
    
    def __init__(self, in_channel, out_channels):
        super(UNet, self).__init__()
        

