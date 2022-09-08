import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2D(nn.Module):
    def __init__(self, nin, nout):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_dim):
        super(BottleNeck, self).__init__()

        self.bottle_neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.bottle_neck(x)
        x = torch.reshape(x, (x.size()[0], 8, 8, 8))
        return x


class DoubleConv(nn.Module):
  # (convolution => bn => relu)*2
  def __init__(self, inc, outc, midc=None):
    super(DoubleConv, self).__init__()
    if not midc:
      midc = outc
    self.double_conv = nn.Sequential(
      SeparableConv2D(inc, midc),
      nn.BatchNorm2d(midc),
      nn.ReLU(inplace=True),
      SeparableConv2D(midc, outc),
      nn.BatchNorm2d(outc),
      nn.ReLU(inplace=True)
    )


  def forward(self, x):
    return self.double_conv(x)

  

class Down(nn.Module):
  # (maxpool for (h, w) not d => DoubleConv)
  def __init__(self, inc, outc):
    super(Down, self).__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)),
      DoubleConv(inc, outc)
    )

  def forward(self, x):
    # x.size() = (batch, inc, d, h, w) => (batch, outc, d, h//2, w//2) 
    return self.maxpool_conv(x)


class Up(nn.Module):
  # ( ConvTranspose3d for (h, w) dimension, not d => DoubleConv)
  def __init__(self, inc1, inc2, outc):
    super(Up, self).__init__()

    # since inc is after torch.cat of x1 and x2, self.up will halve the channel
    self.up = nn.ConvTranspose2d(inc1, inc2//2, kernel_size=(2,2), stride=(2,2))
    self.test = nn.Conv2d(inc2, inc2//2, kernel_size=(1,1), stride=(1,1))
    self.conv = DoubleConv(inc2, outc)


  def forward(self, x1, x2):
    # x1.size() = (batch, inc//2, d, h, w]
    # x2.size() = (batch, inc//2, d, h, w]
    x1 = self.up(x1)
    x2 = self.test(x2)

    # input is CHW, [bn, c, h, w]

    diffX = x2.size()[2] - x1.size()[2]
    diffY = x2.size()[3] - x1.size()[3]


    #print("diffD")
    #print(diffD)
    #print("diffX")
    #print(diffX)
    #print("diffY")
    #print(diffY)

    padding_list = []

    assert not diffX < 0
    assert not diffY < 0

    if diffY == 1:
      padding_list.append(1)
      padding_list.append(0)
    else:
      padding_list.append(diffY//2)
      padding_list.append(diffY-diffY//2)

    if diffX == 1:
      padding_list.append(1)
      padding_list.append(0)
    else:
      padding_list.append(diffX//2)
      padding_list.append(diffX-diffX//2)




    #print("****"*20)
    #print(padding_list)

    

    x1 = F.pad(x1, padding_list)

    #x1 = F.pad(x1, [diffD//2, diffD-diffD//2, diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])

    #print("dude"*10)
    #print(x1.size())
    #print(x2.size())

    # x.size() = (batch, cin, d, h, w)
    x = torch.cat([x1, x2], dim=1)
    
    # self.conv(x).size() = (batch, cin//2, d, 2*h, 2*w)
    return self.conv(x)


class OutConv(nn.Module):
  # last conv which maps to the same dimension of target
  # use kernel_size=1, padding=0, stride=1
  def __init__(self, inc, outc):
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(inc, outc, kernel_size=1)
  
  def forward(self, x):
    # x.size() = (batch, inc, d, h, w)
    # self.conv(x).size() = (batch, outc, d, h, w)
    return self.conv(x)
    
    

class UNet2D(nn.Module):
  '''
    n_channels = c of input = 3
    n_classes = c of target = 1
  '''
  def __init__(self, n_channels, n_classes):
    super(UNet2D, self).__init__() 
    self.n_channels = n_channels
    self.n_classes = n_classes
    
    self.inc = DoubleConv(n_channels, 64)

    self.down1 = Down(64, 96) # 128 -> 64
    self.down2 = Down(96, 144) # 64 -> 32
    self.down3 = Down(144, 216) # 32 -> 16
    self.down4 = Down(216, 216) # 16 -> 8

    self.bottle_neck  = BottleNeck(in_dim = 8*8*216)
    
    self.up1 = Up(8, 216, 216)
    self.up2 = Up(216, 144, 144)
    self.up3 = Up(144, 96, 96)
    self.up4 = Up(96, 64, 64)

    self.outc = OutConv(64, n_classes)


  def forward(self, x):
    #print(f"x: {x.size()}")
    x1 = self.inc(x)
    #print(f"x1: {x1.size()}")
    x2 = self.down1(x1)
    #print(f"x2: {x2.size()}")
    x3 = self.down2(x2)
    #print(f"x3: {x3.size()}")
    x4 = self.down3(x3)
    #print(f"x4: {x4.size()}")
    x5 = self.down4(x4)
    #print(f"x5: {x5.size()}")
    bottom = self.bottle_neck(x5)
    #print(f"bottom: {bottom.size()}")

    x = self.up1(bottom, x4)
    #print(f"x: {x.size()}")
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)

    logits = self.outc(x)

    return logits


if __name__ == "__main__":
    x = torch.rand((1, 3, 128, 128))
    print("here")
    print(x.shape)
    model = UNet2D(3, 1)
    pred = model(x)
    print(pred.shape)
