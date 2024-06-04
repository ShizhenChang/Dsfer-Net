import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math

epsilon = 1e-14

class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base,self).__init__()
        feats = list(models.vgg16(pretrained=True).features)[:30]
        self.feats = nn.ModuleList(feats).eval()

    def forward(self,x):
        feat = []
        for i, model in enumerate(self.feats):
            x = model(x)
            if i in {3,8,15,22,29}:
                feat.append(x)
        return feat




class Linear_qkv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Linear_qkv, self).__init__()
        self.linear_qkv = nn.Linear(dim_in, dim_out, bias= False)

    def forward(self, x):
        return self.linear_qkv(x)


class HopfieldRetrieve(nn.Module):
    def __init__(self, beta, dim_in_q, dim_in_k, dim_out_k, num_heads = 1, v = None, dim_in_v = 0, dim_out_v = 0, logits=False, activation = F.relu):
        super(HopfieldRetrieve, self).__init__()
        # Hopfield Retrieve can be combined with the concept of multi-head 
        self.beta = beta
        self.num_heads = num_heads
        self.activation = activation
        self.linear_q = Linear_qkv(dim_in_q, dim_out_k)
        self.linear_k = Linear_qkv(dim_in_k, dim_out_k)
        if v:
            self.linear_v = Linear_qkv(dim_in_v, dim_out_v) # If v = True, user need to define the dimention of v
        if self.beta % self.num_heads != 0:
            raise ValueError('`beta`({}) should be divisible by `num_heads`({})'.format(self.beta, self.num_heads))
        self._norm_fact = (self.beta // self.num_heads)** -.5
        self.softmax = F.softmax
        if logits:
            self.softmax = F.log_softmax

    def forward(self, q, k, v = None):
        q, k = self.linear_q(q), self.linear_k(k)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
        batch, n, dim_q = q.shape
        _, _, dim_k = k.shape
        assert dim_q == dim_k and dim_q % self.num_heads == 0

        dq = dim_q // self.num_heads  # dim_q of each head
        dk = dim_k // self.num_heads  # dim_k of each head

        q = q.reshape(batch, n, self.num_heads, dq).transpose(1, 2) # (batch, nh, n, dq)
        k = k.reshape(batch, n, self.num_heads, dk).transpose(1, 2) # (batch, nh, n, dk)
              
        # important: normalize k and q ! This could be made optional if need be.    
        E = self._norm_fact * torch.matmul(q, k.transpose(2,3)) # (batch, nh, n, n)
        E = self.softmax(E, dim = -1)
        
        if v is None:
            dist = torch.matmul(E, k) # (batch, nh, n, dk)
            return dist.transpose(1, 2).reshape(batch, n, dim_k)
            
        else:
            v = self.linear_v(v)
            if self.activation is not None:
                v = self.activation(v)
            _, _, dim_v = v.shape
            dv = dim_v // self.num_heads  # dim_v of each head
            assert dim_v % self.num_heads == 0
            #v = v/(torch.norm(v, dim = 1, keepdim = True) + epsilon)
            v = v.reshape(batch, n, self.num_heads, dv).transpose(1, 2) # (batch, nh, n, dv)
            dist = torch.matmul(E, v) # (batch, nh, n, dv)
            return dist.transpose(1, 2).reshape(batch, n, dim_v)


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
        )

    def forward(self, x):
        return self.single_conv(x)   

class SingleConv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
        )

    def forward(self, x):
        return self.single_conv(x)   
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)   

    
class MyNet(nn.Module):
    def __init__(self, n_classes=2, beta = 2.0, dim = 512, numhead = 1, n_channels=3, bilinear=True):
        super(MyNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.base = vgg16_base()
        self.hopf4 = HopfieldRetrieve(beta, 512, 512, dim, numhead, True, 512, 512, logits=False, activation = None)
        self.hopf5 = HopfieldRetrieve(beta, 512, 512, dim, numhead, True, 512, 512, logits=False, activation = None)
        self.UpSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sig = nn.Sigmoid()

        self.dec5 = SingleConv3(1024, 512)
        self.layernorm1 = nn.LayerNorm([512, 16, 16])
        #self.instancenorm1 = nn.InstanceNorm2d(512)
        self.dec51 = SingleConv3(512, 512)
        self.out5 =OutConv(512,n_classes)

        self.dec4 = SingleConv3(1024, 512)
        self.layernorm2 = nn.LayerNorm([512, 32, 32])
        #self.instancenorm2 = nn.InstanceNorm2d(512)
        self.dec41 = SingleConv3(1024, 256)
        self.out4 =OutConv(512,n_classes)

        self.dec3 = SingleConv3(512,256)
        self.dec31 = SingleConv3(512, 128)

        self.dec2 = SingleConv3(256,128)
        self.dec21 = SingleConv3(256, 64)

        self.dec1 = SingleConv3(128,64)
        self.dec11 = SingleConv3(128, 32)
        self.outc = OutConv(32, n_classes)
        # self.dec5 = SingleConv(32,16)
    def forward(self, xA, xB):
        list_t1 = self.base(xA)
        list_t2 = self.base(xB)
        xA1, xA2, xA3, xA4, xA5 = list_t1[0], list_t1[1], list_t1[2], list_t1[3], list_t1[4]
        xB1, xB2, xB3, xB4, xB5 = list_t2[0], list_t2[1], list_t2[2], list_t2[3], list_t2[4]
        # print(f"xA1.shape:", xA1.shape) # torch.Size([32, 64, 256, 256])
        # print(f"xA2.shape:", xA2.shape) # torch.Size([32, 128, 128, 128])
        # print(f"xA3.shape:", xA3.shape) # torch.Size([32, 256, 64, 64])
        # print(f"xA4.shape:", xA4.shape) # torch.Size([32, 512, 32, 32])
        # print(f"xA5.shape:", xA5.shape) # torch.Size([32, 512, 16, 16])
        b1, c1, h1, w1 = xA1.shape
        b4, c4, h4, w4 = xA4.shape
        b5, c5, h5, w5 = xA5.shape
        
        xx = torch.cat((xA5, xB5), dim = 1)
        xx = self.dec5(xx)
        xA5 = self.layernorm1(xA5)
        xB5 = self.layernorm1(xB5)
        x5s = torch.abs(xA5 - xB5)
        x5s = x5s.reshape(b5, c5, h5 * w5).transpose(2, 1)
        xA5s = xA5.reshape(b5, c5, h5 * w5).transpose(2, 1)
        xB5s = xB5.reshape(b5, c5, h5 * w5).transpose(2, 1)
        x_hpA5 = self.hopf5(x5s, xB5s, xB5s)
        x_hpA5 = x_hpA5.transpose(1, 2).reshape(b5, c5, h5, w5)
        x_hpB5 = self.hopf5(x5s, xA5s, xA5s)
        x_hpB5 = x_hpB5.transpose(1, 2).reshape(b5, c5, h5, w5)
        x5 = self.sig(x_hpA5 + x_hpB5)#512 16 16
        xx5 = self.out5(x5)
        x = torch.mul(x5, xx)#512 16 16
        x = self.dec51(x)
        x = self.UpSample(x)#512 32 32

        xx = torch.cat((xA4, xB4), dim = 1)
        xx = self.dec4(xx)
        xA4 = self.layernorm2(xA4)
        xB4 = self.layernorm2(xB4)
        x4s = torch.abs(xA4 - xB4)
        x4s = x4s.reshape(b4, c4, h4 * w4).transpose(2, 1)
        xA4s = xA4.reshape(b4, c4, h4 * w4).transpose(2, 1)
        xB4s = xB4.reshape(b4, c4, h4 * w4).transpose(2, 1)
        x_hpA4 = self.hopf4(x4s, xB4s, xB4s)
        x_hpA4 = x_hpA4.transpose(1, 2).reshape(b4, c4, h4, w4)
        x_hpB4 = self.hopf4(x4s, xA4s, xA4s)
        x_hpB4 = x_hpB4.transpose(1, 2).reshape(b4, c4, h4, w4)
        x4 = self.sig(x_hpA4 + x_hpB4)
        xx4 = self.out4(x4)
        xx = torch.mul(x4, xx) # 512 32 32
        x = torch.cat((xx, x), dim = 1) # 1024 32 32
        x = self.dec41(x)# 256 32 32
        x =self.UpSample(x) #256 64 64

        xx = torch.cat((xA3, xB3), dim = 1)
        xx = self.dec3(xx) # 256 64 64
        x = torch.cat((xx, x), dim = 1)# 512 64 64
        x = self.dec31(x)
        x =self.UpSample(x)# 128 128 128

        xx = torch.cat((xA2, xB2), dim = 1)
        xx = self.dec2(xx)# 128 128 128
        x = torch.cat((xx, x), dim = 1)#256 128 128
        x = self.dec21(x)#64 128 128
        x =self.UpSample(x)

        xx = torch.cat((xA1, xB1), dim = 1).reshape(b1, c1*2, h1, w1)
        xx = self.dec1(xx)#64 256 256
        x = torch.cat((xx, x), dim = 1)#128 256 256
        x = self.dec11(x)#32 256 256
        
        logits = self.outc(x)
        # print(f"logits:{logits.shape}")
        return logits, xx4, xx5
