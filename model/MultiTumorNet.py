import torch
import torch.nn as nn


from torch.nn import Module, Conv2d, Parameter, Softmax
#Cross Contextual Attention

    

class CrossContextualAttention(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(CrossContextualAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
# Augmented Transformation Attention
class AugmentedTransformationAttention(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        out_dim = out_dim or dim
        
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)  # (b, h*w, c)
        
        # Feed-forward path
        ff_output = self.ff(x)
        
        # Attention weights
        attn = self.attention(x)
        
        # Apply attention
        out = attn * ff_output
        
        # Reshape back to original dimensions
        out = out.transpose(1, 2).view(b, c, h, w)
        return out

class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class UPx2(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        self.deconv = nn.ConvTranspose2d(nIn, nOut, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
    def fuseforward(self, input):
        output = self.deconv(input)
        output = self.act(output)
        return output

class CBR2(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
#Convolutional Leaky Resiudal Fusion Network
class CLRF(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv1 = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(nOut)
        self.act1 = nn.LeakyReLU(nOut)

        self.conv2 = nn.Conv2d(nOut, nOut, (kSize, kSize), stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(nOut)
        self.act2 = nn.LeakyReLU(nOut)

        self.downsample = None
        if nIn != nOut:
            self.downsample = nn.Sequential(
                nn.Conv2d(nIn, nOut, 1, stride=stride, bias=False),
                nn.BatchNorm2d(nOut),
            )

    def forward(self, input):
        residual = input

        out = self.conv1(input)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual
        out = self.act2(out)

        return out

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Calculate channel distribution
        self.in_channels = in_channels
        branch_channels = max(in_channels // 3, 1)  # Ensure at least 1 channel
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels, eps=1e-03),
            nn.PReLU(branch_channels)
        )
        
        # 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels, eps=1e-03),
            nn.PReLU(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels, eps=1e-03),
            nn.PReLU(branch_channels)
        )
        
        # 5x5 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels, eps=1e-03),
            nn.PReLU(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(branch_channels, eps=1e-03),
            nn.PReLU(branch_channels)
        )
        
        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels, eps=1e-03),
            nn.PReLU(branch_channels)
        )
        
        # 1x1 convolution to adjust output channels if needed
        total_branch_channels = branch_channels * 4
        self.adjust_channels = None
        if total_branch_channels != out_channels:
            self.adjust_channels = nn.Sequential(
                nn.Conv2d(total_branch_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-03),
                nn.PReLU(out_channels)
            )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        combined = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        if self.adjust_channels is not None:
            combined = self.adjust_channels(combined)
            
        return combined

class InceptionConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1)/2)
        
        # Ensure minimum channel count
        min_channels = max(nIn, 1)
        
        # First layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(nIn, min_channels, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False),
            nn.BatchNorm2d(min_channels, eps=1e-03),
            nn.PReLU(min_channels)
        )
        
        # Inception module (middle layer)
        self.inception = InceptionModule(min_channels, min_channels)
        
        # Final layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(min_channels, nOut, (kSize, kSize), padding=(padding, padding), bias=False),
            nn.BatchNorm2d(nOut, eps=1e-03),
            nn.PReLU(nOut)
        )
        
    def forward(self, input):
        # First layer
        output = self.layer1(input)
        
        # Inception module
        output1 = self.inception(output)
        
        # Final layer
        output = self.layer3(output)+output1
        
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        # print(nIn, nOut, (kSize, kSize))
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        #combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.nOut=nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        # print("bf bn :",input.size(),self.nOut)
        output = self.bn(input)
        # print("after bn :",output.size())
        output = self.act(output)
        # print("after act :",output.size())
        return output

class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = max(int(nOut/5),1)
        n1 = max(nOut - 4*n,1)
        # print(nIn,n,n1,"--")
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
        # print("nOut bf :",nOut)
        self.bn = BR(nOut)
        # print("nOut at :",self.bn.size())
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        # print(d1.size(),add1.size(),add2.size(),add3.size(),add4.size())

        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # print("combine :",combine.size())
        # if residual version
        if self.add:
            # print("add :",combine.size())
            combine = input + combine
        # print(combine.size(),"-----------------")
        output = self.bn(combine)
        return output

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class MLP_G(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.mlp(x)


class LeakyResiudalFusion3d_encoder(nn.Module):

    def __init__(self, p=5, q=3):
        '''        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CLRF(3, 16, 3, 2)  #class defines the convolution layer with batch normalization and PReLU activation
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = CLRF(16 + 3,19,3)
        self.level2_0 = DownSamplerB(16 +3, 64) #downsampler with Conv dilation rate

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64 , 64))
        self.b2 = CLRF(128 + 3,131,3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128 , 128))
        # self.mixstyle = MixStyle2(p=0.5, alpha=0.1)
        self.b3 = CLRF(256,32,3)
        self.sa = CrossContextualAttention(32)
        self.sc = CAM_Module(32)
        self.ff_attention = AugmentedTransformationAttention(32, hidden_dim=128)

        self.conv_sa = CLRF(32,32,3)
        self.conv_sc = CLRF(32,32,3)
        self.conv_ff = CLRF(32,32,3)
        self.mlp = MLP_G(96, hidden_dim=128)
        self.classifier = CLRF(96, 32, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        cat_=torch.cat([output2_0, output2], 1)

        output2_cat = self.b3(cat_)
        out_sa=self.sa(output2_cat)
        out_sa=self.conv_sa(out_sa)
        out_sc=self.sc(output2_cat)
        out_sc=self.conv_sc(out_sc)
        out_ff=self.ff_attention (output2_cat)
        out_ff=self.conv_sc(out_ff)
        out_s=torch.cat([out_sa, out_sc, out_ff], dim=1)  #3d Attention Fusion
        # out_s=out_sa+out_sc
        mlp_out = self.mlp(out_s)
        classifier = self.classifier(mlp_out)

        return classifier

class MultiTumorNet(nn.Module):

    def __init__(self, p=2, q=3, ):

        super().__init__()
        self.encoder = LeakyResiudalFusion3d_encoder(p, q)

        self.up_1_1 = UPx2(32,16)
        self.up_2_1 = UPx2(16,8)

        self.up_1_2 = UPx2(32,16)
        self.up_2_2 = UPx2(16,8)

        self.classifier_1 = UPx2(8,2)
        self.classifier_2 = UPx2(8,2)

    def forward(self, input):

        x=self.encoder(input)
        x1=self.up_1_1(x)
        x1=self.up_2_1(x1)
        classifier1=self.classifier_1(x1)
        
        x2=self.up_1_2(x)
        x2=self.up_2_2(x2)
        classifier2=self.classifier_2(x2)

        return (classifier1,classifier2)

import numpy as np
def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

if __name__ == "__main__":
    # test()
    

    # model = MultiTumorNet()
    model = MultiTumorNet()
    input_ = torch.randn((1, 3, 640, 384))
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    Whole, Core = model(input_)
    # Da_fmap, LL_fmap = SAD_out
    
    print(Whole.shape)
    print(Core.shape)
