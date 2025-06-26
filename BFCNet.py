import torch.nn as nn
import torch
from resnet import Backbone_ResNet152_in3, Backbone_ResNet50_in3,Backbone_ResNet34_in3
import torch.nn.functional as F



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class prediction_decoder(nn.Module):
    def __init__(self, channel1=64, channel2=128, channel3=256, channel4=256, channel5=512, n_classes=2):
        super(prediction_decoder, self).__init__()
        # 16-32
        self.decoder5 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel5, channel5, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel5, channel4, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 32-64
        self.decoder4 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel4, channel4, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel4, channel3, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 64-128
        self.decoder3 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel3, channel3, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel3, channel2, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 128-256
        self.decoder2 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel2, channel2, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel2, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 256-512
        self.decoder1 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 480 640
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
                )

    def forward(self, x5, x4, x3, x2, x1):
        x5_decoder = self.decoder5(x5)
        x4_decoder = self.decoder4(x5_decoder + x4)
        x3_decoder = self.decoder3(x4_decoder + x3)
        x2_decoder = self.decoder2(x3_decoder + x2)
        semantic_pred = self.decoder1(x2_decoder + x1)
        # print('x5_decoder', x5_decoder.shape)
        # print('x4_decoder', x4_decoder.shape)
        # print('x3_decoder', x3_decoder.shape)
        # print('x2_decoder', x2_decoder.shape)
        # print('semantic_pred', semantic_pred.shape)

        return semantic_pred


class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()

    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度方向的全局平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度方向的全局平均池化

        mip = max(8, inp // reduction)  # 计算中间通道数

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)  # 降维
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()  # 激活函数
        # self.sigmoid = nn.Sigmoid()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 高度方向的注意力权重
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 宽度方向的注意力权重

    def forward(self, x):
        identity = x  # 保存输入特征图
        n, c, h, w = x.size()

        # 高度方向的全局平均池化
        x_h = self.pool_h(x)
        # 宽度方向的全局平均池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 调整维度顺序

        # 拼接高度和宽度方向的特征
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分别计算高度和宽度方向的注意力权重
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复宽度方向的维度顺序

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 应用注意力权重
        out = identity * a_w * a_h

        return out

class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, scale=16):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h * w)
        # [b, c, H*W] * [b, H*W, 1]
        concate_QK = torch.matmul(query, key)
        # print(concate_QK.shape)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        # print(concate_QK.shape)
        value = self.Conv_value(concate_QK)
        out = x + value
        return out



##浅层特征 带预测
class BEM(nn.Module):
    def __init__(self, in_channels,out_channels,rates=[1,3,5,7]):
        super(BEM, self).__init__()
        self.rates = rates

        self.aspp1=BasicConv2d(int(in_channels*2), in_channels//2, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
        self.aspp2=BasicConv2d(int(in_channels*2), in_channels//2, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
        self.aspp3=BasicConv2d(int(in_channels*2), in_channels//2, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])
        self.aspp4=BasicConv2d(int(in_channels*2), in_channels//2, kernel_size=3, stride=1, padding=rates[3], dilation=rates[3])

        # 多尺度空洞卷积
        # self.aspp1 = nn.Sequential(
        #     nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0]),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.aspp2 = nn.Sequential(
        #     nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1]),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.aspp3 = nn.Sequential(
        #     nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2]),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.aspp4 = nn.Sequential(
        #     nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=rates[3], dilation=rates[3]),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

        # 特征融合


        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Conv2d( in_channels*2, out_channels, kernel_size=3, stride=1, padding=1)

        self.pred = nn.Conv2d(out_channels, 2, kernel_size=3, padding=1, bias=True)

    def forward(self, x_rgb,x_ir):

        x_a=x_rgb+x_ir
        x_ria_c=x_rgb *x_a
        x_ria_a = x_ir *x_a
        x_ri1=torch.cat([x_ria_c, x_ria_a], dim=1)
        # print('x_ri1:',x_ri1.shape)
        x_aspp1=self.aspp1(x_ri1)
        x_aspp2=self.aspp2(x_ri1)
        x_aspp3=self.aspp3(x_ri1)
        x_aspp4=self.aspp4(x_ri1)

        out_x1=torch.cat([x_aspp1, x_aspp2, x_aspp3, x_aspp4], dim=1)
        # print('out_x1:',out_x.shape)
        out_x2=self.conv_fuse(out_x1)+x_ri1
        # print('out_x2:',out_x.shape)
        out_x=self.conv2(out_x2)
        # print('out_x3:', out_x.shape)
        edge_pred = self.pred(out_x)

        # return out_x
        return out_x, edge_pred

##浅层特征 不带预测
class BEM2(nn.Module):
    def __init__(self, in_channels,out_channels,rates=[1,3,5,7]):
        super(BEM2, self).__init__()
        self.rates = rates

        self.aspp1=BasicConv2d(int(in_channels*2), in_channels//2, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
        self.aspp2=BasicConv2d(int(in_channels*2), in_channels//2, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
        self.aspp3=BasicConv2d(int(in_channels*2), in_channels//2, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])
        self.aspp4=BasicConv2d(int(in_channels*2), in_channels//2, kernel_size=3, stride=1, padding=rates[3], dilation=rates[3])

        # 多尺度空洞卷积
        # self.aspp1 = nn.Sequential(
        #     nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0]),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.aspp2 = nn.Sequential(
        #     nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1]),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.aspp3 = nn.Sequential(
        #     nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2]),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.aspp4 = nn.Sequential(
        #     nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=rates[3], dilation=rates[3]),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

        # 特征融合


        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Conv2d( in_channels*2, out_channels, kernel_size=3, stride=1, padding=1)

        # self.pred = nn.Conv2d(out_channels, 2, kernel_size=3, padding=1, bias=True)

    def forward(self, x_rgb,x_ir):

        x_a=x_rgb+x_ir
        x_ria_c=x_rgb *x_a
        x_ria_a = x_ir *x_a
        x_ri1=torch.cat([x_ria_c, x_ria_a], dim=1)
        # print('x_ri1:',x_ri1.shape)
        x_aspp1=self.aspp1(x_ri1)
        x_aspp2=self.aspp2(x_ri1)
        x_aspp3=self.aspp3(x_ri1)
        x_aspp4=self.aspp4(x_ri1)

        out_x=torch.cat([x_aspp1, x_aspp2, x_aspp3, x_aspp4], dim=1)
        # print('out_x1:',out_x.shape)
        out_x=self.conv_fuse(out_x)+x_ri1
        # print('out_x2:',out_x.shape)
        out_x=self.conv2(out_x)
        # print('out_x3:', out_x.shape)
        # edge_pred = self.pred(out_x)

        return out_x
        # return out_x, edge_pred


##中层特征
class FAM(nn.Module):
    def __init__(self, in_channels,out_channels, reduction=32):
        super(FAM, self).__init__()
        self.gc_block = GlobalContextBlock(out_channels)
        self.coord_att = CoordAtt(out_channels, out_channels, reduction=reduction)
        self.conv = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_rgb,x_ir):
        x_rgb=self.conv(x_rgb)
        x_ir = self.conv(x_ir)
        x_c=x_rgb*x_ir
        x_a=x_rgb+x_ir
        coord_out = self.coord_att(x_a)
        gc_out = self.gc_block(x_c+coord_out)
        # combined_out = gc_out + coord_out*x
        return gc_out

##深层特征
class CLM(nn.Module):
    def __init__(self, all_channels=64):
        super(CLM, self).__init__()
        self.linear_e = nn.Linear(all_channels, all_channels, bias=False)
        self.channel = all_channels

        self.conv = BasicConv2d(all_channels, all_channels, kernel_size=3, stride=1, padding=1)
        # self.pred = nn.Conv2d(all_channels, 2, kernel_size=3, padding=1, bias=True)


    def forward(self, exemplar,query):
        #input rgb tir
        h,w=exemplar.shape[-2:]
        exemplar_flat=exemplar.view(-1,self.channel,h*w) # n c h*w
        query_flat=query.view(-1,self.channel,h*w) # n c h*w
        exemplar_t=torch.transpose(exemplar_flat,1,2).contiguous() # n h*w c
        exemplar_l=self.linear_e(exemplar_t) # n h*w c
        A=torch.bmm(exemplar_l,query_flat)# n h*w h*w
        B=F.softmax(torch.transpose(A,1,2),dim=1)
        query_c=torch.bmm(query_flat,B).contiguous()
        C=F.softmax(A,dim=1)#


        exemplar_c=torch.bmm(exemplar_flat,C)
        out_f=(exemplar_c+query_c).view(-1,self.channel,h,w)

        out_x=out_f + exemplar + query
        out_x = self.conv(out_x)
        # pred = self.pred(out_x)
        return out_x

#0
#三个融合模块，边界损失,主干ResNet50
class BFCNet(nn.Module):
    def __init__(self, n_classes):
        super(BFCNet, self).__init__()
        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet50_in3(pretrained=True)

        # reduce the channel number, input: 512 512
        self.coshare_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 256
        self.coshare_conv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 128
        self.coshare_conv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 64
        self.coshare_conv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 32
        self.coshare_conv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 16


        self.BEM1 = BEM(64,64)
        self.BEM2 = BEM(128,128)
        self.FAM3 = FAM(256,256)
        self.FAM4 = FAM(256,256)
        self.CLM5 = CLM(512)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)

    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)
        # x1： torch.Size([1, 64, 256, 256])
        # x2: torch.Size([1, 256, 128, 128])
        # x3: torch.Size([1, 512, 64, 64])
        # x4: torch.Size([1, 1024, 32, 32])
        # x5: torch.Size([1, 2048, 16, 16])
        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.coshare_conv1(x1)
        x2 = self.coshare_conv2(x2)
        x3 = self.coshare_conv3(x3)
        x4 = self.coshare_conv4(x4)
        x5 = self.coshare_conv5(x5)

        ir1 = self.coshare_conv1(ir1)
        ir2 = self.coshare_conv2(ir2)
        ir3 = self.coshare_conv3(ir3)
        ir4 = self.coshare_conv4(ir4)
        ir5 = self.coshare_conv5(ir5)

        # print('ir1', ir1.shape)
        # print('ir2', ir2.shape)
        # print('ir3', ir3.shape)
        # print('ir4', ir4.shape)
        # print('ir5', ir5.shape)


        out5  = self.CLM5(x5,ir5)
        out4 = self.FAM4(x4, ir4)
        out3 = self.FAM3(x3, ir3)
        # out3 = x3+ ir3
        out2, edge2 = self.BEM2(x2, ir2)
        out1, edge1 = self.BEM1(x1, ir1)

        # print('out5',out5.shape)
        # print('out4',out4.shape)
        # print('out3',out3.shape)
        # print('out1',out1.shape)
        # print('out2',out2.shape)
        # print(edge1.shape)
        # print(edge2.shape)


        semantic = self.decoder(out5, out4, out3, out2, out1)
        edge1 = torch.nn.functional.interpolate(edge1, scale_factor=2, mode='bilinear')
        edge2 = torch.nn.functional.interpolate(edge2, scale_factor=4, mode='bilinear')

        return semantic, edge1, edge2

#1
#删除CLM模块
class BFCNet1(nn.Module):
    def __init__(self, n_classes):
        super(BFCNet1, self).__init__()

        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet50_in3(pretrained=True)

        # reduce the channel number, input: 512 512
        self.coshare_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 256
        self.coshare_conv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 128
        self.coshare_conv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 64
        self.coshare_conv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 32
        self.coshare_conv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 16


        self.BEM1 = BEM(64,64)
        self.BEM2 = BEM(128,128)
        self.FAM3 = FAM(256,256)
        self.FAM4 = FAM(256,256)
        self.CLM5 = CLM(512)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)

    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)
        # x1： torch.Size([1, 64, 256, 256])
        # x2: torch.Size([1, 256, 128, 128])
        # x3: torch.Size([1, 512, 64, 64])
        # x4: torch.Size([1, 1024, 32, 32])
        # x5: torch.Size([1, 2048, 16, 16])
        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.coshare_conv1(x1)
        x2 = self.coshare_conv2(x2)
        x3 = self.coshare_conv3(x3)
        x4 = self.coshare_conv4(x4)
        x5 = self.coshare_conv5(x5)

        ir1 = self.coshare_conv1(ir1)
        ir2 = self.coshare_conv2(ir2)
        ir3 = self.coshare_conv3(ir3)
        ir4 = self.coshare_conv4(ir4)
        ir5 = self.coshare_conv5(ir5)

        # print('ir1', ir1.shape)
        # print('ir2', ir2.shape)
        # print('ir3', ir3.shape)
        # print('ir4', ir4.shape)
        # print('ir5', ir5.shape)


        # out5  = self.CLM5(x5,ir5)
        out5=x5+ir5
        out4 = self.FAM4(x4, ir4)
        out3 = self.FAM3(x3, ir3)
        # out3 = x3+ ir3
        out2,edge2 = self.BEM2(x2, ir2)
        out1, edge1 = self.BEM1(x1, ir1)

        # print('out5',out5.shape)
        # print('out4',out4.shape)
        # print('out3',out3.shape)
        # print('out1',out1.shape)
        # print('out2',out2.shape)
        # print(edge1.shape)
        # print(edge2.shape)


        semantic = self.decoder(out5, out4, out3, out2, out1)
        edge1 = torch.nn.functional.interpolate(edge1, scale_factor=2, mode='bilinear')
        edge2 = torch.nn.functional.interpolate(edge2, scale_factor=4, mode='bilinear')

        return semantic, edge1, edge2

#2
#删除CLM FAM模块
class BFCNet2(nn.Module):
    def __init__(self, n_classes):
        super(BFCNet2, self).__init__()

        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet50_in3(pretrained=True)

        # reduce the channel number, input: 512 512
        self.coshare_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 256
        self.coshare_conv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 128
        self.coshare_conv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 64
        self.coshare_conv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 32
        self.coshare_conv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 16


        self.BEM1 = BEM(64,64)
        self.BEM2 = BEM(128,128)
        self.FAM3 = FAM(256,256)
        self.FAM4 = FAM(256,256)
        self.CLM5 = CLM(512)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)

    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)
        # x1： torch.Size([1, 64, 256, 256])
        # x2: torch.Size([1, 256, 128, 128])
        # x3: torch.Size([1, 512, 64, 64])
        # x4: torch.Size([1, 1024, 32, 32])
        # x5: torch.Size([1, 2048, 16, 16])
        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.coshare_conv1(x1)
        x2 = self.coshare_conv2(x2)
        x3 = self.coshare_conv3(x3)
        x4 = self.coshare_conv4(x4)
        x5 = self.coshare_conv5(x5)

        ir1 = self.coshare_conv1(ir1)
        ir2 = self.coshare_conv2(ir2)
        ir3 = self.coshare_conv3(ir3)
        ir4 = self.coshare_conv4(ir4)
        ir5 = self.coshare_conv5(ir5)

        # out5  = self.CLM5(x5,ir5)
        out5=x5+ir5
        # out4 = self.FAM4(x4, ir4)
        # out3 = self.FAM3(x3, ir3)
        out4 = x4+ ir4
        out3 = x3+ ir3
        out2,edge2 = self.BEM2(x2, ir2)
        out1, edge1 = self.BEM1(x1, ir1)

        semantic = self.decoder(out5, out4, out3, out2, out1)
        edge1 = torch.nn.functional.interpolate(edge1, scale_factor=2, mode='bilinear')
        edge2 = torch.nn.functional.interpolate(edge2, scale_factor=4, mode='bilinear')

        return semantic, edge1, edge2

#2
#删除CLM FAM BEM模块
class BFCNet3(nn.Module):
    def __init__(self, n_classes):
        super(BFCNet3, self).__init__()

        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet50_in3(pretrained=True)
        self.n_classes = n_classes
        # reduce the channel number, input: 512 512
        self.coshare_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 256
        self.coshare_conv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 128
        self.coshare_conv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 64
        self.coshare_conv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 32
        self.coshare_conv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 16


        self.BEM1 = BEM(64,64)
        self.BEM2 = BEM(128,128)
        self.FAM3 = FAM(256,256)
        self.FAM4 = FAM(256,256)
        self.CLM5 = CLM(512)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)
        self.prededge1 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.prededge2 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1)
    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)
        # x1： torch.Size([1, 64, 256, 256])
        # x2: torch.Size([1, 256, 128, 128])
        # x3: torch.Size([1, 512, 64, 64])
        # x4: torch.Size([1, 1024, 32, 32])
        # x5: torch.Size([1, 2048, 16, 16])
        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.coshare_conv1(x1)
        x2 = self.coshare_conv2(x2)
        x3 = self.coshare_conv3(x3)
        x4 = self.coshare_conv4(x4)
        x5 = self.coshare_conv5(x5)

        ir1 = self.coshare_conv1(ir1)
        ir2 = self.coshare_conv2(ir2)
        ir3 = self.coshare_conv3(ir3)
        ir4 = self.coshare_conv4(ir4)
        ir5 = self.coshare_conv5(ir5)



        # out5  = self.CLM5(x5,ir5)
        out5=x5+ir5
        # out4 = self.FAM4(x4, ir4)
        # out3 = self.FAM3(x3, ir3)
        out4 = x4+ ir4
        out3 = x3+ ir3
        out2 = x2+ ir2

        out1 = x1+ ir1

        # out2,edge2 = self.BEM2(x2, ir2)
        # out1, edge1 = self.BEM1(x1, ir1)

        semantic = self.decoder(out5, out4, out3, out2, out1)
        edge1 = self.prededge1(torch.nn.functional.interpolate(out1, scale_factor=2, mode='bilinear'))
        edge2 = self.prededge2(torch.nn.functional.interpolate(out2, scale_factor=4, mode='bilinear'))
        #随机生成，不计算损失
        # edge1=torch.randn(4, self.n_classes, 512, 512)
        # edge2=torch.randn(4, self.n_classes, 512, 512)
        return semantic,edge1,edge2

class BFCNet4(nn.Module):
    def __init__(self, n_classes):
        super(BFCNet4, self).__init__()

        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet50_in3(pretrained=True)
        self.n_classes = n_classes
        # reduce the channel number, input: 512 512
        self.coshare_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 256
        self.coshare_conv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 128
        self.coshare_conv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 64
        self.coshare_conv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 32
        self.coshare_conv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 16


        self.BEM1 = BEM(64,64)
        self.BEM2 = BEM(128,128)
        self.FAM3 = FAM(256,256)
        self.FAM4 = FAM(256,256)
        self.CLM5 = CLM(512)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)
        self.prededge1 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.prededge2 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1)
    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)
        # x1： torch.Size([1, 64, 256, 256])
        # x2: torch.Size([1, 256, 128, 128])
        # x3: torch.Size([1, 512, 64, 64])
        # x4: torch.Size([1, 1024, 32, 32])
        # x5: torch.Size([1, 2048, 16, 16])
        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.coshare_conv1(x1)
        x2 = self.coshare_conv2(x2)
        x3 = self.coshare_conv3(x3)
        x4 = self.coshare_conv4(x4)
        x5 = self.coshare_conv5(x5)

        ir1 = self.coshare_conv1(ir1)
        ir2 = self.coshare_conv2(ir2)
        ir3 = self.coshare_conv3(ir3)
        ir4 = self.coshare_conv4(ir4)
        ir5 = self.coshare_conv5(ir5)



        # out5  = self.CLM5(x5,ir5)
        out5=x5+ir5
        out4 = self.FAM4(x4, ir4)
        out3 = self.FAM3(x3, ir3)
        # out4 = x4+ ir4
        # out3 = x3+ ir3
        out2 = x2+ ir2

        out1 = x1+ ir1

        # out2,edge2 = self.BEM2(x2, ir2)
        # out1, edge1 = self.BEM1(x1, ir1)

        semantic = self.decoder(out5, out4, out3, out2, out1)
        edge1 = self.prededge1(torch.nn.functional.interpolate(out1, scale_factor=2, mode='bilinear'))
        edge2 = self.prededge2(torch.nn.functional.interpolate(out2, scale_factor=4, mode='bilinear'))
        #随机生成，不计算损失
        # edge1=torch.randn(4, self.n_classes, 512, 512)
        # edge2=torch.randn(4, self.n_classes, 512, 512)
        return semantic,edge1,edge2

class BFCNet5(nn.Module):
    def __init__(self, n_classes):
        super(BFCNet5, self).__init__()

        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet50_in3(pretrained=True)
        self.n_classes = n_classes
        # reduce the channel number, input: 512 512
        self.coshare_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 256
        self.coshare_conv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 128
        self.coshare_conv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 64
        self.coshare_conv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 32
        self.coshare_conv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 16


        self.BEM1 = BEM(64,64)
        self.BEM2 = BEM(128,128)
        self.FAM3 = FAM(256,256)

        self.FAM4 = FAM(256,256)
        self.CLM5 = CLM(512)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)
        self.prededge1 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.prededge2 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1)
    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)
        # x1： torch.Size([1, 64, 256, 256])
        # x2: torch.Size([1, 256, 128, 128])
        # x3: torch.Size([1, 512, 64, 64])
        # x4: torch.Size([1, 1024, 32, 32])
        # x5: torch.Size([1, 2048, 16, 16])
        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.coshare_conv1(x1)
        x2 = self.coshare_conv2(x2)
        x3 = self.coshare_conv3(x3)
        x4 = self.coshare_conv4(x4)
        x5 = self.coshare_conv5(x5)

        ir1 = self.coshare_conv1(ir1)
        ir2 = self.coshare_conv2(ir2)
        ir3 = self.coshare_conv3(ir3)
        ir4 = self.coshare_conv4(ir4)
        ir5 = self.coshare_conv5(ir5)



        out5  = self.CLM5(x5,ir5)
        # out5=x5+ir5
        # out4 = self.FAM4(x4, ir4)
        # out3 = self.FAM3(x3, ir3)
        out4 = x4+ ir4
        out3 = x3+ ir3
        out2 = x2+ ir2

        out1 = x1+ ir1

        # out2,edge2 = self.BEM2(x2, ir2)
        # out1, edge1 = self.BEM1(x1, ir1)

        semantic = self.decoder(out5, out4, out3, out2, out1)
        edge1 = self.prededge1(torch.nn.functional.interpolate(out1, scale_factor=2, mode='bilinear'))
        edge2 = self.prededge2(torch.nn.functional.interpolate(out2, scale_factor=4, mode='bilinear'))
        #随机生成，不计算损失
        # edge1=torch.randn(4, self.n_classes, 512, 512)
        # edge2=torch.randn(4, self.n_classes, 512, 512)
        return semantic,edge1,edge2

class BFCNet6(nn.Module):
    def __init__(self, n_classes):
        super(BFCNet6, self).__init__()

        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet50_in3(pretrained=True)
        self.n_classes = n_classes
        # reduce the channel number, input: 512 512
        self.coshare_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 256
        self.coshare_conv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 128
        self.coshare_conv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 64
        self.coshare_conv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 32
        self.coshare_conv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 16


        self.BEM1 = BEM(64,64)
        self.BEM2 = BEM(128,128)
        self.FAM3 = FAM(256,256)

        self.FAM4 = FAM(256,256)
        self.CLM5 = CLM(512)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)
        self.prededge1 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.prededge2 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1)
    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)
        # x1： torch.Size([1, 64, 256, 256])
        # x2: torch.Size([1, 256, 128, 128])
        # x3: torch.Size([1, 512, 64, 64])
        # x4: torch.Size([1, 1024, 32, 32])
        # x5: torch.Size([1, 2048, 16, 16])
        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.coshare_conv1(x1)
        x2 = self.coshare_conv2(x2)
        x3 = self.coshare_conv3(x3)
        x4 = self.coshare_conv4(x4)
        x5 = self.coshare_conv5(x5)

        ir1 = self.coshare_conv1(ir1)
        ir2 = self.coshare_conv2(ir2)
        ir3 = self.coshare_conv3(ir3)
        ir4 = self.coshare_conv4(ir4)
        ir5 = self.coshare_conv5(ir5)



        out5  = self.CLM5(x5,ir5)
        # out5=x5+ir5
        # out4 = self.FAM4(x4, ir4)
        # out3 = self.FAM3(x3, ir3)
        out4 = x4+ ir4
        out3 = x3+ ir3
        # out2 = x2+ ir2
        # out1 = x1+ ir1
        out2, edge2 = self.BEM2(x2, ir2)
        out1, edge1 = self.BEM1(x1, ir1)

        semantic = self.decoder(out5, out4, out3, out2, out1)
        # edge1 = self.prededge1(torch.nn.functional.interpolate(edge1, scale_factor=2, mode='bilinear'))
        # edge2 = self.prededge2(torch.nn.functional.interpolate(edge2, scale_factor=4, mode='bilinear'))
        edge1 = torch.nn.functional.interpolate(edge1, scale_factor=2, mode='bilinear')
        edge2 = torch.nn.functional.interpolate(edge2, scale_factor=4, mode='bilinear')
        #随机生成，不计算损失
        # edge1=torch.randn(4, self.n_classes, 512, 512)
        # edge2=torch.randn(4, self.n_classes, 512, 512)
        return semantic,edge1,edge2

class BFCNet7(nn.Module):
    def __init__(self, n_classes):
        super(BFCNet7, self).__init__()

        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet50_in3(pretrained=True)
        self.n_classes = n_classes
        # reduce the channel number, input: 512 512
        self.coshare_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 256
        self.coshare_conv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 128
        self.coshare_conv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 64
        self.coshare_conv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 32
        self.coshare_conv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 16


        self.BEM1 = BEM(64,64)
        self.BEM2 = BEM(128,128)
        self.FAM3 = FAM(256,256)

        self.FAM4 = FAM(256,256)
        self.CLM5 = CLM(512)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)
        self.prededge1 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.prededge2 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1)
    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)
        # x1： torch.Size([1, 64, 256, 256])
        # x2: torch.Size([1, 256, 128, 128])
        # x3: torch.Size([1, 512, 64, 64])
        # x4: torch.Size([1, 1024, 32, 32])
        # x5: torch.Size([1, 2048, 16, 16])
        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.coshare_conv1(x1)
        x2 = self.coshare_conv2(x2)
        x3 = self.coshare_conv3(x3)
        x4 = self.coshare_conv4(x4)
        x5 = self.coshare_conv5(x5)

        ir1 = self.coshare_conv1(ir1)
        ir2 = self.coshare_conv2(ir2)
        ir3 = self.coshare_conv3(ir3)
        ir4 = self.coshare_conv4(ir4)
        ir5 = self.coshare_conv5(ir5)



        out5  = self.CLM5(x5,ir5)
        # out5=x5+ir5
        out4 = self.FAM4(x4, ir4)
        out3 = self.FAM3(x3, ir3)
        # out4 = x4+ ir4
        # out3 = x3+ ir3
        out2 = x2+ ir2
        out1 = x1+ ir1

        # out2,edge2 = self.BEM2(x2, ir2)
        # out1, edge1 = self.BEM1(x1, ir1)

        semantic = self.decoder(out5, out4, out3, out2, out1)
        edge1 = self.prededge1(torch.nn.functional.interpolate(out1, scale_factor=2, mode='bilinear'))
        edge2 = self.prededge2(torch.nn.functional.interpolate(out2, scale_factor=4, mode='bilinear'))
        #随机生成，不计算损失
        # edge1=torch.randn(4, self.n_classes, 512, 512)
        # edge2=torch.randn(4, self.n_classes, 512, 512)
        return semantic,edge1,edge2

if __name__ == '__main__':

    # for PST900 dataset
    # LASNet(5)
    rgb_image = torch.randn(1, 3, 512, 512)  # (batch_size, channels, height, width)
    ir_image = torch.randn(1, 1, 512, 512)   # (batch_size, channels, height, width)

    model=BFCNet7(2)
    # 前向传播
    output = model(rgb_image, ir_image)
    # print(model)


    # 打印输出形状
    print("Output shape:", output[0].shape)
    print("Output shape:", output[1].shape)
    print("Output shape:", output[2].shape)