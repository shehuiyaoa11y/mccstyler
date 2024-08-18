import torch.nn as nn
import torch

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),

    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),

)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def normal(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized

class Content_SA(nn.Module):
    def __init__(self, in_dim):
        super(Content_SA, self).__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content_feat):
        B, C, H, W = content_feat.size()
        F_Fc_norm = self.f(normal(content_feat)).view(B, -1, H * W).permute(0, 2, 1)

        B, C, H, W = content_feat.size()
        G_Fs_norm = self.g(normal(content_feat)).view(B, -1, H * W)

        energy = torch.bmm(F_Fc_norm, G_Fs_norm)
        attention = self.softmax(energy)

        H_Fs = self.h(content_feat).view(B, -1, H * W)
        out = torch.bmm(H_Fs, attention.permute(0, 2, 1))
        B, C, H, W = content_feat.size()
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        out += content_feat

        return out


class mcc(nn.Module):
    def __init__(self, in_dim):
        super(mcc, self).__init__()
        self.f = nn.Conv2d(in_dim, int(in_dim), (1, 1))
        self.g = nn.Conv2d(in_dim, int(in_dim), (1, 1))
        self.h = nn.Conv2d(in_dim, int(in_dim), (1, 1))
        # self.softmax  = nn.Softmax(dim=-1)    #16
        self.softmax = nn.Softmax(dim=-2)  # 17
        self.out_conv = nn.Conv2d(int(in_dim), in_dim, (1, 1))
        self.fc = nn.Linear(in_dim, in_dim)
        # self.wNet = WNet()

    def forward(self, content_feat, style_feat):
        B, C, H, W = content_feat.size()

        F_Fc_norm = self.f(normal(content_feat))



        B, C, H, W = style_feat.size()
        G_Fs_norm = self.g(normal(style_feat)).view(-1, 1, H * W)

        G_Fs_sum = G_Fs_norm.view(B, C, H * W).sum(-1)

        FC_S = torch.bmm(G_Fs_norm, G_Fs_norm.permute(0, 2, 1)).view(B, C) / G_Fs_sum  # 14

        FC_S = self.fc(FC_S).view(B, C, 1, 1)


        out = F_Fc_norm * FC_S
        B, C, H, W = content_feat.size()
        out = out.contiguous().view(B, -1, H, W)
        out = self.out_conv(out)
        out = content_feat + out

        return out


class MCC_Module(nn.Module):
    def __init__(self, in_dim):
        super(MCC_Module, self).__init__()
        self.CSA = Content_SA(in_dim)
        #self.SSA = Style_SA(in_dim)
        self.MCC=mcc(in_dim)

    def forward(self, content_feats, style_feats):

        style_feat_4 = style_feats[-2]
        content_feat_4 = self.CSA(content_feats[-2])
        #style_feat_4 = self.SSA(style_feats[-2])
        Fcsc = self.MCC(content_feat_4, style_feat_4)

        return Fcsc

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


mlp = nn.Sequential(
            nn.Conv2d(3, 4, (1,1)), # 0
            nn.ReLU(), # 1
            nn.Conv2d(64, 16, (1,1)), # 2
            nn.Conv2d(128, 128, (2,2)), # 3
            nn.ReLU(), # 4
            nn.Conv2d(128, 32, (2,2)), # 5
            nn.Conv2d(256, 128, (2,2)), # 6
            nn.ReLU(), # 7
            nn.Conv2d(128, 64, (2,2)), # 8
            nn.Conv2d(512, 128, (2,2)), # 9
            nn.ReLU(), # 10
            nn.Conv2d(128, 128, (2,2)) # 11
        )



class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]



def squeeze(x, size=2):
    bs, d, new_h, new_w = x.shape[0], x.shape[1], x.shape[2]//size, x.shape[3]//size
    x = x.reshape(bs, d, new_h, size, new_w, size).permute(0, 3, 5, 1, 2, 4)
    return x.reshape(bs, d*(size**2), new_h, new_w)

def unsqueeze(x, size=2):
    bs, new_d, h, w = x.shape[0], x.shape[1]//(size**2), x.shape[2], x.shape[3]
    x = x.reshape(bs, size, size, new_d, h, w).permute(0, 3, 4, 1, 5, 2)
    return x.reshape(bs, new_d, h * size, w * size)




def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)




class residual_block(nn.Module):
    def __init__(self, channel, stride=1, mult=4, kernel=3):
        super().__init__()
        self.stride = stride

        pad = (kernel - 1) // 2
        if stride == 1:
            in_ch = channel
        else:
            in_ch = channel // 4

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_ch, channel//mult, kernel_size=kernel, stride=stride, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(channel // mult, channel // mult, kernel_size=kernel, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(channel // mult, channel, kernel_size=kernel, padding=0, bias=True)
        )
        self.init_layers()

    def init_layers(self):
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.conv(x2)
        if self.stride == 2:
            x1 = squeeze(x1)
            x2 = squeeze(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = unsqueeze(x2)
        Fx2 = - self.conv(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = unsqueeze(x1)

        x = (x1, x2)
        return x

class Transition(nn.Module):
    def __init__(self):
        super(Transition, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, (1,1)),  # 0: [1, 3, 256, 256] -> [1, 4, 256, 256]
            nn.ReLU(),  # 1
            nn.Conv2d(16, 64, (1,1)),  # 2: [1, 4, 256, 256] -> [1, 3, 256, 256]
            nn.Conv2d(64, 128, (2,2)),  # 3: [1, 3, 256, 256] -> [1, 128, 255, 255]
            nn.ReLU(),  # 4
            nn.Conv2d(128, 32, (2,2)),  # 5: [1, 128, 255, 255] -> [1, 32, 254, 254]
            nn.Conv2d(32, 256, (2,2)),  # 6: [1, 32, 254, 254] -> [1, 256, 253, 253]
            nn.ReLU(),  # 7
            nn.Conv2d(256, 64, (2,2)),  # 8: [1, 256, 253, 253] -> [1, 64, 252, 252]
            nn.Conv2d(64, 4, (2,2)),  # 9: [1, 64, 252, 252] -> [1, 4, 251, 251]
        )
        # Upsample to [1, 4, 256, 256]
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x

class Transition_Inverse(nn.Module):
    def __init__(self):
        super(Transition_Inverse, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, (1,1)),  # 0: [1, 4, 256, 256] -> [1, 64, 256, 256]
            nn.ReLU(),  # 1
            nn.Conv2d(64, 128, (3,3), padding=1),  # 2: [1, 64, 256, 256] -> [1, 128, 256, 256]
            nn.ReLU(),  # 3
            nn.Conv2d(128, 64, (3,3), padding=1),  # 4: [1, 128, 256, 256] -> [1, 64, 256, 256]
            nn.ReLU(),  # 5
            nn.Conv2d(64, 3, (1,1))  # 6: [1, 64, 256, 256] -> [1, 3, 256, 256]
        )
        # Optionally, add upsampling if you need to adjust spatial dimensions
        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.features(x)
        # Optionally, apply upsampling if spatial dimensions need adjustment
        # x = self.upsample(x)
        return x


class channel_reduction(nn.Module):
    def __init__(self,in_ch, out_ch, sp_steps=2, n_blocks=2, kernel=3):
        super().__init__()
        self.pad = out_ch * 4 ** sp_steps - in_ch
        self.inj_pad = injective_pad(self.pad)
        self.sp_steps = sp_steps
        self.n_blocks = n_blocks
        self.conv1 = Transition()
        self.conv2 = Transition_Inverse()

        self.block_list = nn.ModuleList()
        for i in range(n_blocks):
            self.block_list.append(residual_block(out_ch * 4 ** sp_steps, stride=1, mult=4, kernel=kernel))

    def forward(self, x):

        x = self.conv1(x)

        x = list(split(x))


        x[0] = self.inj_pad.forward(x[0])
        x[1] = self.inj_pad.forward(x[1])


        for block in self.block_list:
            x = block.forward(x)
        x = merge(x[0], x[1])

        # spread
        for _ in range(self.sp_steps):
            bs, new_d, h, w = x.shape[0], x.shape[1]//2**2, x.shape[2], x.shape[3]
            x = x.reshape(bs, 2, 2, new_d, h, w).permute(0, 3, 4, 1, 5, 2)
            x = x.reshape(bs, new_d, h * 2, w * 2)


        return x

    def inverse(self, x):
        for _ in range(self.sp_steps):
            bs, d, new_h, new_w = x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]//2
            x = x.reshape(bs, d, new_h, 2, new_w, 2).permute(0, 3, 5, 1, 2, 4)
            x = x.reshape(bs, d * 2**2, new_h, new_w)

        x = split(x)

        for block in self.block_list[::-1]:
            x = block.inverse(x)

        x = list(x)
        x[0] = self.inj_pad.inverse(x[0])
        x[1] = self.inj_pad.inverse(x[1])

        x = merge(x[0], x[1])
        x = self.conv2(x)
        return x





class Mcc_Encoder(nn.Module):
    def __init__(self, encoder,gpu_ids=[]):
        super(Mcc_Encoder, self).__init__()
        enc_layers = list(encoder.children())
        # self.enc = nn.Sequential(*enc_layers[:31])  # input -> relu1_1 64
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1 64
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1 128
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1 256
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1 512
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        self.mse_loss = nn.MSELoss()
        self.mcc_module = MCC_Module(512)
        self.decoder = Decoder()
        self.mlp = mlp
        #self.CCPL = CCPL(self.mlp)
        self.channel_reduction = channel_reduction(2, 16, sp_steps=2, kernel=3)


        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False


    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results

    def encode_with_intermediate1(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


    def calc_mean_std(self, feat, eps=1e-5):

        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


    def adain(self, content_feat, style_feat):

        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)


        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)#

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      return self.mse_loss(input, target)

    def forward(self, content, style, encoded_only = False):

        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        Ics_feats = self.encode_with_intermediate(style)

        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + self.calc_content_loss(
            normal(Ics_feats[-2]), normal(content_feats[-2]))


        # Style loss
        style_feats = self.encode_with_intermediate1(style)
        content_feats = self.encode_with_intermediate1(content)
        Ics = self.decoder(self.mcc_module(style_feats,content_feats))
        
        x = self.channel_reduction.forward(Ics)
        Ics = self.channel_reduction.inverse(x)



        Ics_feats = self.encode_with_intermediate1(Ics)
        loss_c1 = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + self.calc_content_loss(
            normal(Ics_feats[-2]), normal(content_feats[-2]))

        Icc = self.decoder(self.mcc_module(content_feats, content_feats))


        loss_lambda1 = self.calc_content_loss(Icc, content)
        if encoded_only:
            return content_feats, style_feats, loss_c,loss_lambda1,loss_c1

        else:
            adain_feat = self.adain(content_feats[-1], style_feats[-1])

            return adain_feat, loss_c


class Decoder(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(Decoder, self).__init__()
        decoder = [
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(), # 256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),# 128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),# 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
            ]
        self.decoder = nn.Sequential(*decoder)
        # if len(gpu_ids) > 0:
        #     assert(torch.cuda.is_available())
        #     self.decoder.to(gpu_ids[0])

    def forward(self, adain_feat):
        fake_image = self.decoder(adain_feat)

        return fake_image

