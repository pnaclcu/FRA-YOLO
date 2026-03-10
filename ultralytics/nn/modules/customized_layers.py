import torch
import torch.nn as nn

__all__ = (
"TripletSelection",
)
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x) #b,h,c,w
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletSelection(nn.Module):
    def __init__(self, channel_in,no_spatial=False):
        super(TripletSelection, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.channel_in = channel_in
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 3, kernel_size=7, padding=(7 - 1) // 2, bias=False)



    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out31 = self.hw(x)
            x_out = (x_out11 + x_out21 + x_out31)/3
        else:
            x_out = (x_out11 + x_out21)/2
        #########Fusion Type#########
        out_flatten = self.avg_pool(x_out)
        # you can use fully connection layer instead of conv to generate corresponding linear features
        # but there is nearly no difference in mAP
        out_flatten = self.conv(out_flatten.squeeze(-1).transpose(-1, -2))
        out_flatten = torch.softmax(out_flatten, dim=1)
        out_flatten = out_flatten.unsqueeze(-1).unsqueeze(-1)
        branches = [x_out11, x_out21, x_out31]
        branches = torch.stack(branches, dim=1)
        out_flatten = out_flatten.expand_as(branches)
        x_out = (out_flatten * branches).sum(dim=1)
        #########Fusion Type#########

        #########Seperation Type #########
        '''
        The seperation type performances worse compared with the upper fusion type
        # x_out11_linear = self.avg_pool(x_out11).squeeze(2,3).unsqueeze(-1)
        # x_out21_linear = self.avg_pool(x_out21).squeeze(2,3).unsqueeze(-1)
        # x_out31_linear = self.avg_pool(x_out31).squeeze(2,3).unsqueeze(-1)
        # x_out_linear = torch.cat([x_out11_linear, x_out21_linear, x_out31_linear], dim=-1)
        # x_linear_refined = torch.softmax(x_out_linear, dim=-1).permute(0, 2, 1)
        # branches = [x_out11, x_out21, x_out31]
        # branches = torch.stack(branches, dim=1)
        # x_out = (branches * x_linear_refined.view(-1, 3, self.channel_in, 1, 1)).sum(dim=1)
        '''
        #########Seperation Type #########

        return x_out

if __name__ == '__main__':
    x = torch.randn(8, 16, 64, 64)
    model = TripletSelection(16)
    y = model(x)
    print(y.shape)