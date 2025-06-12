import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50', output_stride=16, in_channels=1):
        super(DeepLabV3Plus, self).__init__()
        self.in_channels = in_channels

        # 加载 ResNet50 骨干网络
        self.backbone = models.resnet50(pretrained=True)
        if self.in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 明确各层输出通道数（修正 layer1 为 256 通道）
        self.layer1 = self.backbone.layer1  # 输出通道数：256
        self.layer4 = self.backbone.layer4  # 输出通道数：2048

        # 构建 ASPP 模块（输入通道为 layer4 的 2048）
        self.aspp = self._make_aspp(2048, output_stride)

        # 构建解码器（低级特征通道数修正为 256）
        self.decoder = self._make_decoder(num_classes)

    def _make_aspp(self, in_channels, output_stride):
        """ASPP 模块（保持不变）"""
        dilations = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            *[nn.Sequential(
                nn.Conv2d(in_channels, 256, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for dilation in dilations],
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        ])

    def _make_decoder(self, num_classes):
        """修正解码器低级特征通道数为 256"""
        return nn.Sequential(
            # 低级特征处理：256 通道 → 48 通道（关键修正）
            nn.Sequential(
                nn.Conv2d(256, 48, 1, bias=False),  # 原为 64→48，现改为 256→48
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ),
            # 特征融合与预测
            nn.Sequential(
                nn.Conv2d(256 * 5 + 48, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1)
            )
        )

    def forward(self, x):
        # 提取低级特征（layer1 输出通道数为 256，尺寸为输入的 1/4）
        low_level = self.backbone.conv1(x)
        low_level = self.backbone.bn1(low_level)
        low_level = self.backbone.relu(low_level)
        low_level = self.backbone.maxpool(low_level)
        low_level = self.layer1(low_level)  # [B, 256, H/4, W/4]

        # 提取高级特征（layer4 输出通道数为 2048，尺寸为输入的 1/32）
        x = self.backbone.layer2(low_level)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # [B, 2048, H/32, W/32]

        # ASPP 处理
        aspp_feats = []
        for module in self.aspp:
            feat = module(x)
            feat = F.interpolate(feat, size=low_level.shape[2:], mode='bilinear', align_corners=False)
            aspp_feats.append(feat)
        x = torch.cat(aspp_feats, dim=1)  # [B, 1280, H/4, W/4]

        # 解码器处理低级特征
        low_level = self.decoder[0](low_level)  # [B, 48, H/4, W/4]
        x = torch.cat([x, low_level], dim=1)  # [B, 1328, H/4, W/4]
        x = self.decoder[1](x)  # [B, 1, H/4, W/4]

        # 最终上采样到输入尺寸（关键修正）
        target_size = tuple(s * 4 for s in x.shape[2:])  # 计算目标尺寸 (H/4*4, W/4*4) = (H, W)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x

    def print_model_structure(self):
        """打印模型结构和参数信息（保持不变）"""
        print("模型结构：")
        print(self)

        print("\n参数列表（名称-尺寸）：")
        total_params = 0
        for name, param in self.named_parameters():
            param_size = param.numel()
            total_params += param_size
            print(f"{name}: {tuple(param.shape)} - 参数数量: {param_size}")

        print(f"\n总参数数量: {total_params}")


# 示例用法（保持不变）
if __name__ == "__main__":
    model = DeepLabV3Plus(in_channels=1, num_classes=1)
    model.print_model_structure()

    input_tensor = torch.randn(2, 1, 512, 512)
    output_tensor = model(input_tensor)
    print(f"\n输出形状: {output_tensor.shape}")  # 应输出 [2, 1, 512, 512]