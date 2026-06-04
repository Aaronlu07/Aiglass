import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 基础注意力机制 (Channel Attention)
# 满足毕设要求：“加入基础的注意力机制来减少图像噪声”
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# 2. 简化的 DE-Retinex 低光增强模型
# 满足毕设要求：“搭建简化的 DE-Retinex 低光图像增强模型”
class DERetinex(nn.Module):
    def __init__(self):
        super(DERetinex, self).__init__()
        
        # 光照估计网络 (Illumination Estimation)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 引入注意力机制
        self.attention = ChannelAttention(32)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 提取特征
        feat = self.relu(self.conv1(x))
        # 注意力筛选，减少暗光下的噪点特征
        feat = self.attention(feat)
        feat = self.relu(self.conv2(feat))
        
        # 估计光照图 (Illumination Map)
        illu = self.sigmoid(self.conv3(feat))
        
        # Retinex 理论: 反射图 (Enhanced) = 原图 (S) / 光照图 (I)
        # 加 1e-4 避免除以 0
        enhanced = x / (illu + 1e-4)
        
        # 限制在 0~1 之间
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        return enhanced, illu

if __name__ == "__main__":
    # 测试网络结构是否正常
    model = DERetinex()
    dummy_input = torch.rand(1, 3, 256, 256)
    enhanced, illu = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Enhanced shape:", enhanced.shape)
    print("Illumination shape:", illu.shape)
