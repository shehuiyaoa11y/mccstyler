import numpy as np
import torch
import lpips
from PIL import Image

# 假设您已经有了要计算LPIPS距离的两张图片 image1 和 image2
# 加载图像文件
image1 = Image.open(r"D:\Appdata\learn\code\Diffstyler-main\Diffstyler-main\graph.4\forest10_256.jpg")
image2 = Image.open(r"D:\Appdata\learn\code\Diffstyler-main\Diffstyler-main\test4\forest10A monet styler painting of tree.jpg")

# 加载预训练的LPIPS模型
lpips_model = lpips.LPIPS(net="alex")

# 将图像转换为PyTorch的Tensor格式
image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# 使用LPIPS模型计算距离
distance = lpips_model(image1_tensor, image2_tensor)

print("LPIPS distance:", distance.item())
