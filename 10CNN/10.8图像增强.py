import torch
import torch.nn as nn   
from PIL import Image                          # 图像处理库
import matplotlib.pyplot as plt                 # 用于显示图像
from torchvision import transforms               # 图像预处理库
import numpy as np
import random

def imshow(img_path,transform):
    img = Image.open(img_path)
    fig, ax = plt.subplots(1,2,figsize=(15,4))                    # 创建一个新的图形和坐标轴对象
    ax[0].set_title(f"Original Image {img.size}")                 # 在第一个坐标轴上显示原始图像，并设置标题为图像的尺寸
    ax[0].imshow(img)                                             # 显示原始图像
    img = transform(img)                                          # 对图像进行旋转变换
    ax[1].set_title(f"Transformed Image {img.size}")              # 在第二个坐标轴上显示变换后的图像，并设置标题为图像的尺寸
    ax[1].imshow(img)                                             # 显示变换后的图像
    plt.show()                                                    # 显示图形

# 对图片进行-30度到30度的随机旋转
path = r"E:\code\DeepLearning\10卷积神经网络\PetImages\Cat\6039.jpg"  # 图像文件路径
transform = transforms.RandomRotation(degrees=30)                    # 定义一个随机旋转变换，旋转角度为-30到30度
imshow(path,transform)                                               # 显示原始图像和变换后的图像

# 对图片进行水平翻转
path = r"E:\code\DeepLearning\10卷积神经网络\PetImages\Cat\6039.jpg"  # 图像文件路径
transform = transforms.RandomHorizontalFlip(p=1.0)                   # 定义一个随机水平翻转变换，p=1.0表示始终进行翻转
imshow(path,transform)                                               # 显示原始图像和变换后的图像

# 对图片进行垂直翻转
path = r"E:\code\DeepLearning\10卷积神经网络\PetImages\Cat\6039.jpg"  # 图像文件路径
transform = transforms.RandomVerticalFlip(p=1.0)                     # 定义一个随机垂直翻转变换，p=1.0表示始终进行翻转
imshow(path,transform)                                               # 显示原始图像和变换后的图像

# 对图片进行随机裁剪
path = r"E:\code\DeepLearning\10卷积神经网络\PetImages\Cat\6039.jpg"  # 图像文件路径
transform = transforms.RandomCrop(size=(120,120))                    # 定义一个随机裁剪变换，裁剪后的图像大小为120x120
imshow(path,transform)                                               # 显示原始图像和变换后的图像

# 对图像进行透视变换
path = r"E:\code\DeepLearning\10卷积神经网络\PetImages\Cat\6039.jpg"  # 图像文件路径
transform = transforms.RandomPerspective(               # 定义一个随机透视变换
    distortion_scale=0.5,                               # distortion_scale控制变形程度,0~1之间的值,值越大变形越明显
    p=1.0,                                              # p=1.0表示始终进行变换
    interpolation=transforms.InterpolationMode.BILINEAR # 插值方法，使用双线性插值                                   
)   
imshow(path,transform)                                  # 显示原始图像和变换后的图像

# 对图像进行颜色变换
path = r"E:\code\DeepLearning\10卷积神经网络\PetImages\Cat\6039.jpg"  # 图像文件路径
transform = transforms.ColorJitter(                # 定义一个颜色抖动变换
    brightness=0.5,                                # 亮度因子的范围为[0.5, 1.5],值越大亮度变化越明显
    contrast=0.5,                                  # 对比度因子的范围为[0.5, 1.5],值越大对比度变化越明显
    saturation=0.5,                                # 饱和度因子的范围为[0.5, 1.5],值越大饱和度变化越明显
    hue=0.1                                        # 色调因子的范围为[-0.1, 0.1],值越大色调变化越明显
)
imshow(path,transform)                             # 显示原始图像和变换后的图像

#对图片进行高斯模糊
path = r"E:\code\DeepLearning\10卷积神经网络\PetImages\Cat\6039.jpg"  # 图像文件路径
transform = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)) # 定义一个高斯模糊变换，kernel_size为模糊核的大小(必须为奇数,值越大模糊越明显)，sigma为模糊程度的范围
imshow(path,transform)                                               # 显示原始图像和变换后的图像


# 对图片进行局部遮罩

# from PIL import Image
# import numpy as np
# import random

def cutout_pil_multi(image, mask_size=50, num_masks=3):    # PyTorch中没有内置的Cutout变换,我们可以自己实现一个函数来对图像进行Cutout变换,
    # 该函数接受一个PIL.Image对象,遮挡块的大小和数量作为参数,并返回一个新的PIL.Image对象,其中包含了多个随机位置的遮挡块
    """
    对图像应用多个 Cutout 遮挡块

    参数:
    - image: PIL.Image 对象
    - mask_size: 每个遮挡块的大小（正方形边长）
    - num_masks: 遮挡块的数量
    """
    image_np = np.array(image).copy()
    h, w = image_np.shape[0], image_np.shape[1]

    for _ in range(num_masks):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        y1 = max(0, y - mask_size // 2)
        y2 = min(h, y + mask_size // 2)
        x1 = max(0, x - mask_size // 2)
        x2 = min(w, x + mask_size // 2)

        # 遮挡区域设置为黑色
        image_np[y1:y2, x1:x2, :] = 0

    return Image.fromarray(image_np)

path = r"E:\code\DeepLearning\10卷积神经网络\PetImages\Cat\6039.jpg"  # 图像文件路径
imshow(path, cutout_pil_multi)                                       # 显示原始图像和变换后的图像
