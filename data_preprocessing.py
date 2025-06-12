import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# 新增：用于保存和加载数据集
import torch


class MedicalDataset(Dataset):
    def __init__(self, data_root, image_dir="data1", mask_dir="data2", transform=None, image_size=512):
        self.data_root = data_root
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # 自动获取图像与掩码路径
        self.images = sorted([
            os.path.join(data_root, image_dir, f)
            for f in os.listdir(os.path.join(data_root, image_dir))
            if f.lower().endswith(('.png'))
        ])
        self.masks = sorted([
            os.path.join(data_root, mask_dir, f)
            for f in os.listdir(os.path.join(data_root, mask_dir))
            if f.lower().endswith(('.jpg', '.jdg'))
        ])

        self._validate_size()  # 校验数据完整性
        self.transform = transform

    def _validate_size(self):
        if len(self.images) != len(self.masks):
            raise ValueError(f"图像与掩码数量不匹配：{len(self.images)} vs {len(self.masks)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 安全读取文件
        img = self._safe_read(self.images[idx])
        mask = self._safe_read(self.masks[idx], is_mask=True)

        # 尺寸统一化
        img = self._resize(img)
        mask = self._resize(mask, is_mask=True)

        # 预处理
        img = self._preprocess_image(img)
        mask = self._preprocess_mask(mask)

        # 数据增强（仅训练集使用）
        if self.transform:
            img, mask = self.transform((img, mask))

        # 确保内存连续
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()

    def _resize(self, array, is_mask=False):
        """根据类型选择插值方式"""
        return cv2.resize(
            array, (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        )

    def _safe_read(self, path, is_mask=False):
        """处理中文路径和异常文件"""
        try:
            img = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR
            )
            if img is None:
                raise FileNotFoundError(f"文件损坏或路径错误：{path}")
            return img
        except Exception as e:
            print(f"警告：{path} 加载失败，使用空白图像替代 | 错误：{str(e)}")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8) if not is_mask else np.zeros(
                (self.image_size, self.image_size), dtype=np.uint8)

    def _preprocess_image(self, img):
        """图像预处理：转灰度+归一化"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img[np.newaxis, ...] / 255.0  # [1, H, W]

    def _preprocess_mask(self, mask):
        """掩码预处理：归一化+二值化"""
        mask = mask[np.newaxis, ...] / 255.0  # 归一化
        return np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)  # 二值化


class DataTransform:
    """数据增强（训练集专用：随机翻转、裁剪、形变、添加噪声）"""

    def __init__(self,
                 crop_size=480,  # 随机裁剪大小
                 noise_level=0.05,  # 高斯噪声水平
                 elastic_alpha=100,  # 弹性形变参数
                 elastic_sigma=10):  # 弹性形变参数
        self.crop_size = crop_size
        self.noise_level = noise_level
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

    def __call__(self, data_pair):
        img, mask = data_pair

        # 1. 随机水平/垂直翻转
        if random.random() > 0.5:
            img = np.flip(img, axis=2)  # 水平翻转
            mask = np.flip(mask, axis=2)
        if random.random() > 0.5:
            img = np.flip(img, axis=1)  # 垂直翻转
            mask = np.flip(mask, axis=1)

        # 2. 随机旋转90/180/270度
        if random.random() > 0.5:
            k = random.randint(1, 3)  # 旋转次数(1-3次，每次90度)
            img = np.rot90(img, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(1, 2))

        # 3. 随机裁剪（保持原始尺寸）
        if random.random() > 0.5 and self.crop_size < img.shape[1]:
            h, w = img.shape[1], img.shape[2]
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)

            # 裁剪
            img = img[:, y:y + self.crop_size, x:x + self.crop_size]
            mask = mask[:, y:y + self.crop_size, x:x + self.crop_size]

            # 缩放回原始尺寸
            img = cv2.resize(img[0], (w, h), interpolation=cv2.INTER_LINEAR)[np.newaxis, ...]
            mask = cv2.resize(mask[0], (w, h), interpolation=cv2.INTER_NEAREST)[np.newaxis, ...]

        # 4. 添加高斯噪声（仅对图像）
        if random.random() > 0.5:
            noise = np.random.normal(0, self.noise_level, img.shape)
            img = np.clip(img + noise, 0, 1)  # 限制在[0,1]范围

        # 5. 随机弹性形变（可选，计算量较大）
        if random.random() > 0.5:
            img, mask = self._elastic_transform(img, mask)

        return img, mask

    def _elastic_transform(self, img, mask):
        """弹性形变增强（参考Simard et al. 2003）"""
        from scipy.ndimage.interpolation import map_coordinates
        from scipy.ndimage.filters import gaussian_filter

        # 只对单通道处理
        img = img[0]
        mask = mask[0]

        h, w = img.shape
        # 生成随机位移场
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.elastic_sigma) * self.elastic_alpha

        # 创建网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # 应用变换
        img = map_coordinates(img, indices, order=1).reshape(h, w)
        mask = map_coordinates(mask, indices, order=0).reshape(h, w)

        return img[np.newaxis, ...], mask[np.newaxis, ...]


def visualize_sample(img_tensor, mask_tensor, title_prefix=""):
    """可视化样本（支持训练/验证/测试集标注）"""
    img_np = img_tensor.squeeze().numpy()
    mask_np = mask_tensor.squeeze().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap='gray', vmin=0, vmax=1)
    plt.title(f"{title_prefix}image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
    plt.title(f"{title_prefix}mask")
    plt.axis('off')
    plt.show()


# 新增：保存划分好的数据集
def save_datasets(train_dataset, val_dataset, test_dataset, save_dir='./saved_datasets'):
    """保存划分好的训练集、验证集和测试集"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存前先移除数据增强变换（避免保存随机状态）
    train_dataset.dataset.transform = None
    val_dataset.dataset.transform = None
    test_dataset.dataset.transform = None

    # 保存数据集
    torch.save(train_dataset, os.path.join(save_dir, 'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(save_dir, 'val_dataset.pt'))
    torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pt'))

    print(f"数据集已保存至: {save_dir}")

    # 恢复训练集的数据增强变换
    train_dataset.dataset.transform = DataTransform()


# 新增：加载保存的数据集
def load_datasets(save_dir='./saved_datasets'):
    """加载保存的训练集、验证集和测试集"""
    try:
        train_dataset = torch.load(os.path.join(save_dir, 'train_dataset.pt'))
        val_dataset = torch.load(os.path.join(save_dir, 'val_dataset.pt'))
        test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pt'))

        # 恢复训练集的数据增强变换
        train_dataset.dataset.transform = DataTransform()

        print(f"已从 {save_dir} 加载数据集")
        return train_dataset, val_dataset, test_dataset
    except FileNotFoundError:
        print(f"错误：未找到保存的数据集，请先运行数据划分并保存！")
        return None, None, None


if __name__ == "__main__":
    DATA_ROOT = "./data"  # 数据根目录
    IMAGE_SIZE = 512  # 统一尺寸
    BATCH_SIZE = 4  # 批次大小
    RANDOM_SEED = 42  # 随机种子，确保划分可复现
    SAVE_DIR = './saved_datasets'  # 保存数据集的目录

    # 尝试加载已保存的数据集
    train_dataset, val_dataset, test_dataset = load_datasets(SAVE_DIR)

    # 如果没有保存的数据集，则进行划分并保存
    if train_dataset is None:
        print("未找到保存的数据集，重新进行划分...")

        # ====================== 数据集划分 ======================
        # 实例化完整数据集（不包含数据增强，划分后再应用）
        full_dataset = MedicalDataset(
            data_root=DATA_ROOT,
            image_size=IMAGE_SIZE,
            transform=None  # 划分时不进行数据增强
        )

        # 计算划分比例：7:2:1
        total_samples = len(full_dataset)
        train_size = int(0.7 * total_samples)
        val_size = int(0.2 * total_samples)
        test_size = total_samples - train_size - val_size

        # 确保划分尺寸正确（处理余数问题）
        train_size, val_size, test_size = map(int, [train_size, val_size, test_size])

        # 随机划分数据集（固定随机种子保证可复现）
        random.seed(RANDOM_SEED)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )

        # ====================== 为训练集/验证集/测试集添加变换 ======================
        # 训练集：包含数据增强
        train_dataset.dataset.transform = DataTransform()
        # 验证集/测试集：无数据增强，仅标准化
        val_dataset.dataset.transform = None
        test_dataset.dataset.transform = None

        # 保存划分好的数据集
        save_datasets(train_dataset, val_dataset, test_dataset, SAVE_DIR)

    # ====================== 创建 DataLoader ======================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ====================== 验证划分结果 ======================
    print(f"数据集划分结果：")
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"验证集样本数：{len(val_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")


    # ====================== 可视化各数据集样本 ======================
    def visualize_loader(loader, title_prefix):
        """可视化数据加载器中的首个样本"""
        try:
            images, masks = next(iter(loader))
            print(f"\n{title_prefix}数据形状：")
            print(f"图像张量：{images.shape}")
            print(f"掩码张量：{masks.shape}")
            visualize_sample(images[0], masks[0], title_prefix=title_prefix)
        except StopIteration:
            print(f"{title_prefix}数据集为空，请检查划分比例！")

    # 可选：取消注释以下行以可视化数据集样本
    visualize_loader(train_loader, "training")
    visualize_loader(val_loader, "val")
    visualize_loader(test_loader, "test")