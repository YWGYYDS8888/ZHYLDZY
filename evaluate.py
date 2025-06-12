import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from data_preprocessing import load_datasets, MedicalDataset, DataLoader  # 导入加载函数
# 导入模型定义
from model import DeepLabV3Plus

# 配置参数
DATA_ROOT = "./data"
IMAGE_SIZE = 512
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./saved_models/final_model.pth"
SAVE_DIR = "./test_results"
os.makedirs(SAVE_DIR, exist_ok=True)
DATASET_SAVE_DIR = "./saved_datasets"  # 数据集保存路径（需与训练时一致）


# ====================== 数据加载（修改为加载预保存测试集） ======================
def load_test_data():
    # 加载预保存的训练集、验证集、测试集（忽略前两个，只取测试集）
    _, _, test_dataset = load_datasets(DATASET_SAVE_DIR)

    if test_dataset is None:
        raise FileNotFoundError("未找到预保存的测试集，请先在训练阶段划分并保存数据集！")

    # 创建测试集数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"测试集样本数: {len(test_loader.dataset)}")
    return test_loader


# ====================== 模型加载 ======================
def load_model(model_path):
    model = DeepLabV3Plus(
        in_channels=1,
        num_classes=1,
        output_stride=16
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


# ====================== 可视化函数 ======================
def visualize_test_result(img, mask_true, mask_pred, filename):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_true.squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.title('True Mask')
    plt.axis('off')

    mask_pred_binary = torch.sigmoid(mask_pred).detach().cpu().numpy().squeeze()
    mask_pred_binary = np.where(mask_pred_binary > 0.5, 1, 0)
    plt.subplot(1, 3, 3)
    plt.imshow(mask_pred_binary, cmap='gray', vmin=0, vmax=1)
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()


# ====================== 计算评价指标 ======================
def calculate_metrics(pred_mask, true_mask, smooth=1e-5):
    """
    计算Dice系数和IoU指标

    参数:
    pred_mask: 预测掩码 (B, 1, H, W)
    true_mask: 真实掩码 (B, 1, H, W)
    smooth: 平滑项，防止除零错误

    返回:
    dice: Dice系数
    iou: IoU指标
    """
    # 应用sigmoid并二值化预测掩码
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()

    # 计算交集
    intersection = (pred_mask * true_mask).sum(dim=[1, 2, 3])

    # 计算Dice系数: 2*|X∩Y| / (|X|+|Y|)
    dice = (2. * intersection + smooth) / (pred_mask.sum(dim=[1, 2, 3]) + true_mask.sum(dim=[1, 2, 3]) + smooth)

    # 计算IoU: |X∩Y| / (|X∪Y|)
    union = pred_mask.sum(dim=[1, 2, 3]) + true_mask.sum(dim=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return dice.mean().item(), iou.mean().item()


# ====================== 测试函数 ======================
def test_model(model, test_loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    print("\n--- 开始测试 ---")
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # 计算评价指标
            batch_dice, batch_iou = calculate_metrics(outputs, masks)
            total_dice += batch_dice * images.size(0)
            total_iou += batch_iou * images.size(0)

            if i == 0:
                img = images[0].cpu()
                mask_true = masks[0].cpu()
                mask_pred = outputs[0].cpu()
                visualize_test_result(img, mask_true, mask_pred, "test_result.png")

    test_loss = running_loss / len(test_loader.dataset)
    avg_dice = total_dice / len(test_loader.dataset)
    avg_iou = total_iou / len(test_loader.dataset)

    print(f"\n测试集平均损失率: {test_loss:.4f}")
    print(f"平均Dice系数: {avg_dice:.4f}")
    print(f"平均IoU指标: {avg_iou:.4f}")
    print(f"--- 测试完成 ---\n结果图已保存至: {os.path.join(SAVE_DIR, 'test_result.png')}")


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 1. 加载预保存的测试集
    test_loader = load_test_data()

    # 2. 加载训练好的模型
    model = load_model(MODEL_PATH)

    # 3. 执行测试
    test_model(model, test_loader)
