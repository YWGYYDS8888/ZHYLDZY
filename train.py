import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from data_preprocessing import MedicalDataset, DataTransform, DataLoader, save_datasets, load_datasets
import matplotlib.pyplot as plt
import numpy as np

# 导入模型定义
from model import DeepLabV3Plus

# 配置参数
DATA_ROOT = "./data"
IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./saved_models"
DATASET_SAVE_DIR = "./saved_datasets"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATASET_SAVE_DIR, exist_ok=True)

# 新增：学习率调度器配置
LR_SCHEDULER = "plateau"  # 可选: "plateau", "cosine", "none"
PLATEAU_PATIENCE = 5  # 验证损失停滞多少轮后降低学习率
PLATEAU_FACTOR = 0.5  # 学习率降低的因子
MIN_LR = 1e-6  # 最小学习率


def load_prepared_data():
    # 加载数据集
    train_dataset, val_dataset, _ = load_datasets(DATASET_SAVE_DIR)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    lr_history = []  # 新增：记录学习率变化

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # 新增：记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # 新增：更新学习率调度器
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)  # 基于验证损失更新
            else:
                scheduler.step()  # 基于epoch更新

        # 跟踪最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()  # 保存最佳模型状态

        # 新增：在日志中显示学习率
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Best Val Loss: {best_val_loss:.4f} | LR: {current_lr:.6f}")

    # 新增：绘制损失曲线和学习率曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'o-', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, 's-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), lr_history, '^-', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')  # 使用对数刻度以便更好地观察变化

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"))
    plt.close()

    # 返回最佳模型状态和损失
    return best_model_state, train_losses, val_losses


def main():
    print("开始运行训练脚本...")

    # 加载数据集
    print("\n--- 加载数据集 ---")
    train_loader, val_loader = load_prepared_data()

    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")

    # 初始化模型
    print("\n--- 模型初始化 ---")
    model = DeepLabV3Plus(
        in_channels=1,
        num_classes=1,
        output_stride=16
    ).to(DEVICE)
    print(f"模型已加载至 {DEVICE}")

    # 定义损失函数和优化器
    print("\n--- 配置训练参数 ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    # 新增：设置学习率调度器
    scheduler = None
    if LR_SCHEDULER == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',  # 监控指标是减小
            factor=PLATEAU_FACTOR,  # 学习率降低因子
            patience=PLATEAU_PATIENCE,  # 等待轮数
            verbose=True,  # 打印更新信息
            min_lr=MIN_LR  # 最小学习率
        )
        print(f"使用 ReduceLROnPlateau 学习率调度器 (patience={PLATEAU_PATIENCE}, factor={PLATEAU_FACTOR})")
    elif LR_SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,  # 周期长度
            eta_min=MIN_LR  # 最小学习率
        )
        print(f"使用 CosineAnnealingLR 学习率调度器 (T_max={NUM_EPOCHS})")
    else:
        print("不使用学习率调度器")

    print(f"使用 {criterion.__class__.__name__} 损失函数和 {optimizer.__class__.__name__} 优化器")

    # 训练模型并获取最佳模型状态
    print("\n--- 开始训练 ---")
    best_model_state, train_losses, val_losses = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, DEVICE
    )

    # 将最佳模型状态加载到模型中
    model.load_state_dict(best_model_state)

    # 保存最终模型（即最佳模型）
    final_model_path = os.path.join(SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # 训练结果汇总
    print("\n--- 训练完成 ---")
    print(f"最佳验证损失: {min(val_losses):.4f}")
    print(f"最终模型（最佳模型）保存路径: {final_model_path}")
    print(f"训练曲线保存路径: {os.path.join(SAVE_DIR, 'training_curves.png')}")


if __name__ == "__main__":
    main()
