from datetime import datetime
from pathlib import Path
import torch
import torch.utils.tensorboard as tb

def test_logging(logger: tb.SummaryWriter):
    # 强化后的最简“训练”循环，仅用于写日志
    global_step = 0
    for epoch in range(10):
        metrics = {"train_acc": [], "val_acc": []}

        # ---- train ----
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)

            # 1) 每一步都记训练 loss
            logger.add_scalar("train_loss", dummy_train_loss, global_step)

            # 2) 先把本步的10个随机数平均，留到epoch末统一求均值
            metrics["train_acc"].append(dummy_train_accuracy.mean())

            global_step += 1

        # 在该 epoch 的“最后一个训练 step”位置记训练精度
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        logger.add_scalar("train_accuracy", epoch_train_acc.item(), global_step - 1)

        # ---- val ----
        # 注意：验证也用同一个 epoch 的 seed，且只做10次
        torch.manual_seed(epoch)
        for _ in range(10):
            dummy_val_acc = epoch / 10.0 + torch.randn(10)
            metrics["val_acc"].append(dummy_val_acc.mean())

        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        logger.add_scalar("val_accuracy", epoch_val_acc.item(), global_step - 1)
