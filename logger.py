import torch

def test_logging(writer):
    global_step = 0
    for epoch in range(10):
        # 1) 每步写一次训练损失（总共 10*20 次）
        for _ in range(20):
            writer.add_scalar("train_loss", 0.9 ** (global_step / 20.0), global_step)
            global_step += 1

        # 2) 训练精度：seed=epoch，20 组，每组10个N(0,1)取均值，再对20组求均值，最后 + epoch/10
        torch.manual_seed(epoch)
        train_acc = torch.stack([torch.randn(10).mean() for _ in range(20)]).mean() + epoch / 10.0
        # 记在该 epoch 最后一个训练 step 上（global_step-1）
        writer.add_scalar("train_accuracy", train_acc.item(), global_step - 1)

        # 3) 验证精度：同一个 seed=epoch，10 组，每组10个，方式完全同上（只是 10 组）
        torch.manual_seed(epoch)
        val_acc = torch.stack([torch.randn(10).mean() for _ in range(10)]).mean() + epoch / 10.0
        writer.add_scalar("val_accuracy", val_acc.item(), global_step - 1)
