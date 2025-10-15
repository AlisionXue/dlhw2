# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# ① 分类损失函数（10 pts）
# -------------------------
class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred: [B, num_classes]
        target: [B]
        """
        return F.cross_entropy(pred, target)


# -------------------------
# ② 线性分类器（5 pts）
# -------------------------
class LinearClassifier(nn.Module):
    def __init__(self, in_dim=3 * 64 * 64, num_classes=6):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------------------------
# ③ 基础 MLP（20 pts）
# -------------------------
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=3 * 64 * 64, hidden_dim=512, num_classes=6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


# -------------------------
# ④ 深层 MLP（15 pts）
# -------------------------
class MLPDeep(nn.Module):
    def __init__(self, in_dim=3 * 64 * 64, hidden_dim=256, num_layers=4, num_classes=6):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# -------------------------
# ⑤ 残差 MLP（20 pts）
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(x + out)


class MLPResidual(nn.Module):
    def __init__(self, in_dim=3 * 64 * 64, hidden_dim=256, num_blocks=3, num_classes=6):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_in(x))
        for block in self.blocks:
            x = block(x)
        return self.fc_out(x)


# -------------------------
# ⑥ 模型加载函数
# -------------------------
def load_model(model_name: str):
    """
    根据 model_name 返回相应模型实例。
    同时对参数数量加以限制（官方要求）。
    """
    model_name = model_name.lower()

    if model_name == "linear":
        model = LinearClassifier()
    elif model_name == "mlp":
        model = MLPClassifier()
    elif model_name == "mlp_deep":
        model = MLPDeep()
    elif model_name == "mlp_residual":
        model = MLPResidual()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 官方 grader 限制模型大小（参数数量）
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 5e6, f"Model too large ({total_params} params)"
    print(f"[INFO] Model '{model_name}' loaded with {total_params:,} parameters.")
    return model


# -------------------------
# ⑦ 模型保存函数
# -------------------------
def save_model(model):
    torch.save(model.state_dict(), "model.th")
