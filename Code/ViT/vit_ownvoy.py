import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import *
import torchvision.transforms as transforms
import torch.nn.init as init

# 필요한 추가 라이브러리들을 여기에 임포트하세요.

from einops import rearrange
from tqdm.auto import tqdm
import wandb
import tyro
from dataclasses import dataclass

torch.manual_seed(0)


@dataclass
class Args:
    wandb_project_name: str = "ViT"
    wandb_run_name: str = "ViT Cifar10 xavier init"
    epochs: int = 200
    learning_rate: int = 0.001
    batch_size: int = 64
    total_steps: int = int(50000 % batch_size) * epochs
    warmup_steps: int = 4000
    num_class: int = 10
    img_size: int = 32
    patch_size: int = 4
    patch_dim: int = patch_size * patch_size * 3
    num_patch: int = int(img_size / patch_size) ** 2
    encoder_in_dim: int = 256
    num_heads: int = 8
    model_dim: int = 512
    encoder_ffn_dim: int = 1024
    num_blocks: int = 6


args = tyro.cli(Args)

wandb.init(
    project=args.wandb_project_name,
    name=args.wandb_run_name,
    config=vars(args),
    save_code=True,
)
transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
}

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transforms["train"]
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transforms["test"]
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=False
)


# 이미지 패치 처리를 위한 클래스
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # TODO: 이미지 패치 처리 로직 구현
        return rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=args.patch_size,
            p2=args.patch_size,
        )


# Class Token 추가 클래스
class AddClassToken(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # TODO: Class Token 변수 초기화
        self.cls_token = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        # TODO: Class Token 추가 로직 구현
        batch_size = len(x)
        batch_cls_token = self.cls_token.repeat((batch_size, 1, 1))
        batch_cls_token.cuda()
        return torch.cat([batch_cls_token, x], dim=1)


# Positional Embedding 클래스
class PositionalEmbedding(nn.Module):
    def __init__(self, num_seq, patch_dim, in_dim):
        super().__init__()
        # TODO: Positional Embedding 변수 초기화
        self.pos_emb = nn.Parameter(torch.randn(num_seq, patch_dim))
        self.mlp = nn.Linear(patch_dim, in_dim)
        init.xavier_normal_(self.mlp.weight)

    def forward(self, x):
        # TODO: Positional Embedding 추가 로직 구현
        x += self.pos_emb
        x = self.mlp(x)
        return x


# Transformer Encoder 클래스
class EncoderBlock(nn.Module):
    def __init__(self, num_heads, seq_len, in_dim, d_model, dff):
        super().__init__()
        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.in_dim = in_dim

        w_q = nn.init.xavier_uniform_(
            torch.empty(self.in_dim, self.d_k), gain=nn.init.calculate_gain("relu")
        )
        w_k = nn.init.xavier_uniform_(
            torch.empty(self.in_dim, self.d_k), gain=nn.init.calculate_gain("relu")
        )
        w_v = nn.init.xavier_uniform_(
            torch.empty(self.in_dim, self.d_k), gain=nn.init.calculate_gain("relu")
        )
        self.q_heads = nn.ParameterList([nn.Parameter(w_q) for _ in range(num_heads)])
        self.k_heads = nn.ParameterList([nn.Parameter(w_k) for _ in range(num_heads)])
        self.v_heads = nn.ParameterList([nn.Parameter(w_v) for _ in range(num_heads)])

        self.o_w = nn.Linear(self.d_model, self.d_model, bias=False).cuda()
        init.xavier_normal_(self.o_w.weight)

        self.conv1 = nn.Conv2d(self.in_dim, self.d_model, 1)
        self.layer_norm1 = nn.LayerNorm([seq_len, self.d_model])
        self.layer_norm2 = nn.LayerNorm([seq_len, in_dim])

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, dff),
            nn.GELU(),
            nn.Linear(dff, self.in_dim),
        )

        self.conv2 = nn.Conv2d(self.d_model, self.in_dim, 1)

    def one_one_conv(self, x, in_channel, out_channel):
        self.conv = nn.Conv2d(in_channel, out_channel, 1).cuda()
        # 현재 x.shape: (B, seq_len, C)
        x = x.transpose(0, 2).unsqueeze(dim=0)
        # (1,C, seq_len, B)의 채널로 바꾸어줌. (conv가 C에 대해서 하니까)
        x = self.conv(x)
        # x.shape (1,out_C, seq_len, B)
        x = x.transpose(1, 3).squeeze()
        # x.shape (B, seq_len, C)로 원상복귀
        return x

    def forward(self, x):
        # TODO: Transformer Encoder 로직 구현
        q_head_weights = torch.cat([h for h in self.q_heads], dim=1)
        k_head_weights = torch.cat([h for h in self.k_heads], dim=1)
        v_head_weights = torch.cat([h for h in self.v_heads], dim=1)
        q = x @ q_head_weights
        k = x @ k_head_weights
        v = x @ v_head_weights

        dot = (
            torch.matmul(q, torch.permute(k, (0, 2, 1)))
            / torch.sqrt(torch.tensor([self.d_model])).cuda()
        )
        dot = torch.softmax(dot, dim=-1)
        attention = torch.matmul(dot, v)
        attention = self.o_w(attention)
        x = self.one_one_conv(x, self.in_dim, self.d_model)

        # Residual Connection
        attention = x + attention

        attention = self.layer_norm1(attention)

        # Layer normalization

        xx = self.ffn(x)
        x = self.one_one_conv(x, self.d_model, self.in_dim)

        # Residual Connection & BN
        x = x + xx
        x = self.layer_norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_heads, num_seq, in_dim, d_model, ffn_dim, num_blocks):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(num_heads, num_seq, in_dim, d_model, ffn_dim)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x


# Classifier Head 클래스
class ClassifierHead(nn.Module):
    def __init__(self, num_classes, in_dim):
        super().__init__()
        self.classify_mlp = nn.Linear(in_dim, num_classes)
        init.xavier_normal_(self.classify_mlp.weight)

    def forward(self, x):
        # TODO: Classifier Head 로직 구현
        return self.classify_mlp(x)


# 전체 ViT 모델 조립 클래스
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.add_class_token = AddClassToken(dim=args.patch_dim)
        self.positional_embedding = PositionalEmbedding(
            num_seq=args.num_patch + 1,
            patch_dim=args.patch_dim,
            in_dim=args.encoder_in_dim,
        )
        self.encoder = Encoder(
            num_heads=args.num_heads,
            num_seq=args.num_patch + 1,
            in_dim=args.encoder_in_dim,
            d_model=args.model_dim,
            ffn_dim=args.encoder_ffn_dim,
            num_blocks=args.num_blocks,
        )
        self.classifier = ClassifierHead(
            in_dim=args.encoder_in_dim, num_classes=args.num_class
        )

    def forward(self, x):
        # TODO: 전체 모델 순서대로 로직 구현
        x = self.patch_embedding(x)
        x = self.add_class_token(x)
        x = self.positional_embedding(x)
        x = self.encoder(x)
        x = self.classifier(x[:, 0, :])
        return x


# 모델 초기화 및 테스트
model = ViT().cuda()

optimizer = AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    amsgrad=False,
)

"""
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        super(GradualWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr_scale = step / self.warmup_steps
        else:
            # Convert the calculation to a Tensor to use with torch.cos()
            step_tensor = torch.tensor(
                [(step - self.warmup_steps) / (self.total_steps - self.warmup_steps)],
                dtype=torch.float32,
            )
            lr_scale = 0.5 * (1 + torch.cos(torch.pi * step_tensor))
            lr_scale = lr_scale.item()  # Convert back to float

        return [base_lr * lr_scale for base_lr in self.base_lrs]
 """

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


def train(model, trainloader):
    for batch in tqdm(trainloader, desc="train", leave=False):
        data, label = batch
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()

        pred = model(data)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        wandb.log({"loss": loss})
        pred = pred.argmax(dim=1)
        num_samples = label.size(0)
        num_correct = (pred == label).sum()
        wandb.log({"train_acc": (num_correct / num_samples * 100).item()})
        loss.backward()
        optimizer.step()
        scheduler.step()


@torch.inference_mode()
def evaluate(model, testloader):
    model.eval()
    num_samples = 0
    num_correct = 0
    for batch in tqdm(testloader, desc="eval", leave=False):
        data, label = batch
        data = data.cuda()
        label = label.cuda()

        pred = model(data)
        pred = pred.argmax(dim=1)
        num_samples += label.size(0)
