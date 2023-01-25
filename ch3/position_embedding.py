# 必要なモジュールをインポート
import os
import math
import models
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vit_tiny_16_224', choices=['vit_tiny_16_224', 'vit_small_16_224'], type=str, help='model name')
parser.add_argument('--checkpoint', default='/path/to/checkpoint.pth', type=str, help='checkpoint')
args = parser.parse_args()

# ViTモデルを読み込む
model = create_model(args.model, pretrained=False)
# 学習済みモデルを読み込む
#finetune = os.path.join('./path/to/file.pth')
checkpoint = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(checkpoint["model"])
model.eval()

# position embedding
# モデルから位置埋め込みを読み込む
# N:パッチ数+クラストークン、D:次元数
pos_embed = model.state_dict()['pos_embed'] # shape:(1, N, D)
H_and_W = int(math.sqrt(pos_embed.shape[1]-1)) # クラストークン分を引いて平方根をとる
# パッチ間のコサイン類似度を求め可視化
fig = plt.figure(figsize=(10, 10))
for i in range(1, pos_embed.shape[1]):
    sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
    sim = sim.reshape((H_and_W, H_and_W)).detach().cpu().numpy()
    ax = fig.add_subplot(H_and_W, H_and_W, i)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(sim)
plt.savefig("./position_embedding.pdf")
plt.clf()