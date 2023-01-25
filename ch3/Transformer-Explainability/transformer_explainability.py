# 必要なモジュールをインポート
import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vit_tiny_16_224', choices=['vit_tiny_16_224', 'vit_small_16_224'], type=str, help='model name')
parser.add_argument('--checkpoint', default='/path/to/checkpoint.pth', type=str, help='checkpoint') # '../ImageNet/tiny16/best_checkpoint.pth'
args = parser.parse_args()

# 画像上のマスクからヒートマップを作成
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# Attention Weightを取得するための関数
def generate_visualization(original_image, class_index=None):
    # モデルの勾配とAttention RolloutからAttention Weightを求める
    # N:パッチ数 (196)
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach() # shape: (1, N)
    # 14x14にリサイズしてバイリニア補間しながらAttention Mapを可視化
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

if args.model == 'vit_tiny_16_224':
    from baselines.ViT.ViT_LRP import vit_tiny_16_224 as vit_LRP
    from baselines.ViT.ViT_explanation_generator import LRP_VIS
    # ViTモデルを読み込む
    model = vit_LRP(pretrained=False).cuda()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # モデルの勾配を求める
    attribution_generator = LRP_VIS(model)
else: # vit_small_16_224
    from baselines.ViT.ViT_LRP import vit_small_16_224 as vit_LRP
    from baselines.ViT.ViT_explanation_generator import LRP_VIS
    # ViTモデルを読み込む
    model = vit_LRP(pretrained=False).cuda()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # モデルの勾配を求める
    attribution_generator = LRP_VIS(model)

# 画像をリサイズしてセンタークロップ
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

########################################################################################################
# dog or cat
# 画像ファイルを読み込み、猫と犬のAttention Mapを可視化
image = Image.open('samples/catdog.png')
dog_cat_image = transform(image)

fig, axs = plt.subplots(1, 3, figsize=[20, 20])
axs[0].imshow(image);
axs[0].get_xaxis().set_visible(False);
axs[0].get_yaxis().set_visible(False);

output = model(dog_cat_image.unsqueeze(0).cuda())

# 予測クラス:'Tiger cat' (クラス番号: 282). 下記他のcatのクラス番号一覧
# generate visualization for class 281: 'tabby cat'
# generate visualization for class 282: 'tiger cat'
# generate visualization for class 283: 'Persian cat'
# generate visualization for class 284: 'Siamese cat'
# generate visualization for class 285: 'Egyptian cat'
cat = generate_visualization(dog_cat_image, class_index=282)

# class_indexを変えることで、任意の対象クラスのAttention Mapを可視化可能
# 例えば、'Bull mastiff' (クラス番号: 243)のAttention Mapを可視化したい場合には、class_indexを243にする
# generate visualization for class 243: 'Bull mastiff'
# generate visualization for class 244: 'Tibetan mastiff'
# generate visualization for class 245: 'French bulldog'
dog = generate_visualization(dog_cat_image, class_index=243)

# 可視化
axs[1].imshow(cat);
axs[1].get_xaxis().set_visible(False);
axs[1].get_yaxis().set_visible(False);
axs[2].imshow(dog);
axs[2].get_xaxis().set_visible(False);
axs[2].get_yaxis().set_visible(False);
plt.savefig("./catdog.pdf")


########################################################################################################
# zebra or elephant
# 画像ファイルを読み込み、シマウマと象のAttention Mapを可視化
image = Image.open('samples/el2.png')
tusker_zebra_image = transform(image)

fig, axs = plt.subplots(1, 3, figsize=[20, 20])
axs[0].imshow(image);
axs[0].get_xaxis().set_visible(False);
axs[0].get_yaxis().set_visible(False);

output = model(tusker_zebra_image.unsqueeze(0).cuda())

# 予測クラス:'zebra' (クラス番号: 340).
# generate visualization for class 340: 'zebra'
tusker = generate_visualization(tusker_zebra_image, class_index=340)

# class_indexを変えることで、任意の対象クラスのAttention Mapを可視化可能
# 例えば、'African elephant' (クラス番号: 386)のAttention Mapを可視化したい場合には、class_indexを386にする
# generate visualization for class 101: 'Tusker'
# generate visualization for class 385: 'Indian elephant'
# generate visualization for class 386: 'African elephant'
zebra = generate_visualization(tusker_zebra_image, class_index=101)

# 可視化
axs[1].imshow(tusker);
axs[1].get_xaxis().set_visible(False);
axs[1].get_yaxis().set_visible(False);
axs[2].imshow(zebra);
axs[2].get_xaxis().set_visible(False);
axs[2].get_yaxis().set_visible(False);
plt.savefig("./el2.pdf")
