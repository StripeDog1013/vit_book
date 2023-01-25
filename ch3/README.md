# Vision Transformer入門（第3章 実験と可視化によるVision Transformerの探求）

本書籍をご購入頂いてありがとうございます．こちらではVision Transformer入門の第3章で用いたコードを公開しております．学習・評価コードがありますので，2章と合わせてプログラムレベルでVision Transformerの理解を深めて頂ければ幸いです．また，Attention mapや位置埋め込みを可視化するプログラムを公開しておりますので，こちらについてもプログラムレベルで理解頂けるかと思います．ViTのTinyモデルのImageNet-1kとFractalDB-1kで学習した重みを```/ImageNet```と```/FractalDB-1k```にアップロードしておりますので，下記手順に沿って気軽に試してみてください．

## 位置埋め込みの可視化
ViTには位置埋め込みと呼ばれる各パッチトークンに位置情報を付与させる学習可能なパラメータがあります．こちらを可視化するプログラムは```position_embedding.py```で下記のように実行してください．
```
python position_embedding.py --model {model-name} --checkpoint {/path/to/checkpoint.pth}
```
ここで，```--model```はモデル名を示し，{model-name}に例えば```vit_tiny_16_224```と記入します．```--checkpoint```は学習した重みパラメータを読み込む際に必要な引数で，このリポジトリでは```./ImageNet/tiny16/best_checkpoint.pth```などが該当します．プログラムを実行すると下記のように可視化されます．

[写真]

## Attention Mapの可視化
本書籍ではAttention RolloutとTransformer-Explainabilityの2種類のAttention Mapを可視化する方法を用いています．Attention Rolloutを用いたAttention Mapは```attention_rollout.py```で可視化することができます．こちらは下記のように実行します．
```
python attention_rollout.py --model {model-name} --checkpoint {/path/to/checkpoint.pth}
```
引数は```position_embedding.py```と同じものになります．こちらを実行すると下記のように可視化されます．

[写真]

Transformer-Explainabilityを用いたAttention Mapは```/Transformer-Explainability/transformer_explainability.py```で可視化することができます．こちらも下記のように実行します．
```
python transformer_explainability.py --model {model-name} --checkpoint {/path/to/checkpoint.pth}
```
こちらも引数は```position_embedding.py```と同じものになります．こちらを実行すると下記のように可視化されます．

[写真]

また，```/Transformer-Explainability/transformer_explainability.py```の```generate_visualization()```内で指定できる```class_index```を任意の数値に変更することで，下記のように注視する場所(クラスなど)を変更することができます．

[写真]

# 学習と評価
3章で行った実験を試してみましょう．

### データの前処理
ImageNet-1kとFractalDB-1kで事前学習するにあたり，下記ディレクトリ構造に合わせてください．ImageNet-1kはtrain/val，FractalDB-1kはtrainフォルダ下にクラスのフォルダを作成します．[ImageNet-1k](https://image-net.org/download.php)と[FractalDB-1k](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/)のデータのリンク先です．各データをダウンロードしておいてください．

```
# ImageNet-1k
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
      
# FractalDB-1k
/path/to/fractaldb/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
```

## DeiTで設定したデータ拡張を使わない学習と評価
本書籍の表3.2で行ったViT-16TのImageNetの学習は下記になります．こちらはDeiTで設定されたデータ拡張なしで学習するものになります．
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --batch-size 512 --data-path /path/to/dataset_dir/ --data-set ImageNet --output_dir /path/to/out_dir --model vit_tiny_16_224 --no-aug --mixup 0 --cutmix 0 --smoothing 0 --reprob 0
```
ここで，```--nproc_per_node```は各ノードで使用できるGPU数を表し，ご自身の環境によって変更ください．```--no-aug```はDeiTで設定されたデータ拡張を使いません．したがって，これを除くことでDeiTで設定されたデータ拡張を使用できます．```mixup```はmixupのmixする割合を指定でき，mixupを使わない場合は```0```と設定します．```--cutmix, --smoothing, --reprob```についても同様に，使わない場合```0``` と設定します．学習した重みは```--output_dir```に保存し，下記で評価します．
```
python main.py --eval --batch-size 512 --data-path /path/to/dataset_dir/ --data-set ImageNet --model vit_tiny_16_224 --resume /path/to/out_dir/best_checkpoint.pth
```
ここで，```--eval```を指定することで評価用のコードになります．また```--resume```は```output_dir```で保存された```checkpoint.pth```までのパスを指定してください．

次に，FractalDB-1kで事前学習しImageNet-1k等に転移学習してみます．FractalDB-1kは下記で実行します．
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --batch-size 512 --data-path /path/to/dataset_dir/ --data-set FractalDB-1k --output_dir /path/to/fractal_out_dir --model vit_tiny_16_224 --no-aug --mixup 0 --cutmix 0 --smoothing 0 --reprob 0
```
ImageNetで学習する時とほぼ変更はありませんが，```--data-set```のみFractalDB-1kに指定してください．学習された重みは```--output_dir```にエポック毎に保存されていると思います．その重みを用いて下記のように転移学習します．この例ではImageNetになります．
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --batch-size 512 --data-path /path/to/dataset_dir/ --data-set ImageNet --output_dir /path/to/imagenet_out_dir --model vit_tiny_16_224 --no-aug --mixup 0 --cutmix 0 --smoothing 0 --reprob 0 --finetune /path/to/fractal_out_dir/checkpoint.pth
```
事前学習した重みは```--finetune```でパスを指定します．あとは```--data-set```をImageNetに変更，```--output_dir```を任意のパスに変更すると，指定したディレクトリにImageNetで学習された重みパラメータが保存されるかと思います．

## DeiTで設定したデータ拡張を使う学習と評価
続いて，表3.3のDeiTで設定したデータ拡張を使う場合の学習と評価です．こちらは下記の引数を任意に変更することで学習します．
* random horizontal + random resize crop (default),
* AutoAug(--aa original), 
* RandAug(--aa rand-m9-mstd0.5-inc1), 
* mixup(--mixup) : ```default 0.8```
* cutmix(--cutmix) : ```default 1.0```
* color jitter(--color-jitter) : ```default 0.4```
* Random erase prob(--reprob) : ```default 0.25```

例えば，```AutoAug```のImageNetで学習する場合は次のようにします．
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --batch-size 512 --data-path /path/to/dataset_dir/ --data-set ImageNet --output_dir /path/to/out_dir --model vit_tiny_16_224 --aa original
```
すでに```main.py```にmixup等は上記の```default```値で設定しているため，DeiTで設定したデータ拡張を使う場合は特に引数として用いなくても良いかと思います．ただ，表3.3のようにデータ拡張の有無によるAblation Studyをしたい場合は，該当する引数を```0```にすることで，そのデータ拡張を使わずに学習できます．

評価コードは```output_dir```で指定されたパスを```--resume```に設定することで評価できます．
```
python main.py --eval --batch-size 512 --data-path /path/to/dataset_dir/ --data-set ImageNet --model vit_tiny_16_224 --resume /path/to/out_dir/best_checkpoint.pth
```

参考コード
[DeiTのGitHub](https://github.com/facebookresearch/deit)
[Transformer ExplainabilityのGitHub](https://github.com/hila-chefer/Transformer-Explainability)
