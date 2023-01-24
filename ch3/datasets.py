# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import torchvision

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(os.path.join(args.data_path, args.data_set), train=is_train, download=False, transform=transform)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(os.path.join(args.data_path, args.data_set), train=is_train, download=False, transform=transform)
        nb_classes = 10
    elif args.data_set == 'ImageNet':
        root = os.path.join(args.data_path, args.data_set, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'FractalDB-1k':
        root = os.path.join(args.data_path, args.data_set, 'train')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, args):
    if args.data_set == 'CIFAR10' or args.data_set == 'CIFAR100':
        size = 32
        resize = transforms.Resize(224) if args.finetune else transforms.Resize(32)
        # CIFAR-10/100
        if is_train:
            transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            resize,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
            resize,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        return transform

    resize_im = args.input_size > 32
    mean = [0.485, 0.456, 0.406] if args.data_set == 'ImageNet' else [0.5, 0.5, 0.5]
    std = [0.229, 0.224, 0.225] if args.data_set == 'ImageNet' else [0.5, 0.5, 0.5]
    if is_train:
        if not args.aug:
            transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                        ])
            return transform
        # this should always dispatch to transforms_imagenet_train
        # ImageNet only
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform
    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)