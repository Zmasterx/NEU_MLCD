# -*- coding: utf-8 -*-
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet

from dr_dataset import DRDataset
from dr_model import train_model, train_model_Eff, eval_model
from dr_cam import calc_cam

np.random.seed(0)

# user parameters
epochs = 200
scale = 224
batch_size = 8
is_training = True


def get_transformations():
    train_transforms = transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.25, saturation=0.25, hue=0.15),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor()])
    val_transforms = transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(scale),
        transforms.ToTensor()])
    return (train_transforms, val_transforms)


if __name__ == '__main__':
    ## read info of available images into DataFrame
    base_image_dir = os.path.join('..')
    train_image_dir = os.path.join(base_image_dir, '../dataset/train')
    test_image_dir = os.path.join(base_image_dir, '../daraset/test')
    output_dir = os.path.join(base_image_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    retina_df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
    print(retina_df)
    retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(train_image_dir,
                                                                      '{}.jpg'.format(x)))
    retina_df['exists'] = retina_df['path'].map(os.path.exists)
    print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

    retina_df['PatientId'] = list(range(1, 1001))
    print(retina_df)
    n_class = 1 + retina_df['level'].max()  # 值为4

    from sklearn.model_selection import train_test_split

    rr_df = retina_df[['PatientId', 'level']].drop_duplicates()
    train_ids, valid_ids = train_test_split(rr_df['PatientId'],
                                            test_size=0.1,
                                            random_state=2018,
                                            stratify=rr_df['level'])
    train_df = retina_df[retina_df['PatientId'].isin(train_ids)]
    valid_df = retina_df[retina_df['PatientId'].isin(valid_ids)]
    print('train', train_df.shape[0], 'validation', valid_df.shape[0])

    (train_transforms, val_transforms) = get_transformations()

    train_ds = DRDataset(train_df[['image', 'level']], train_image_dir, transform=train_transforms)
    valid_ds = DRDataset(valid_df[['image', 'level']], train_image_dir, transform=val_transforms)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    ## Load pre-trained model
    cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    models_dir = os.path.join(cache_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    use_gpu = torch.cuda.is_available()
    model = EfficientNet.from_pretrained('efficientnet-b5')
    print("Use GPU: ", use_gpu)

    # freeze the first few layers. This is done in two stages:
    freeze_layers = True

    # Stage-1 Freezing all the layers

    if freeze_layers:
        # 遍历所有参数，冻结它们
        for i, param in model.named_parameters():
            param.requires_grad = False
            # print(i, param.requires_grad)

        print(20*'#')

        flag = 0
        # 解冻指定层 "_blocks.30._expand_conv.weight"
        for i, param in model.named_parameters():
            if i == '_blocks.20._expand_conv.weight':
                flag = 1
            if flag == 0:
                param.requires_grad = False
            else:
                param.requires_grad = True

        print(20*'#')

        for i, param in model.named_parameters():
            print(i, param.requires_grad)


    # Change the last layer
    # Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
    feature = model._fc.in_features
    model._fc = torch.nn.Linear(in_features=feature, out_features=4, bias=True)


    print("[Using CrossEntropyLoss...]")
    criterion = torch.nn.CrossEntropyLoss()

    print("[Using small learning rate with momentum...]")
    # optimizer_conv = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr= 1e-3, momentum=0.9)
    # optimizer_conv = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.0001)
    optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.0001 )
    print("[Creating Learning rate scheduler...]")
    # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    dataloaders = {'train': train_dl, 'valid': valid_dl}
    dataset_sizes = {'train': len(train_ds), 'valid': len(valid_ds)}
    #
    if is_training:
        print("[Training the model begun ....]")
        # train_model function is here: https://github.com/Prakashvanapalli/pytorch_classifiers/blob/master/tars/tars_training.py
        model_ft = train_model_Eff(model, dataloaders, dataset_sizes, criterion, optimizer,
                               num_epochs=epochs)
        torch.save(model_ft.state_dict(), os.path.join(output_dir, "best_dr_state_Eff"))
        torch.save(model_ft, os.path.join(output_dir, "best_dr_Eff"))

