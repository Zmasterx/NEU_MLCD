# -*- coding: utf-8 -*-
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models

from dr_dataset import DRDataset
from dr_model import train_model, eval_model
from dr_cam import calc_cam

np.random.seed(0)

# user parameters
epochs = 200
scale = 299
batch_size = 32
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
    # retina_df.dropna(inplace = True)
    # retina_df = retina_df[retina_df['exists']]
    
    # retina_df['eye'] = retina_df['image'].map(lambda x: 1 if x.split('_')[-1]=='left' else 0)
    # retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
    retina_df['PatientId'] = list(range(1, 1001))
    print(retina_df)
    n_class = 1+retina_df['level'].max()  # 值为4
    # sanity check 对数据进行初步的检查和可视化
    # retina_df.sample(3)
    # retina_df[['level', 'eye']].hist(figsize = (10, 5))
    
    ## split data into training and validation sets
    from sklearn.model_selection import train_test_split
    rr_df = retina_df[['PatientId', 'level']].drop_duplicates()
    train_ids, valid_ids = train_test_split(rr_df['PatientId'], 
                                       test_size = 0.1,
                                       random_state = 2018,
                                       stratify = rr_df['level'])
    train_df = retina_df[retina_df['PatientId'].isin(train_ids)]
    valid_df = retina_df[retina_df['PatientId'].isin(valid_ids)]
    print('train', train_df.shape[0], 'validation', valid_df.shape[0])
    # # sanity check 再次对数据进行检查与可视化
    # train_df[['level', 'eye']].hist(figsize = (10, 5))
    # valid_df[['level', 'eye']].hist(figsize = (10, 5))
    
    ## Put data into Dataset class for PyTorch framework with data augmentation
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
    model_conv = models.inception_v3(pretrained=True)
    print("Use GPU: ", use_gpu)

    freeze_layers = True

    if freeze_layers:
        for i, param in model_conv.named_parameters():
            param.requires_grad = False

        ct = []
        for name, child in model_conv.named_children():
            if "Conv2d_4a_3x3" in ct:
                for params in child.parameters():
                    params.requires_grad = True
            else:
                ct.append(name)

        # for i, param in model_conv.named_parameters():
            #     print(i, param.requires_grad)`

        # for name, child in model_conv.named_children():
        #     for name2, child2 in child.named_parameters():
        #         if "features.7.bias" in ct:
        #             for params in child2.parameters():
        #                 params.requires_grad = True
        #         ct.append(name2)
        #     ct.append(name)
    
        #for name, child in model_conv.named_children():
        #    for name_2, params in child.named_parameters():
        #        print(name_2, params.requires_grad)

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, n_class)
    
    print("[Using CrossEntropyLoss...]")
    criterion = torch.nn.CrossEntropyLoss()
    
    print("[Using small learning rate with momentum...]")
    #optimizer_conv = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr= 1e-3, momentum=0.9)
    # optimizer_conv = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr = 0.0001)
    optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr=0.0001 )
    print("[Creating Learning rate scheduler...]")
    # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    
    dataloaders = {'train':train_dl, 'valid':valid_dl}
    dataset_sizes = {'train': len(train_ds), 'valid': len(valid_ds)}
    
    if is_training:
        print("[Training the model begun ....]")
        # train_model function is here: https://github.com/Prakashvanapalli/pytorch_classifiers/blob/master/tars/tars_training.py
        model_ft = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer,
                             num_epochs=epochs)
        torch.save(model_ft.state_dict(), os.path.join(output_dir, "best_dr_state_Inc"))
        torch.save(model_ft, os.path.join(output_dir, "best_dr_Inc"))

    # ## visualize the CNN
    # model_best = torch.load(os.path.join(output_dir, "best_dr_Inc"))
    # results, _ = calc_cam(output_dir, retina_df['path'][0], model_best, val_transforms, 'inception_v3', 3, True)
    #
    # fig = plt.figure(4, figsize=(20, 12))
    # for i in range(len(results)):
    #     pil_im = Image.open(results[i])
    #     plt.subplot(3,5,i+1)
    #     plt.imshow(np.asarray(pil_im))
    #     plt.title(results[i])
    # plt.show()
