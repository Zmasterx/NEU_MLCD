import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, train=True):
        if(train):
            self.images = images
            self.labels = labels
            # print(labels.shape) #torch.Size([64, 2]) torch.Size([16, 2]),这里检查的标签维度还是正常的
            self.transform = transform
            dataset = []
            for i in range(len(labels)):
                temp_img = Image.open(images[i])
                temp_img = self.transform(temp_img)
                dataset.append((temp_img, labels[i]))
            # print(dataset[0])
            self.dataset = dataset
            self.len = len(self.labels)
        else:
            self.images = images
            self.transform = transform
            dataset = []
            # print(len(images))
            for i in range(len(images)):
                temp_img = Image.open(images[i])
                temp_img = self.transform(temp_img)
                dataset.append(temp_img)
            self.dataset = dataset
            self.len = len(images)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return self.len


def load_dataset_test(self):
    # 加载图片位置
    tra_data = []
    all_imgs_path = glob.glob(r'../../data_original/traindataset/*.jpg') #读入了文件的路径
    for ip in all_imgs_path:
        tra_data.append(ip)

    val_data = []
    predict_imgs_path = glob.glob(r'../../data_original/test_crop/*.jpg')
    for ip in predict_imgs_path:
        val_data.append(ip)

    predict_data = []
    predict_imgs_path = glob.glob(r'../../data_original/test_crop/*.jpg')
    for ip in predict_imgs_path:
        predict_data.append(ip)

    # 加载标签
    tra_labels = np.loadtxt(r'../../data_original/fovea_localization_traindataest_GT_crop.csv', delimiter=",", skiprows=1, usecols=[1, 2])
    tra_labels = torch.from_numpy(tra_labels)
    # print(type(labels), labels[0]) #<class 'torch.Tensor'> torch.Size([80, 2])
    val_labels = np.loadtxt(r'../../data_original/2_crop.csv', delimiter=",", skiprows=1, usecols=[1])
    val_labels = torch.from_numpy(val_labels)
    val_labels = val_labels.reshape(20, 2)
    # print(val_labels)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), #考虑要不要先不改变图像的大小了，因为这样还得写变换中心位置坐标的代码！！！
        transforms.ToTensor( ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = MyDataset(images=tra_data, labels=tra_labels, transform=transform)
    val_dataset = MyDataset(images=val_data, labels=val_labels, transform=transform)
    predict_dataset = MyDataset(images=predict_data, transform=transform, train=False)

    tr_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=0,
                           pin_memory=True)
    va_loader = DataLoader(val_dataset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=0,
                           pin_memory=True)
    pr_loader = DataLoader(predict_dataset, batch_size=20, shuffle=False)
    return tr_loader, va_loader, pr_loader

# load_dataset_test()
