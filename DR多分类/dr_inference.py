import os
import torch
import glob
from torchvision import transforms
from PIL import Image
import re
import numpy as np
import pandas as pd

scale = 224  # Inc = 299 Eff = 224

# 定义转换
transform = transforms.Compose([
    transforms.Resize(scale),
    transforms.CenterCrop(scale),
    transforms.ToTensor()])

model_path = '../output/best_dr_Eff'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

def extract_number(filename):  # 对读取的图片数据按照数字的大小进行排序
    # 从文件名中提取数字
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# 文件夹路径
predict_data = []
predict_imgs_path = glob.glob(r'../../dataset/test/*.jpg')
predict_imgs_path = sorted(predict_imgs_path, key=extract_number)
for ip in predict_imgs_path:
    predict_data.append(ip)
result = []

# 遍历文件夹
for filename in predict_data:
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # 读取图像
        image_path = os.path.join(filename)
        image = Image.open(image_path).convert("RGB")

        # 对图像进行预处理
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # 增加 batch 维度

        # 进行推理
        with torch.no_grad():
            output = model(input_batch)

        # 处理输出
        _, predicted_class = torch.max(output, 1)
        result.append(predicted_class.numpy())
        print(f"Image: {filename}, Predicted class: {predicted_class.item()}")

print(result)
result = np.array(result)
result = np.reshape(result, (-1,1))
res = pd.DataFrame(result, index=range(1, 301))
res.to_csv(f'../submit_best_Eff.csv')