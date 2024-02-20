###划分数据集按8：2的数据集进行划分

import os
import random
import shutil

def split_dataset(image_folder, json_folder, train_ratio):
    # 创建新的文件夹来存放训练集和测试集
    if not os.path.exists('train/images'):
        os.makedirs('train/images')
    if not os.path.exists('train/json'):
        os.makedirs('train/json')
    if not os.path.exists('test/images'):
        os.makedirs('test/images')
    if not os.path.exists('test/json'):
        os.makedirs('test/json')

    image_files = os.listdir(image_folder)
    json_files = os.listdir(json_folder)

    num_samples = len(image_files)
    num_train = int(num_samples * train_ratio)
    train_indices = random.sample(range(num_samples), num_train)
    test_indices = [i for i in range(num_samples) if i not in train_indices]

    for i in train_indices:
        # 将训练集图片复制到train/images文件夹
        image_file = image_files[i]
        shutil.copy(os.path.join(image_folder, image_file), 'train/images')

        # 找到对应的JSON文件并复制到train/json文件夹
        image_name = os.path.splitext(image_file)[0]
        json_file = f'{image_name}.json'
        shutil.copy(os.path.join(json_folder, json_file), 'train/json')

    for i in test_indices:
        # 将测试集图片复制到test/images文件夹
        image_file = image_files[i]
        shutil.copy(os.path.join(image_folder, image_file), 'test/images')

        # 找到对应的JSON文件并复制到test/json文件夹
        image_name = os.path.splitext(image_file)[0]
        json_file = f'{image_name}.json'
        shutil.copy(os.path.join(json_folder, json_file), 'test/json')

# 指定图片和JSON文件的文件夹路径
image_folder = r'D:\datasets\QX_Datasets\new\new_pic'
json_folder = r'D:\datasets\QX_Datasets\new\new_json'

# 指定训练集所占比例
train_ratio = 0.8

# 调用函数进行数据集划分
split_dataset(image_folder, json_folder, train_ratio)




