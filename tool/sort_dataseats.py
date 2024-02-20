import os
import shutil
import re

def sort_files(input_folders, output_folder):
    # 获取所有文件的路径
    file_paths = []
    for folder in input_folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and (filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.json')):
                file_paths.append(file_path)

    # 按文件名进行递增排序
    sorted_file_paths = sorted(file_paths, key=lambda x: extract_number(os.path.basename(x)))

    # 创建新的输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 复制文件到新的文件夹中
    for i, file_path in enumerate(sorted_file_paths):
        basename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(basename)
        new_basename = f"{i+1:04d}_{base_name}{ext}"
        output_path = os.path.join(output_folder, new_basename)
        shutil.copyfile(file_path, output_path)

    print("文件排序完成，已保存至新文件夹：", output_folder)

def extract_number(file_name):
    # 使用正则表达式提取文件名中的数字部分
    number = re.findall(r'\d+', file_name)
    return int(number[0]) if number else -1

# 调用示例
input_folders = [
                 r"D:\数据集\标签\0817结果\1\1.1", r"D:\数据集\标签\0817结果\1\1.2", r"D:\数据集\标签\0817结果\1\1.3",
                 r'D:\数据集\标签\0817结果\2\2.1', r'D:\数据集\标签\0817结果\2\2.2', r'D:\数据集\标签\0817结果\2\2.3',
                 r'D:\数据集\标签\0817结果\3\3.1',
                 r'D:\数据集\标签\0817结果\5\2\1',r'D:\数据集\标签\0817结果\5\2\2',r'D:\数据集\标签\0817结果\5\2\3',
                 r'D:\数据集\标签\0817结果\5\2\4',r'D:\数据集\标签\0817结果\5\2\5',r'D:\数据集\标签\0817结果\5\2\6'
                 ]  # 输入文件夹路径列表
output_folder = r"D:\数据集\标签\0817结果\out1"  # 新文件夹路径
sort_files(input_folders, output_folder)

##将标注的数据及json按递增顺序放到一个文件夹里
