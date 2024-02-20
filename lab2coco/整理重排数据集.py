###整理数据集，整合到一个文件夹里，把多余的文件夹删除
import os
import shutil
import itertools

# 原始图片和 json 文件夹路径
images_folder = r"D:\datasets\QX_Datasets\all\images"
json_folder = r"D:\datasets\QX_Datasets\all\labelme_json"

# 新图片和 json 文件夹路径
new_images_folder = r"D:\datasets\QX_Datasets\new\new_pic"
new_json_folder = r"D:\datasets\QX_Datasets\new\new_json"

# 获取所有原始图片和 json 文件的路径
image_files = []
json_files = []

for root, dirs, files in os.walk(images_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.png')):
            image_files.append(os.path.join(root, file))

for root, dirs, files in os.walk(json_folder):
    for file in files:
        if file.lower().endswith('.json'):
            json_files.append(os.path.join(root, file))

# 对文件按原始名字进行排序
image_files.sort()
json_files.sort()

# 创建新的图片和 json 文件夹
os.makedirs(new_images_folder, exist_ok=True)
os.makedirs(new_json_folder, exist_ok=True)

# 为了防止同名文件被覆盖，需要给新的文件名加上递增数字
def increment_filename(filename, count=1):
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_{count}{ext}"
    if os.path.exists(os.path.join(new_images_folder, new_filename)) or \
       os.path.exists(os.path.join(new_json_folder, new_filename)):
        return increment_filename(filename, count + 1)
    return new_filename

# 将文件复制到新的文件夹并保持对应关系
for image_file, json_file in itertools.zip_longest(image_files, json_files):
    image_name = os.path.basename(image_file) if image_file else None
    json_name = os.path.basename(json_file) if json_file else None

    # 拷贝图片文件
    if image_name:
        new_image_name = increment_filename(image_name)
        new_image_path = os.path.join(new_images_folder, new_image_name)
        shutil.copy(image_file, new_image_path)

    # 拷贝 json 文件
    if json_name:
        new_json_name = increment_filename(json_name)
        new_json_path = os.path.join(new_json_folder, new_json_name)
        shutil.copy(json_file, new_json_path)
