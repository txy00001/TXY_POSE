# from pycocotools.coco import COCO
#
# my_coco = COCO(r"C:\Users\ASUS\Desktop\CastPose\data\coco\val_coco.json")

from pycocotools.coco import COCO
import json

def validate_coco_json(json_file):
    try:
        coco = COCO(json_file)
        coco.info  # 验证基本信息字段是否存在
        coco.images  # 验证图像信息字段是否存在
        coco.annotations  # 验证注释信息字段是否存在
        coco.categories  # 验证类别信息字段是否存在

        return True
    except Exception as e:
        print(f"Invalid COCO JSON format: {e}")
        return False

# 指定要验证的json文件路径
coco_json_file = r"C:\Users\ASUS\Desktop\CastPose\data\coco\val_coco.json"

# 验证json文件是否符合COCO格式
is_valid = validate_coco_json(coco_json_file)

# # if is_valid:
# #     print("COCO JSON format is valid.")
# # else:
# #     print("COCO JSON format is invalid.")