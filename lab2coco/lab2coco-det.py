import os
import json

Dataset_root = r'F:\help_coco\4'
class_list = {
        'supercategory': 'body',
        'id': 1,
        'name': 'body',

    },


def process_single_json(labelme, img_id):
    '''
    输入labelme的json数据，输出coco格式的每个框的关键点标注信息
    '''

    global ann_id

    coco_annotations = []

    for each_ann in labelme['shapes']:
        if each_ann['shape_type'] == 'rectangle':
            # 个体框元数据
            bbox_dict = {}
            bbox_dict['category_id'] = 1
            bbox_dict['segmentation'] = []
            bbox_dict['iscrowd'] = 0
            bbox_dict['segmentation'] = []
            bbox_dict['image_id'] = img_id
            bbox_dict['id'] = ann_id
            ann_id += 1

            # 获取个体框坐标
            bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_w = bbox_right_bottom_x - bbox_left_top_x
            bbox_h = bbox_right_bottom_y - bbox_left_top_y
            bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]
            bbox_dict['area'] = bbox_w * bbox_h

            # # 筛选出该个体框中的所有关键点
            # bbox_keypoints_dict = {}
            # for each_ann in labelme['shapes']:
            #     if each_ann['shape_type'] == 'point':
            #         x = int(each_ann['points'][0][0])
            #         y = int(each_ann['points'][0][1])
            #         label = each_ann['label']
            #         if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & (y < bbox_right_bottom_y) & (
            #                 y > bbox_left_top_y):
            #             bbox_keypoints_dict[label] = [x, y]
            #
            # bbox_dict['num_keypoints'] = len(bbox_keypoints_dict)
            #
            # # 把关键点按照类别顺序排好
            # bbox_dict['keypoints'] = []
            # for each_class in class_list:
            #     keypoints = each_class['keypoints']
            #     for keypoint in keypoints:
            #         if keypoint in bbox_keypoints_dict:
            #             bbox_dict['keypoints'].append(bbox_keypoints_dict[keypoint][0])
            #             bbox_dict['keypoints'].append(bbox_keypoints_dict[keypoint][1])
            #             bbox_dict['keypoints'].append(2)
            #         else:
            #             bbox_dict['keypoints'].append(0)
            #             bbox_dict['keypoints'].append(0)
            #             bbox_dict['keypoints'].append(0)

            coco_annotations.append(bbox_dict)

    return coco_annotations

def process_folder():
    global ann_id

    ann_id = 0
    img_id = 0

    for labelme_json in os.listdir(path):
        if labelme_json.split('.')[-1] == 'json':
            with open(os.path.join(path, labelme_json), 'r', encoding='utf-8') as f:
                labelme = json.load(f)

                # 提取图像元数据
                img_dict = {}
                img_name, img_ext = os.path.splitext(labelme['imagePath'])
                img_name = img_name.replace('/', '_')
                img_dict['file_name'] = img_name + img_ext
                img_dict['height'] = labelme['imageHeight']
                img_dict['width'] = labelme['imageWidth']
                img_dict['id'] = img_id
                coco['images'].append(img_dict)

                # 提取框和关键点信息
                coco_annotations = process_single_json(labelme, img_id)
                coco['annotations'] += coco_annotations

                img_id += 1

                print(labelme_json, '已处理完毕')

    # 写入coco格式的json文件
    with open(os.path.join(Dataset_root, 'val_coco_det.json'), 'w') as f:
        json.dump(coco, f, indent=2)

# 构建coco格式的字典
coco = {}
coco['categories'] = class_list
coco['images'] = []
coco['annotations'] = []

# labelme格式的数据路径
path = os.path.join(Dataset_root, 'labelme_jsons', 'val_labelme_jsons')
process_folder()
