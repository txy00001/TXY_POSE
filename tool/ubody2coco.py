import argparse
import os
from copy import deepcopy
from multiprocessing import Pool

import mmengine
import numpy as np
from pycocotools.coco import COCO


def split_dataset(video_root: str, annotation_path: str, split_path: str):
    folders = os.listdir(video_root)
    splits = np.load(split_path)
    train_annos = []
    val_annos = []
    train_imgs = []
    val_imgs = []
    t_id = 0
    v_id = 0
    categories = [{'supercategory': 'person', 'id': 1, 'name': 'person'}]

    for scene in folders:
        scene_train_anns = []
        scene_val_anns = []
        scene_train_imgs = []
        scene_val_imgs = []
        image_path = os.path.join(video_root, scene)  # 图片路径
        data = COCO(os.path.join(annotation_path, scene, 'keypoint_annotation.json'))
        print(f'Processing {scene}.........')
        progress_bar = mmengine.ProgressBar(len(data.anns.keys()))
        for aid in data.anns.keys():
            ann = data.anns[aid]
            img = data.loadImgs(ann['image_id'])[0]

            if img['file_name'].startswith('/'):
                file_name = img['file_name'][1:]  # [1:] means delete '/'
            else:
                file_name = img['file_name']
            video_name = file_name.split('/')[-2]
            if 'Trim' in video_name:
                video_name = video_name.split('_Trim')[0]

            img_path = os.path.join(image_path, file_name)
            if not os.path.exists(img_path):
                progress_bar.update()
                continue

            img['file_name'] = os.path.join(scene, file_name)
            ann_ = deepcopy(ann)
            img_ = deepcopy(img)
            if video_name in splits:
                scene_val_anns.append(ann_)
                scene_val_imgs.append(img_)
                ann_['id'] = v_id
                ann_['image_id'] = v_id
                img_['id'] = v_id
                val_annos.append(ann_)
                val_imgs.append(img_)
                v_id += 1
            else:
                scene_train_anns.append(ann_)
                scene_train_imgs.append(img_)
                ann_['id'] = t_id
                ann_['image_id'] = t_id
                img_['id'] = t_id
                train_annos.append(ann_)
                train_imgs.append(img_)
                t_id += 1

            progress_bar.update()

        scene_train_data = dict(
            images=scene_train_imgs,
            annotations=scene_train_anns,
            categories=categories)
        scene_val_data = dict(
            images=scene_val_imgs,
            annotations=scene_val_anns,
            categories=categories)

        mmengine.dump(
            scene_train_data,
            os.path.join(annotation_path, scene, 'train_annotations.json'))
        mmengine.dump(
            scene_val_data,
            os.path.join(annotation_path, scene, 'val_annotations.json'))

    train_data = dict(
        images=train_imgs, annotations=train_annos, categories=categories)
    val_data = dict(
        images=val_imgs, annotations=val_annos, categories=categories)

    mmengine.dump(train_data,
                  os.path.join(annotation_path, 'train_annotations.json'))
    mmengine.dump(val_data,
                  os.path.join(annotation_path, 'val_annotations.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/mnt/P40_NFS/20_Research/10_公共数据集/10_Pose/UBody')
    args = parser.parse_args()
    video_root = f'{args.data_root}/images_new'
    split_path = f'{args.data_root}/splits/intra_scene_test_list.npy'
    annotation_path = f'{args.data_root}/annotations_new'

    split_dataset(video_root, annotation_path, split_path)
