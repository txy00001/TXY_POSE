import orjson as json
import os.path as osp
from copy import deepcopy

import os
scene = ['Magic_show', 'Entertainment', 'ConductMusic', 'Online_class', 
         'TalkShow', 'Speech', 'Fitness', 'Interview', 'Olympic', 'TVShow', 
         'Singing', 'SignLanguage', 'Movie', 'LiveVlog', 'VideoConference']
for key in scene:
    print(key)
    
    with open('/mnt/P40_NFS/20_Research/10_公共数据集/10_Pose/UBody/annotations/'+key+'/keypoint_annotation.json', 'r',encoding='utf-8') as fl:
        lines = fl.readlines()
        print(len(lines), len(lines[0]), lines[0][-100:])
        cat=json.loads(lines[0])
        cat['categories'] = [
                {
                    "supercategory": "person",
                    "id": 1,
                    "name": "person",
                    "keypoints": [
                        "nose",
                        "left_eye",
                        "right_eye",
                        "left_ear",
                        "right_ear",
                        "left_shoulder",
                        "right_shoulder",
                        "left_elbow",
                        "right_elbow",
                        "left_wrist",
                        "right_wrist",

                        'left_hand_root',
                        'left_thumb1',
                        'left_thumb2',
                        'left_thumb3',
                        'left_thumb4',
                        'left_forefinger1',
                        'left_forefinger2',
                        'left_forefinger3',
                        'left_forefinger4',
                        'left_middle_finger1',
                        'left_middle_finger2',
                        'left_middle_finger3',
                        'left_middle_finger4',
                        'left_ring_finger1',
                        'left_ring_finger2',
                        'left_ring_finger3',
                        'left_ring_finger4',
                        'left_pinky_finger1',
                        'left_pinky_finger2',
                        'left_pinky_finger3',
                        'left_pinky_finger4',

                        'right_hand_root',
                        'right_thumb1',
                        'right_thumb2',
                        'right_thumb3',
                        'right_thumb4',
                        'right_forefinger1',
                        'right_forefinger2',
                        'right_forefinger3',
                        'right_forefinger4',
                        'right_middle_finger1',
                        'right_middle_finger2',
                        'right_middle_finger3',
                        'right_middle_finger4',
                        'right_ring_finger1',
                        'right_ring_finger2',
                        'right_ring_finger3',
                        'right_ring_finger4',
                        'right_pinky_finger1',
                        'right_pinky_finger2',
                        'right_pinky_finger3',
                        'right_pinky_finger4',

                        "left_hip",
                        "right_hip",
                        "left_knee",
                        "right_knee",
                        "left_ankle",
                        "right_ankle"
                    ],
                    
                }
            ]
        
    if not osp.exists('/mnt/P40_NFS/20_Research/10_公共数据集/10_Pose/UBody/annotations_change/'+key):
        os.mkdir('/mnt/P40_NFS/20_Research/10_公共数据集/10_Pose/UBody/annotations_change/'+key)
    with open('/mnt/P40_NFS/20_Research/10_公共数据集/10_Pose/UBody/annotations_change/'+key+'/keypoint_annotation.json', 'wb') as fp:
        result = json.dumps(cat)#, fp, ensure_ascii=False)
        fp.write(result)