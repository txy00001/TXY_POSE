
# 关键点检测数据集
dataset_info = dict(
        dataset_name='qx_castpose_data',
        paper_info=dict(
        author='tangxiyu',
        title='QX_Keypoint Detection'),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),

        11:
        dict(
            name='left_hand_root',
            id=11,
            color=[255, 255, 255],
            type='',
            swap='right_hand_root'),
        12:
        dict(
            name='left_thumb1',
            id=12,
            color=[255, 128, 0],
            type='',
            swap='right_thumb1'),
        13:
        dict(
            name='left_thumb2',
            id=13,
            color=[255, 128, 0],
            type='',
            swap='right_thumb2'),
        14:
        dict(
            name='left_thumb3',
            id=14,
            color=[255, 128, 0],
            type='',
            swap='right_thumb3'),
        15:
        dict(
            name='left_thumb4',
            id=15,
            color=[255, 128, 0],
            type='',
            swap='right_thumb4'),
        16:
        dict(
            name='left_forefinger1',
            id=16,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger1'),
        17:
        dict(
            name='left_forefinger2',
            id=17,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger2'),
        18:
        dict(
            name='left_forefinger3',
            id=18,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger3'),
        19:
        dict(
            name='left_forefinger4',
            id=19,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger4'),
        20:
        dict(
            name='left_middle_finger1',
            id=20,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger1'),
        21:
        dict(
            name='left_middle_finger2',
            id=21,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger2'),
        22:
        dict(
            name='left_middle_finger3',
            id=22,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger3'),
        23:
        dict(
            name='left_middle_finger4',
            id=23,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger4'),
        24:
        dict(
            name='left_ring_finger1',
            id=24,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger1'),
        25:
        dict(
            name='left_ring_finger2',
            id=25,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger2'),
        26:
        dict(
            name='left_ring_finger3',
            id=26,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger3'),
        27:
        dict(
            name='left_ring_finger4',
            id=27,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger4'),
        28:
        dict(
            name='left_pinky_finger1',
            id=28,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger1'),
        29:
        dict(
            name='left_pinky_finger2',
            id=29,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger2'),
        30:
        dict(
            name='left_pinky_finger3',
            id=30,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger3'),
        31:
        dict(
            name='left_pinky_finger4',
            id=31,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger4'),
        32:
        dict(
            name='right_hand_root',
            id=32,
            color=[255, 255, 255],
            type='',
            swap='left_hand_root'),
        33:
        dict(
            name='right_thumb1',
            id=33,
            color=[255, 128, 0],
            type='',
            swap='left_thumb1'),
        34:
        dict(
            name='right_thumb2',
            id=34,
            color=[255, 128, 0],
            type='',
            swap='left_thumb2'),
        35:
        dict(
            name='right_thumb3',
            id=35,
            color=[255, 128, 0],
            type='',
            swap='left_thumb3'),
        36:
        dict(
            name='right_thumb4',
            id=36,
            color=[255, 128, 0],
            type='',
            swap='left_thumb4'),
        37:
        dict(
            name='right_forefinger1',
            id=37,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger1'),
        38:
        dict(
            name='right_forefinger2',
            id=38,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger2'),
        39:
        dict(
            name='right_forefinger3',
            id=39,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger3'),
        40:
        dict(
            name='right_forefinger4',
            id=40,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger4'),
        41:
        dict(
            name='right_middle_finger1',
            id=41,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger1'),
        42:
        dict(
            name='right_middle_finger2',
            id=42,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger2'),
        43:
        dict(
            name='right_middle_finger3',
            id=43,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger3'),
        44:
        dict(
            name='right_middle_finger4',
            id=44,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger4'),
        45:
        dict(
            name='right_ring_finger1',
            id=45,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger1'),
        46:
        dict(
            name='right_ring_finger2',
            id=46,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger2'),
        47:
        dict(
            name='right_ring_finger3',
            id=47,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger3'),
        48:
        dict(
            name='right_ring_finger4',
            id=48,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger4'),
        49:
        dict(
            name='right_pinky_finger1',
            id=49,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger1'),
        50:
        dict(
            name='right_pinky_finger2',
            id=50,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger2'),
        51:
        dict(
            name='right_pinky_finger3',
            id=51,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger3'),
        52:
        dict(
            name='right_pinky_finger4',
            id=52,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger4')
    },
    skeleton_info={
        0:
            dict(link=('left_shoulder', 'right_shoulder'),id=0,color=[51, 153, 255]),
        1:
            dict(link=('left_shoulder', 'left_elbow'), id=1, color=[0, 255, 0]),
        2:
            dict(
                link=('right_shoulder', 'right_elbow'), id=2, color=[255, 128, 0]),
        3:
            dict(link=('left_elbow', 'left_wrist'), id=3, color=[0, 255, 0]),
        4:
            dict(link=('right_elbow', 'right_wrist'), id=4, color=[255, 128, 0]),
        5:
            dict(link=('left_eye', 'right_eye'), id=5, color=[51, 153, 255]),
        6:
            dict(link=('nose', 'left_eye'), id=6, color=[51, 153, 255]),
        7:
            dict(link=('nose', 'right_eye'), id=7, color=[51, 153, 255]),
        8:
            dict(link=('left_eye', 'left_ear'), id=8, color=[51, 153, 255]),
        9:
            dict(link=('right_eye', 'right_ear'), id=9, color=[51, 153, 255]),
        10:
            dict(link=('left_ear', 'left_shoulder'), id=10, color=[51, 153, 255]),
        11:
            dict(
                link=('right_ear', 'right_shoulder'), id=11, color=[51, 153, 255]),
        12:
        dict(
            link=('left_hand_root', 'left_thumb1'), id=12, color=[255, 128,
                                                                  0]),
        13:
        dict(link=('left_thumb1', 'left_thumb2'), id=13, color=[255, 128, 0]),
        14:
        dict(link=('left_thumb2', 'left_thumb3'), id=14, color=[255, 128, 0]),
        15:
        dict(link=('left_thumb3', 'left_thumb4'), id=15, color=[255, 128, 0]),
        16:
        dict(
            link=('left_hand_root', 'left_forefinger1'),
            id=16,
            color=[255, 153, 255]),
        17:
        dict(
            link=('left_forefinger1', 'left_forefinger2'),
            id=17,
            color=[255, 153, 255]),
        18:
        dict(
            link=('left_forefinger2', 'left_forefinger3'),
            id=18,
            color=[255, 153, 255]),
        19:
        dict(
            link=('left_forefinger3', 'left_forefinger4'),
            id=19,
            color=[255, 153, 255]),
        20:
        dict(
            link=('left_hand_root', 'left_middle_finger1'),
            id=20,
            color=[102, 178, 255]),
        21:
        dict(
            link=('left_middle_finger1', 'left_middle_finger2'),
            id=21,
            color=[102, 178, 255]),
        22:
        dict(
            link=('left_middle_finger2', 'left_middle_finger3'),
            id=22,
            color=[102, 178, 255]),
        23:
        dict(
            link=('left_middle_finger3', 'left_middle_finger4'),
            id=23,
            color=[102, 178, 255]),
        24:
        dict(
            link=('left_hand_root', 'left_ring_finger1'),
            id=24,
            color=[255, 51, 51]),
        25:
        dict(
            link=('left_ring_finger1', 'left_ring_finger2'),
            id=25,
            color=[255, 51, 51]),
        26:
        dict(
            link=('left_ring_finger2', 'left_ring_finger3'),
            id=26,
            color=[255, 51, 51]),
        27:
        dict(
            link=('left_ring_finger3', 'left_ring_finger4'),
            id=27,
            color=[255, 51, 51]),
        28:
        dict(
            link=('left_hand_root', 'left_pinky_finger1'),
            id=28,
            color=[0, 255, 0]),
        29:
        dict(
            link=('left_pinky_finger1', 'left_pinky_finger2'),
            id=29,
            color=[0, 255, 0]),
        30:
        dict(
            link=('left_pinky_finger2', 'left_pinky_finger3'),
            id=30,
            color=[0, 255, 0]),
        31:
        dict(
            link=('left_pinky_finger3', 'left_pinky_finger4'),
            id=31,
            color=[0, 255, 0]),
        32:
        dict(
            link=('right_hand_root', 'right_thumb1'),
            id=32,
            color=[255, 128, 0]),
        33:
        dict(
            link=('right_thumb1', 'right_thumb2'), id=33, color=[255, 128, 0]),
        34:
        dict(
            link=('right_thumb2', 'right_thumb3'), id=34, color=[255, 128, 0]),
        35:
        dict(
            link=('right_thumb3', 'right_thumb4'), id=35, color=[255, 128, 0]),
        36:
        dict(
            link=('right_hand_root', 'right_forefinger1'),
            id=36,
            color=[255, 153, 255]),
        37:
        dict(
            link=('right_forefinger1', 'right_forefinger2'),
            id=37,
            color=[255, 153, 255]),
        38:
        dict(
            link=('right_forefinger2', 'right_forefinger3'),
            id=38,
            color=[255, 153, 255]),
        39:
        dict(
            link=('right_forefinger3', 'right_forefinger4'),
            id=39,
            color=[255, 153, 255]),
        40:
        dict(
            link=('right_hand_root', 'right_middle_finger1'),
            id=40,
            color=[102, 178, 255]),
        41:
        dict(
            link=('right_middle_finger1', 'right_middle_finger2'),
            id=41,
            color=[102, 178, 255]),
        42:
        dict(
            link=('right_middle_finger2', 'right_middle_finger3'),
            id=42,
            color=[102, 178, 255]),
        43:
        dict(
            link=('right_middle_finger3', 'right_middle_finger4'),
            id=43,
            color=[102, 178, 255]),
        44:
        dict(
            link=('right_hand_root', 'right_ring_finger1'),
            id=44,
            color=[255, 51, 51]),
        45:
        dict(
            link=('right_ring_finger1', 'right_ring_finger2'),
            id=45,
            color=[255, 51, 51]),
        46:
        dict(
            link=('right_ring_finger2', 'right_ring_finger3'),
            id=46,
            color=[255, 51, 51]),
        47:
        dict(
            link=('right_ring_finger3', 'right_ring_finger4'),
            id=47,
            color=[255, 51, 51]),
        48:
        dict(
            link=('right_hand_root', 'right_pinky_finger1'),
            id=61,
            color=[0, 255, 0]),
        49:
        dict(
            link=('right_pinky_finger1', 'right_pinky_finger2'),
            id=49,
            color=[0, 255, 0]),
        50:
        dict(
            link=('right_pinky_finger2', 'right_pinky_finger3'),
            id=50,
            color=[0, 255, 0]),
        51:
        dict(
            link=('right_pinky_finger3', 'right_pinky_finger4'),
            id=51,
            color=[0, 255, 0])
    },
    joint_weights=[1.]*53,
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018,
        0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019, 0.022,
        0.031,0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018,
        0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019, 0.022,
        0.031
    ])

