import torch
from mmpose.apis.inference import dataset_meta_from_config

# 加载原始.pth文件
state_dict = torch.load('/home/txy/code/CastPose/pth/rtmpose/rtmw-x_384.pth', map_location=torch.device('cpu'))

# 获取原始权重矩阵的键列表
keys = list(state_dict.keys())

print(state_dict['state_dict'].keys())

state_dict['meta']['dataset_meta'] = dataset_meta_from_config('/home/txy/code/CastPose/configs/castpose/cocktail3.py', dataset_mode='train')
# print(state_dict['meta'].keys())

# 创建一个新的空的状态字典
new_state_dict = {}

# 保留的通道范围
channel_range = list(range(0, 11)) + list(range(91, 133))

# 遍历原始状态字典的键
for key in state_dict['state_dict'].keys():
    # 获取原始权重矩阵
    weight = state_dict['state_dict'][key]

    # 如果权重矩阵的形状是(133, ...)，则进行裁剪和重新排列
    if isinstance(weight, torch.Tensor) and (133 in weight.shape):
        print(key, weight.shape)
        change_dimension = list(weight.shape).index(133)

        # 裁剪权重矩阵
        if change_dimension == 0:

            # 重新排列保留的通道和对应的权重参数信息
            new_weight = torch.zeros((53, *weight.shape[1:]))
            for i, channel in enumerate(channel_range):
                new_weight[i, ...] = weight[channel, ...]
            weight = new_weight

    # 将裁剪后的权重矩阵添加到新的状态字典中
    new_state_dict[key] = weight

# 将剩余的权重矩阵添加到新的状态字典中
for key in state_dict['state_dict'].keys():
    if key not in new_state_dict:
        new_state_dict[key] = state_dict['state_dict'][key]

state_dict['state_dict'] = new_state_dict

# 保存新的.pth文件
torch.save(state_dict, '/home/txy/code/CastPose/pth/castpose/modified_rtw.pth')
