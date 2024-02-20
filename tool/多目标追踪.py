import mmcv
import tempfile
from mmtrack.apis import inference_mot, init_model

# 输入输出视频路径
input_video = 'QX_Labelme_dataset/mot_people_short.mp4'
output = 'outputs/F3_MOT_people_short.mp4'

# 指定 config 配置文件 和 模型权重文件，创建模型
mot_config = './configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
mot_checkpoint = 'https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth'
# 初始化多目标追踪模型
mot_model = init_model(mot_config, mot_checkpoint, device='cuda:0')

# 读入待预测视频
imgs = mmcv.VideoReader(input_video)
prog_bar = mmcv.ProgressBar(len(imgs))
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name
# 逐帧输入模型预测
for i, img in enumerate(imgs):
    result = inference_mot(mot_model, img, frame_id=i)

    mot_model.show_result(
        img,
        result,
        show=False,
        wait_time=int(1000. / imgs.fps),
        out_file=f'{out_path}/{i:06d}.jpg')
    prog_bar.update()

print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
out_dir.cleanup()