import os
import argparse
from multiprocessing import Pool

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

def convert(video_path:str, output_folder:str):
    video_name = video_path.split('/')[-1]
    image_path = video_path.replace(video_name, video_name.split('.')[0])
    image_path = image_path.replace('/videos/', '/images/')
    output_image_path = image_path.replace(args.video_folder, output_folder, 1)
    os.makedirs(output_image_path, exist_ok=True)
    os.system(f'ffmpeg -i {video_path} -f image2 -r 30 -b:v 5626k {output_image_path}/%06d.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, default='/mnt/P40_NFS/20_Research/10_公共数据集/10_Pose/UBody/videos')
    parser.add_argument('--output_folder', type=str, default='/mnt/P40_NFS/20_Research/10_公共数据集/10_Pose/UBody/images')
    args = parser.parse_args()
    video_paths = findAllFile(args.video_folder)
    pool = Pool(processes=8)
    for video_path in video_paths:
        pool.apply_async(convert, (video_path, args.output_folder))
    pool.close()
    pool.join()