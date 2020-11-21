from concurrent.futures import ThreadPoolExecutor
import subprocess
from PIL import Image
import cv2
from glob import glob
from tqdm import tqdm
import os, shutil, sys
from video_feature_extract import crop_folder
from time import time
import numpy as np

def process_vid(fn):
    out_file = os.path.join(out_dir,folder,fn + '_img.npz')
    # if os.path.isfile(out_file):
    #     return
    vid_dir = os.path.join(out_dir, folder, fn + '_imgs')
    os.makedirs(vid_dir, exist_ok=True)
    # path = os.path.join(root_dir, folder, fn + '.mp4')
    path = os.path.join(root_dir, fn + '.mp4')
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    out_id = 0
    outs = []
    while True:
        imgs = []
        while success:
          # cv2.imwrite(os.path.join(vid_dir,"frame{:03d}.jpg".format(count)), image)
          imgs.append(image)
          success,image = vidcap.read()
          count += 1
          if count % 1000 == 0:
              break
        images, embeddings = crop_folder(imgs, out_file)
        outs.append(embeddings)
        for i,out_image in enumerate(images):
            out_image = Image.fromarray(out_image)
            out_image.save(os.path.join(vid_dir,"frame{:04d}.jpg".format(out_id)))
            out_id += 1
        if not success:
            break
    vidcap.release()
    embeddings = np.concatenate(outs, 0)
    np.savez(out_file, embeddings)
    

def process(fn):
    process_vid(fn)
        # process_audio(f, folder)
    # elif f.endswith('npz'):
        # shutil.copyfile(f, os.path.join(out_dir, folder, f.split('/')[-1]))

FNULL = open(os.devnull, 'w')
# root_dir = "/usr/cs/public/mohd/2020-1/"
root_dir = "/usr/cs/public/mohd/MOSEI/Videos/Full/Combined"
out_dir = "/usr/cs/public/mohd/data"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir,'tmp'), exist_ok=True)

def main(folder):
    os.makedirs(os.path.join(out_dir, folder), exist_ok = True)
    if len(sys.argv) > 2:
        files = [line.strip() for line in open('/usr/cs/public/mohd/%s_data.txt'%folder)][int(sys.argv[2]):]
    else:
        files = [line.strip() for line in open('/usr/cs/public/mohd/%s_data.txt'%folder)]
    n = len(files)
    with ThreadPoolExecutor(3) as threads:
        for fn in tqdm(threads.map(process,files), total = n):
            pass
    # for fn in tqdm(files, total = n):
    #     process(fn)
# process(files[0])

if __name__ == '__main__':
    folder = sys.argv[1]
    if folder == 'all':
        for folder in ['train', 'val', 'test']:
            main(folder)
    else:
        main(folder)
